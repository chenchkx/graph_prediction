
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from dgl.ops import segment
from torch_geometric.utils import to_dense_batch


# Fast Matrix Power Normalized SPD Matrix Function
class FastMPNSPDMatrixFunction(Function):
    @staticmethod
    def forward(ctx, input, iterN):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        normA = (1.0 / 3.0) * x.mul(I3).sum(dim=1).sum(dim=1)
        A = x.div(normA.view(batchSize, 1, 1).expand_as(x))
        Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad=False, device=x.device).type(dtype)
        Z = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, iterN, 1, 1).type(dtype)
        if iterN < 2:
            ZY = 0.5 * (I3 - A)
            YZY = A.bmm(ZY)
        else:
            ZY = 0.5 * (I3 - A)
            Y[:, 0, :, :] = A.bmm(ZY)
            Z[:, 0, :, :] = ZY
            for i in range(1, iterN - 1):
                ZY = 0.5 * (I3 - Z[:, i - 1, :, :].bmm(Y[:, i - 1, :, :]))
                Y[:, i, :, :] = Y[:, i - 1, :, :].bmm(ZY)
                Z[:, i, :, :] = ZY.bmm(Z[:, i - 1, :, :])
            YZY = 0.5 * Y[:, iterN - 2, :, :].bmm(I3 - Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :]))
        y = YZY * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        ctx.save_for_backward(input, A, YZY, normA, Y, Z)
        ctx.iterN = iterN
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, A, ZY, normA, Y, Z = ctx.saved_tensors
        iterN = ctx.iterN
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        der_postCom = grad_output * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        der_postComAux = (grad_output * ZY).sum(dim=1).sum(dim=1).div(2 * torch.sqrt(normA))
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        if iterN < 2:
            der_NSiter = 0.5 * (der_postCom.bmm(I3 - A) - A.bmm(der_postCom))
        else:
            dldY = 0.5 * (der_postCom.bmm(I3 - Y[:, iterN - 2, :, :].bmm(Z[:, iterN - 2, :, :])) -
                          Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :]).bmm(der_postCom))
            dldZ = -0.5 * Y[:, iterN - 2, :, :].bmm(der_postCom).bmm(Y[:, iterN - 2, :, :])
            for i in range(iterN - 3, -1, -1):
                YZ = I3 - Y[:, i, :, :].bmm(Z[:, i, :, :])
                ZY = Z[:, i, :, :].bmm(Y[:, i, :, :])
                dldY_ = 0.5 * (dldY.bmm(YZ) -
                               Z[:, i, :, :].bmm(dldZ).bmm(Z[:, i, :, :]) -
                               ZY.bmm(dldY))
                dldZ_ = 0.5 * (YZ.bmm(dldZ) -
                               Y[:, i, :, :].bmm(dldY).bmm(Y[:, i, :, :]) -
                               dldZ.bmm(ZY))
                dldY = dldY_
                dldZ = dldZ_
            der_NSiter = 0.5 * (dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
        der_NSiter = der_NSiter.transpose(1, 2)
        grad_input = der_NSiter.div(normA.view(batchSize, 1, 1).expand_as(x))
        grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
        for i in range(batchSize):
            grad_input[i, :, :] += (der_postComAux[i] \
                                    - grad_aux[i] / (normA[i] * normA[i])) \
                                   * torch.ones(dim, device=x.device).diag().type(dtype)
        return grad_input, None


class DKEPooling(nn.Module):
    def __init__(self, iterN = 5, snr_value = 15, min_power=1e-10):
        super(DKEPooling, self).__init__()

        self.noise_weight = nn.Parameter(torch.Tensor(1))
        self.noise_bias = nn.Parameter(torch.Tensor(1))
        self.noise_weight.data.fill_(1)
        self.noise_bias.data.fill_(0)

        self.register_buffer('snr_value', torch.tensor(snr_value))
        self.register_buffer('min_power', torch.tensor(min_power))
        self.register_buffer('iterN', torch.tensor(iterN))

    def to_batch_tensors(self, tensor, batch_list, batch_indx):
        tensor_means = torch.mean(segment.segment_reduce(batch_list, tensor, reducer='mean'), 1)
        batch_tensor, batch_mask = to_dense_batch(tensor, batch_indx)
        batch_tensor = batch_tensor-tensor_means.reshape(tensor_means.shape[0],1,1)
        tensor_ = batch_tensor[batch_mask]
        batch_tensor_, _ = to_dense_batch(tensor_, batch_indx)
        batch_tensor_var = torch.sum(batch_tensor_*batch_tensor_, dim=(1,2))/(batch_list*tensor.shape[1])
        return batch_tensor_var, batch_tensor_,  batch_mask

    def gauss_perturbation(self, tensor, batch_list, batch_indx, snr = 15):
        noise = torch.randn(tensor.shape[0], tensor.shape[1]).to(tensor.device)
        batch_noise_var, batch_noise_, batch_mask = self.to_batch_tensors(noise, batch_list, batch_indx)
        batch_tensor_var, _, _ = self.to_batch_tensors(tensor, batch_list, batch_indx) 
        batch_var_snr = batch_tensor_var /(10**(snr / 10))
        noise_snr = batch_noise_*(torch.sqrt(batch_var_snr) / torch.sqrt(batch_noise_var)).reshape(batch_noise_.shape[0],1,1)
        noise_snr = noise_snr[batch_mask]
        return tensor + noise_snr

    def batch_gaussperturbation(self, tensor, snr = 15):
        noise = torch.randn(tensor.shape[0], tensor.shape[1]).to(tensor.device)
        noise = noise - torch.mean(noise)
        signal_power = torch.norm(tensor - tensor.mean()) ** 2 / (tensor.shape[0]*tensor.shape[1])  # signal power
        noise_variance = signal_power /(10**(snr / 10))   # noise power
        noise = (torch.sqrt(noise_variance) / torch.std(noise)) * noise
        signal_noise = noise + tensor
        return signal_noise

    def batch_perturbation(self, batch_cov, snr_value):
        assert batch_cov.shape[1] == batch_cov.shape[2]
        noise = torch.randn(batch_cov.size()).to(batch_cov.device)
        batch_noise = noise.bmm(noise.transpose(1,2))/batch_cov.shape[1]
        batch_noise = self.noise_weight*batch_noise+self.noise_bias
        signal_power= batch_cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)/(batch_cov.shape[1]*batch_cov.shape[2]) # signal power
        signal_power[torch.where(signal_power<=self.min_power)] = self.min_power
        noise_power = signal_power /(10**(snr_value / 10))   # noise power
        noise_power = torch.sqrt(noise_power)/torch.std(noise, dim=[1,2])
        noise_power = noise_power.detach()
        return batch_noise*noise_power.reshape(batch_cov.shape[0],1,1)

    def forward(self, batch_list, feat):
        
        batch_indx = torch.arange(len(batch_list)).to(feat.device).repeat_interleave(batch_list) 
        feat = self.gauss_perturbation(feat, batch_list, batch_indx, snr = 15)
        batch_mean = segment.segment_reduce(batch_list, feat, reducer='mean')
        feat_mean = batch_mean[batch_indx]
        ### Function torch.repeat_interleave() is faster. But it makes seed fail, the result is not reproducible.
        # feat_mean = torch.repeat_interleave(batch_mean, batch_list, dim=0, output_size = feat.shape[0])
        feat_diff = feat - feat_mean
        batch_diff, _ = to_dense_batch(feat_diff, batch_indx)
        batch_cov = batch_diff.transpose(1, 2).bmm(batch_diff)
        batch_cov = batch_cov/((batch_list-1).reshape(batch_list.shape[0],1,1))
        # batch_cov = batch_cov + self.batch_perturbation(batch_cov, self.snr_value)
        batch_cov = FastMPNSPDMatrixFunction.apply(batch_cov, self.iterN)

        return batch_cov.bmm(batch_mean.unsqueeze(2)).squeeze(2)

