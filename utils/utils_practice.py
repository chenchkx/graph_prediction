
import torch

def batch_tensor_trace(batch_tensor):
    assert batch_tensor.shape[1] == batch_tensor.shape[2]
    batch_trace = batch_tensor.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    batch_trace = torch.reshape(batch_trace, (batch_trace.shape[0], 1, 1))
    return batch_trace



