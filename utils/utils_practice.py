

import torch
from dgl.ops import segment

def repeat_tensor_interleave(tensor, num_list):
    data_index = torch.arange(len(num_list)).to(tensor.device).repeat_interleave(num_list)
    return tensor[data_index]

def batch_tensor_trace(batch_tensor):
    assert batch_tensor.shape[1] == batch_tensor.shape[2]
    batch_trace = batch_tensor.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    batch_trace = torch.reshape(batch_trace, (batch_trace.shape[0], 1, 1))
    return batch_trace

def to_batch_tensor(tensor, num_nodes, batch_index, fill_value=0):
    if batch_index is None:
        batch_index = torch.arange(len(num_nodes)).to(num_nodes.device).repeat_interleave(num_nodes) 
    max_num_nodes = int(num_nodes.max())
    batch_size = int(batch_index.max()) + 1
    cum_nodes = torch.cat([batch_index.new_zeros(1), num_nodes.cumsum(dim=0)])
    
    idx = torch.arange(batch_index.size(0), dtype=torch.long, device=tensor.device)
    idx = (idx - cum_nodes[batch_index]) + (batch_index * max_num_nodes)
    size = [batch_size * max_num_nodes] + list(tensor.size())[1:]
    out = tensor.new_full(size, fill_value)
    out[idx] = tensor
    out = out.view([batch_size, max_num_nodes] + list(tensor.size())[1:])

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool, device=tensor.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)
    return out, mask

def batch_tensor_var_mask(tensor, num_nodes, batch_index=None):
    if batch_index is None:
        batch_index = torch.arange(len(num_nodes)).to(num_nodes.device).repeat_interleave(num_nodes) 
    tensor_means = torch.mean(segment.segment_reduce(num_nodes, tensor, reducer='mean'), 1)
    batch_tensor, batch_mask = to_batch_tensor(tensor, num_nodes, batch_index)
    batch_tensor = batch_tensor-tensor_means.reshape(tensor_means.shape[0],1,1)
    batch_tensor[~batch_mask] = 0
    batch_tensor_var = torch.sum(batch_tensor*batch_tensor, dim=(1,2))/(num_nodes*tensor.shape[1])
    return batch_tensor, batch_tensor_var, batch_mask

