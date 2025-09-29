
import torch


def norm(data:torch.Tensor, attn_mask:torch.Tensor=None) -> torch.Tensor:
    """
    Zero-Normalize data to have mean=0 and standard_deviation=1

    Parameters
    ----------
    data, attn_mask: (B, C, H, W)
    """
    if attn_mask is None:
        mean = torch.mean(data, dim=-1, keepdim=True)
        var = torch.var(data, dim=-1, keepdim=True)
        
        return (data - mean) / (var + 1e-9)**0.5
    else:
        mean = torch.sum(data, dim=-1, keepdim=True) / (torch.sum(attn_mask, dim=-1, keepdim=True) + 1e-9)

        # aux_mask matrix is introduced to make sure that the data (zero) padding is not effected by the norm operations
        # i.e. zero entries remain untouched after the norm operations 
        aux_mask = ((1 - attn_mask) * mean)
        var = torch.sum((data - mean + aux_mask)**2, dim=-1, keepdim=True) / (torch.sum(attn_mask, dim=-1, keepdim=True) + 1e-9)
        
        return (data - mean + aux_mask) / (var + 1e-9)**0.5

def ncc(data_0:torch.Tensor, data_1:torch.Tensor, attn_mask:torch.Tensor=None, keep_batch:bool=False) -> torch.Tensor:
    """
    Zero-Normalized cross-correlation coefficient between two data sets

    Zero-Normalized cross-correlation equals the cosine of the angle between the unit vectors F and T, 
    being thus 1 if and only if F equals T multiplied by a positive scalar. 

    Parameters
    ----------
    data_0, data_1, attn_mask : (B, C, H, W)
    """
    nb_of_signals = 1
    if attn_mask is None:
        for dim in range(data_0.dim() - 1): # all but the last dimension (which is the actual signal)
            # (B)
            nb_of_signals = nb_of_signals * data_0.shape[dim]

        # (B, C, H)
        cross_corrs = torch.sum(norm(data=data_0) * norm(data=data_1), dim=-1) / (data_0.shape[-1] + 1e-9)
    else:
        for dim in range(data_0.dim() - 2):
            # (B)
            nb_of_signals = nb_of_signals * data_0.shape[dim]

        # check whether a channel is padding (only zero elements) or signal (at least one non-zero element)
        # (B, C)
        nb_of_channels = torch.max(attn_mask, dim=-1)[0].sum(-1)
        nb_of_signals = nb_of_signals * nb_of_channels
        # (B, C, H)
        nb_of_signals = nb_of_signals.unsqueeze(-1)

        # (B, C, H)
        cross_corrs = torch.sum(norm(data=data_0, attn_mask=attn_mask) * norm(data=data_1, attn_mask=attn_mask), dim=-1) / (torch.sum(attn_mask, dim=-1) + 1e-9)

    if keep_batch == True:
        # compute ncc of each sample
        nb_of_signals = nb_of_signals / cross_corrs.shape[0]
        # (B, C, H)
        ncc = cross_corrs / nb_of_signals
        # (B)
        ncc = ncc.flatten(1).sum(-1)
    else:
        # compute ncc of the entire batch
        ncc = (cross_corrs / nb_of_signals).sum()

    return ncc

def masked_mean(tensor, attn_mask=None, dim=None, keep_dim=False):
    """
        Determine the mean while considering the attention mask

        tensor: (B, N, D)
        attn_mask: (B, N)
    """
    if attn_mask is None:
        if dim is None:
            return torch.mean(tensor)
        else:
            return torch.mean(tensor, dim=dim, keepdim=keep_dim)
    else:
        tensor_masked = tensor * attn_mask.unsqueeze(-1)
        if dim is None:
            return torch.sum(tensor_masked) / (torch.sum(attn_mask.unsqueeze(-1).repeat(1, 1, tensor.shape[-1])) + 1e-9)
        else:
            numerator = torch.sum(tensor_masked, dim=dim, keepdim=keep_dim)
            denominator = torch.sum(attn_mask.unsqueeze(-1).repeat(1, 1, tensor.shape[-1]), dim=dim, keepdim=keep_dim) + 1e-9
            return numerator / denominator

def masked_var(tensor, attn_mask=None, dim=None, keep_dim=False):
    """
        Determine the variance while considering the attention mask

        tensor: (B, N, D)
        attn_mask: (B, N)
    """
    if attn_mask is None:
        if dim is None:
            return torch.var(tensor)
        else:
            return torch.var(tensor, dim=dim, keepdim=keep_dim)
    else:
        tensor_masked = tensor * attn_mask.unsqueeze(-1)
        deviations = tensor_masked - masked_mean(tensor, attn_mask, dim=dim, keep_dim=True)
        deviations = deviations * attn_mask.unsqueeze(-1)
        squared_deviations = (deviations ** 2) * attn_mask.unsqueeze(-1)
        if dim is None:
            return torch.sum(squared_deviations) / (torch.sum(attn_mask.unsqueeze(-1).repeat(1, 1, tensor.shape[-1])) - 1 + 1e-9)
        else:
            return torch.sum(squared_deviations, dim=dim, keepdim=keep_dim) / (torch.sum(attn_mask.unsqueeze(-1).repeat(1, 1, tensor.shape[-1]), dim=dim, keepdim=keep_dim) - 1 + 1e-9)