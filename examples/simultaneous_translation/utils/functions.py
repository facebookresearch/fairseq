import torch
def exclusive_cumprod(tensor, dim: int, eps: float = 1e-10):
    """
    Implementing exclusive cumprod.
    There is cumprod in pytorch, however there is no exclusive mode.
    cumprod(x) = [x1, x1x2, x2x3x4, ..., prod_{i=1}^n x_i]
    exclusive means cumprod(x) = [1, x1, x1x2, x1x2x3, ..., prod_{i=1}^{n-1} x_i]
    """
    tensor_size = list(tensor.size())
    tensor_size[dim] = 1
    return_tensor = safe_cumprod(
        torch.cat([torch.ones(tensor_size).type_as(tensor), tensor], dim=dim), dim=dim, eps=eps
    )

    if dim == 0:
        return return_tensor[:-1]
    elif dim == 1:
        return return_tensor[:, :-1]
    elif dim == 2:
        return return_tensor[:, :, :-1]
    else:
        raise RuntimeError("Cumprod on dimension 3 and more is not implemented")

def safe_cumprod(tensor, dim: int, eps: float = 1e-10):
    """
    An implementation of cumprod to prevent precision issue.
    cumprod(x)
    = [x1, x1x2, x1x2x3, ....]
    = [exp(log(x1)), exp(log(x1) + log(x2)), exp(log(x1) + log(x2) + log(x3)), ...]
    = exp(cumsum(log(x)))
    """

    if (tensor + eps < 0).any().item():
        raise RuntimeError(
            "Safe cumprod can only take non-negative tensors as input."
            "Consider use torch.cumprod if you want to calculate negative values."
        )

    log_tensor = torch.log(tensor + eps)
    cumsum_log_tensor = torch.cumsum(log_tensor, dim)
    exp_cumsum_log_tensor = torch.exp(cumsum_log_tensor)
    return exp_cumsum_log_tensor