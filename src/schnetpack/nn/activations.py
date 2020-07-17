import numpy as np
from torch.nn import functional


def shifted_softplus(x):
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    """
    return functional.softplus(x) - np.log(2.0)


def swish(x, beta=1):
    """Compute swish activation function

    Args:
        x (torch.Tensor): input tensor.
        beta (int): beta value of swish

    Returns:
        torch.Tensor: swish of input.

    """
    return x * functional.sigmoid(beta * x)
