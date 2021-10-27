"""
Generic random sampling methods.

Fuctions
--------
random_normal(loc, scale, shape, device)
    Random number generator for a normal distribution with mean `loc`
    and standard deviation `scale`.

random_uniform(low, high, shape, device)
    Uniform random number generator where the values are over the range
    [low, high).
"""
import torch
from typing import Optional, Tuple

def random_normal(
    loc    : float, 
    scale  : float, 
    shape  : Tuple[int], 
    device : Optional[torch.device] = None
) -> torch.Tensor:
    """
    Random number generator for a normal distribution with mean `loc`
    and standard deviation `scale`.

    Parameters
    ----------
    loc : float
        The mean of the distribution.

    scale : float
        The standard deviation of the distribution.

    shape : tuple[int]
        The shape of the output tensor.

    device : torch.device, optional
        The device to generate the values on. Default is None.

    Returns
    -------
    sample : Tensor
        The random sample.

    Examples
    --------
    import matplotlib.pyplot as plt
    from galsyn.random.sample import random_normal

    sample = random_normal(loc=85, scale=5, shape=(10000,1))

    fig, ax = plt.subplots()
    ax.hist(sample.flatten().numpy())
    fig.show()
    """
    return torch.randn(shape, device=device) * scale + loc

def random_uniform(
    low    : float, 
    high   : float, 
    shape  : Tuple[int], 
    device : Optional[torch.device] = None
) -> torch.Tensor:
    """
    Uniform random number generator where the values are over the range
    [low, high).

    Parameters
    ----------
    low : float
        The lower bound of the range.

    high : float
        The upper bound of the range.

    shape : tuple[int]
        The shape of the output tensor.

    device : torch.device, optional
        The device to generate the values on. Default is None.

    Returns
    -------
    sample : torch.Tensor
        The random sample.

    Examples
    --------
    import matplotlib.pyplot as plt
    from galsyn.random.sample import random_uniform

    sample = random_uniform(low=0, high=100, shape=(10000,1))

    fig, ax = plt.subplots()
    ax.hist(sample.flatten().numpy())
    fig.show()
    """
    return torch.rand(shape, device=device) * (high - low) + low