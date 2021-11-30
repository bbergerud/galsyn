"""
Methods relating to the functional form of a logarithmic spiral arm

θ_logarithmic(u, α, φ, u0, n, method)
    Calculates the azimuthal angle of the logarithmic spiral arm
    at the given radial coordinates. If `α` is callable, then a
    numerical integration is performed for several points over the
    range and an interpolation performed to calculate the value at
    all radial distances.
"""

import torch
import torchquad
from typing import Optional, Union
from .interpolate import interp1d

if torch.cuda.is_available() and torch.cuda.current_device():
    torchquad.enable_cuda()

def θ_logarithmic(
    α      : Union[callable, float, torch.Tensor], 
    φ      : Union[float, torch.Tensor], 
    r      : Optional[torch.Tensor] = None, 
    r0     : Optional[Union[float, torch.Tensor]] = None,
    u      : Optional[torch.Tensor] = None, 
    u0     : Optional[Union[float, torch.Tensor]] = None,
    n      : int = 15, 
    method : callable = torchquad.Trapezoid(),
    eps    : float = 1e-7,
) -> torch.Tensor:
    """
    Calculates the azimuthal angle of the logarithmic spiral arm
    at the given radial coordinates. If `α` is callable, then a
    numerical integration is performed for several points over the
    range and an interpolation performed to calculate the value at
    all radial distances.

    Parameters
    ----------
    α : callable, float, Tensor
        The pitch angle in radians. Can be a callable function
        that is dependent on either u or r which is determined
        by whether r/r0 or u/u0 parameters are passed.

    φ : float, Tensor
        The phase angle at u=u0 in radians.

    r : Tensor, optional
        The radial coordinate

    r0 : float, Tensor, optional
        The reference distance at which θ(r0) = φ

    u : Tensor, optional
        The logarithm of the radial coordinate, u = log(r)

    u0 : float, Tensor, optional
        The reference distance at which θ(u0) = φ
    
    n : int
        The number of sampling points for constructing the interpolation
        function.
    
    method : callable
        The numerical integration method. Default is torchquad.Trapezoid.

    Returns
    -------
    θ : Tensor
        The azimuth angle of the logarithmic spiral arm at the given
        radial coordinates.

    Examples
    --------
    import math
    import matplotlib.pyplot as plt
    import torch
    from galsyn.galaxy.spiral.logarithmic import θ_logarithmic

    u  = torch.linspace(0, 1, 10)
    u0 = 0.5
    r  = u.exp()
    r0 = math.exp(u0)
    α  = lambda x, x0: -0.25 * (x - x0) + 0.5
    φ  = 0

    kwargs = {'u': u, 'u0': u0}
    θ1 = θ_logarithmic(α, φ, **kwargs)
    θ2 = θ_logarithmic(α(*kwargs.values()), φ, **kwargs)

    fig, ax = plt.subplots()
    ax.plot(θ1, u)
    ax.plot(θ2, u)
    fig.show()
    """
    if (r is not None) and (r0 is not None):
        u = r.add(eps).log()
        if isinstance(r0, (float,int)):
            r0 = torch.as_tensor(r0, device=r.device, dtype=torch.float32)
        u0 = r0.add(eps).log()

        def f(u):
            r = u.exp().sub(eps)
            return 1 / torch.tan(α(r, r0))

    elif (u is not None) and (u0 is not None):
        def f(u):
            return 1/torch.tan(α(u, u0))
    else:
        raise Exception("Need to pass in either r/r0 or u/u0")

    if not callable(α):
        return φ + (u - u0) / torch.as_tensor(α, dtype=torch.float32, device=u.device).tan()

    qu = torch.linspace(u.min(), u.max(), n, device=u.device, dtype=torch.float32)
    qθ = torch.empty(n, device=u.device, dtype=torch.float32)

    for i,ui in enumerate(qu):
        if u0 < ui:
            domain = [[u0, ui]]
            qθ[i] = φ + method.integrate(f, dim=1, N=51, integration_domain=domain)
        else:
            domain = [[ui, u0]]
            qθ[i] = φ - method.integrate(f, dim=1, N=51, integration_domain=domain)            
    
    return interp1d(qu, qθ, u.flatten()).view(u.shape)