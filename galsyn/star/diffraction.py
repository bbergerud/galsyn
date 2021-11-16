"""
Methods for modeling diffraction patterns in PSF profiles.

Classes
-------
ApertureDiffraction
    Analytical solution to the diffraction pattern.

BesselJ1
    Wrapper to the scipy.special j1 function.

DiffractionWithoutPhase:
    Adds a diffraction pattern to an underlying PSF where the
    diffraction spikes lack the phase information.

Functions
---------
bessel_j1(input)
    Bessel j1 function.
"""

import torch
from galkit.spatial import coordinate
from math import pi
from scipy.special import j1
from torch.autograd import Function
from typing import Optional, Union
from ..utils import accessor

class BesselJ1(Function):
    """Interface to the scipy j1 function."""

    @staticmethod
    def forward(ctx, input:torch.Tensor) -> torch.Tensor:
        numpy_input = input.detach().numpy()
        result = j1(numpy_input)
        return input.new(result)

def bessel_j1(input:torch.Tensor) -> torch.Tensor:
    return BesselJ1.apply(input)

class ApertureDiffraction:
    """
    Analytical solution to the diffraction pattern.

    Parameters
    ----------
    eps : float
        The ratio of the diameter of the secondary mirror to the
        diameter of the primary mirror.

    width : float
        The relative width of the spider to the diameter of the
        primary mirror.

    orientation : tuple
        A tuple of the orientation directions of the spiders.

    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.spatial import coordinate, grid
    from galsyn.star.diffraction import ApertureDiffraction

    scale = 0.1
    shape = (256,256)
    dx, dy = coordinate.cartesian(
        grid = grid.pixel_grid(shape),
        h0 = (shape[0]-1)/2,
        w0 = (shape[1]-1)/2,
        scale = scale,
    )

    kwargs = {
        'dataframe'  : None,
        'device'     : None,
        'dictionary' : {'psfWidth': [0.05], 'alpha': [5], 'gamma': [5]},
        'filter_band': 'r',
    }

    model = ApertureDiffraction(eps=0.48, width=0.1)

    image = model(dx, dy, 1, plate_scale=0.396, **kwargs)

    print(f'sum(image): {image.sum() * scale**2}')

    fig, ax = plt.subplots()
    ax.imshow(image.squeeze().pow(0.25))
    fig.show()
    """
    def __init__(self,
        eps   : float = 0.48,
        width : float = 0.1,
        orientation : tuple = tuple(pi/4 + pi/2*i for i in range(4))
    ):
        self.eps    = eps
        self.strut_length = 1 - eps
        self.strut_width  = width
        self.strut_center = 0.5 * (1 + self.eps)
        self.area = pi * (1 - self.eps**2) - len(orientation) * self.strut_width * self.strut_length
        self.orientation = orientation

    def __call__(self, 
        dx          : torch.Tensor, 
        dy          : torch.Tensor, 
        flux        : float = 1, 
        plate_scale : float = 0.396, 
        fwhm_column : str = 'psfWidth',
        **kwargs
    ) -> torch.Tensor:
        """
        Computes the diffraction pattern at the specified positions.

        Parameters
        ----------
        dx : tensor
            The vertical separation from the center of the source

        dy : tensor
            The horizontal separation from the center of the source

        flux : Tensor
            The total flux of the source

        plate_scale : float
            The plate scale of the image, in arcseconds / pixel

        fwhm_column : str
            The name of the column associated with the fwhm.

        **kwargs
            Any additional arguments to the accessor function.
        """
        fwhm  = accessor(fwhm_column, **kwargs) / plate_scale   # FWHM in pixels
        scale = 1 / (4 * fwhm)
        area  = self.area * scale**2

        rho = (dx.pow(2) + dy.pow(2)).sqrt().add(1e-8)
        ft  = self.circle(scale, rho=rho) - self.circle(self.eps * scale, rho=rho)

        for x in self.orientation:
            ft = ft - self.strut(*coordinate.rotate(dx,dy,x), scale=scale)

        return flux * (ft * ft.conj()).real / area


    def circle(self,
        radius : float,
        u      : Optional[torch.Tensor] = None,
        v      : Optional[torch.Tensor] = None,
        rho    : Optional[torch.Tensor] = None,
    ):
        """
        Diffraction pattern for a circular aperture.
        """
        if rho is None:
            if (u is None) or (v is None):
                raise Exception("Either (u,v) or rho must be passed")
            rho = (u.pow(2) + v.pow(2)).sqrt().add(1e-8)

        x = 2 * pi * rho * radius
        return pi * radius**2 * 2 * bessel_j1(x) / x

    def strut(self,
        u : torch.Tensor,
        v : torch.Tensor,
        scale : float = 1,
    ): 
        """
        Diffraction pattern for a rectangular strut.
        """
        strut_length = self.strut_length * scale
        strut_width  = self.strut_width * scale
        strut_center = self.strut_center * scale

        # Pytorch defines sinc(x) = sinc(pi*x) / (pi * x)
        x1 = u * strut_width
        x2 = v * strut_length
        phase = torch.exp(-2j * pi * v * strut_center)
        return strut_length * strut_width * phase * torch.sinc(x1) * torch.sinc(x2)


class DiffractionWithoutPhase:
    """
    Adds a diffraction pattern to an underlying PSF where the
    diffraction spikes lack the phase information.

    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.spatial import coordinate, grid
    from galsyn.star import gaussian_model, moffat_model, DiffractionWithoutPhase

    dx, dy = coordinate.cartesian(
        grid = grid.pixel_grid(200, 200),
        h0 = 99.5,
        w0 = 99.5,
    )

    kwargs = {
        'dataframe'  : None,
        'device'     : None,
        'dictionary' : {'psfWidth': [1.5], 'alpha': [5], 'beta': [5]},
        'filter_band': 'r',
    }

    model = DiffractionWithoutPhase(moffat_model, fraction=0.5)

    image = model(dx, dy, 1, plate_scale=0.396, **kwargs)

    print(f'sum(image): {image.sum()}')

    fig, ax = plt.subplots()
    ax.imshow(image.squeeze().pow(1))
    fig.show()
    """
    def __init__(self,
        model : callable,
        eps   : float = 0.48,
        width : float = 0.1,
        orientation : tuple = [pi/4 + pi/2*i for i in range(2)],
        first_pixel_zero : float = 1,
        fraction : Union[callable, float] = 0.90,
    ):
        """
        Parameters
        ----------
        model : callable
            The base PSF model.

        eps : float
            The ratio of the diameter of the secondary mirror to the
            diameter of the primary mirror.

        width : float
            The relative width of the spider to the diameter of the
            primary mirror.

        orientation : tuple
            A tuple of the orientation directions of the spiders.

        first_pixel_zero : float
            The pixel number at which the diffraction reaches its first
            minimum. Used as a scaling factor.

        fraction : callable, float
            The fractional amount that the base PSF makes up, or a function
            that returns this amount.
        """
        length = 1 - eps
        const = 1 / (first_pixel_zero * max(length, width))

        self.length = length * const
        self.width  = width * const
        self.orientation = orientation
        self.norm = self.width * self.length / len(self.orientation)
        self.model = model
        self.fraction = fraction

    def __call__(self, dx, dy, flux, plate_scale=None, **kwargs):
        model = self.model(dx, dy, flux, plate_scale=plate_scale, **kwargs)

        intensity = 0
        for x in self.orientation:
            u, v = coordinate.rotate(dx, dy, x)
            x1 = u * self.width
            x2 = v * self.length
            intensity = intensity + torch.sinc(x1).pow(2) * torch.sinc(x2).pow(2)
        spikes = intensity * flux * self.norm

        f = self.fraction() if callable(self.fraction) else self.fraction
        return f*model + (1-f)*spikes