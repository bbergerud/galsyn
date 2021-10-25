"""
Methods that define analytical expressions for the PSF.
Used by the StarGenerator class to generate synthetic
images of stars.

Classes
-------
GaussianModel(fwhm_column)
    Interface to the gaussian_model function that stores
    the names of the fwhm parameter upon initialization.

MoffatModel(core_width, power_index)
    Interface to the moffat_model function that stores
    the names of the core_width and power_index parameters
    upon initialization.

SDSSModel(b, beta, p0, sigma1, sigma2, sigmaP)
    Interface to the sdss_model function that stores the
    names of the column parameters upon initialization.

Functions
---------
gaussian_model(dx, dy, flux, plate_scale, fwhm_column, **kwargs)
    Analytical model for a Gaussian profile. Assumes the FWHM has units
    of arcseconds.

moffat_model(dx, dy, flux, plate_scale, core_width_column, power_index_column **kwargs)
    Analytical model for a Moffat profile. Assumes the core width has units 
    of arcseconds. The normalized profile is of the form
        f(r;α,β) = (β-1)/(πα²)⋅[1 + (r²/α²)]ᵝ
    where α is the core width and β the power index.

sdss_model(dx, dy, flux, plate_scale, b_column, beta_column, p0_column,
           sigma1_column, sigma2_column, sigmaP_column, **kwargs)
    Analytical model for a SDSS profile. Assumes the width has units of
    pixels (0.396'' plate scale). The normalized profile is of the form
        f(r) = [exp(-r²/(2σ₁²)) + b⋅exp(-r²/(2σ₂²)) + p₀[1 + r²/(βσₚ²)]^(-β/2)] / [1+b+p₀]
"""
import math
import torch
from dataclasses import dataclass
from ..utils import accessor

fwhm2std = 1 / math.sqrt(8 * math.log(2))   # Conversion factor

def gaussian_model(
    dx          : torch.Tensor, 
    dy          : torch.Tensor, 
    flux        : torch.Tensor, 
    plate_scale : float,
    fwhm_column : str = 'psfWidth',
    **kwargs
) -> torch.Tensor:
    """
    Analytical model for a Gaussian profile. Assumes the FWHM has units
    of arcseconds.

    Parameters
    ----------
    dx : Tensor
        The vertical pixel separation from the center of the source

    dy : Tensor
        The horiztonal pixel separation from the center of the source

    flux : Tensor
        The total flux of the source

    plate_scale : float
        The plate scale of the image, in arcseconds / pixel

    fwhm_column : str
        The name of the column associated with the FWHM.

    **kwargs
        Any additional arguments to the accessor function.

    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.spatial import coordinate, grid
    from galsyn.star.psf import gaussian_model

    dx, dy = coordinate.cartesian(
        grid = grid.pixel_grid(100, 100),
        h0 = 49.5,
        w0 = 49.5,
    )

    kwargs = {
        'dataframe'  : None,
        'device'     : None,
        'dictionary' : {'psfWidth': [1.5]},
        'filter_band': 'r',
    }

    image = gaussian_model(dx, dy, 1, plate_scale=0.396, **kwargs)

    print(f'sum(image): {image.sum()}')

    fig, ax = plt.subplots()
    ax.imshow(image.squeeze())
    fig.show()
    """
    fwhm  = accessor(fwhm_column, **kwargs)             # FWHM in arcseconds
    sigma = fwhm2std * fwhm / plate_scale               # STD in pixels
    amplitude = flux / (2 * math.pi * sigma * sigma)

    flux = amplitude * torch.exp(-0.5 * (dx**2 + dy**2)/sigma**2)
    return flux

def moffat_model(
    dx                 : torch.Tensor,
    dy                 : torch.Tensor, 
    flux               : torch.Tensor, 
    plate_scale        : float, 
    core_width_column  : str = 'alpha', 
    power_index_column : str = 'beta', 
    **kwargs
) -> torch.Tensor:
    """
    Analytical model for a Moffat profile. Assumes the core width has units 
    of arcseconds. The normalized profile is of the form

        f(r;α,β) = (β-1)/(πα²)⋅[1 + (r²/α²)]ᵝ
 
    where α is the core width and β the power index.

    Parameters
    ----------
    dx : Tensor
        The vertical pixel separation from the center of the source

    dy : Tensor
        The horiztonal pixel separation from the center of the source

    flux : Tensor
        The total flux of the source

    plate_scale : float
        The plate scale of the image, in arcseconds / pixel

    core_width_column : str
        The name of the column associated with the core width of the profile.
        Traditionally denoted by `alpha`, but astropy uses `gamma`.

    power_index_column : str
        The name of the column associated with the power index of the profile.
        Traditionally denoted by `beta`, but astropy uses `alpha`.

    **kwargs
        Any additional arguments to the accessor function.

    Examples
    --------
    import matplotlib.pyplot as plt
    from galnet.spatial import coordinate, grid
    from galnet.synthetic.star.psf import moffat_model

    dx, dy = coordinate.cartesian(
        grid = grid.pixel_grid(100, 100),
        h0 = 49.5,
        w0 = 49.5,
    )

    kwargs = {
        'dataframe'  : None,
        'device'     : None,
        'dictionary' : {'alpha': [1.5], 'beta': [3]},
        'filter_band': 'r',
    }

    image = moffat_model(dx, dy, 1, plate_scale=0.396, **kwargs)

    print(f'sum(image): {image.sum()}')

    fig, ax = plt.subplots()
    ax.imshow(image.squeeze())
    fig.show()
    """
    alpha = accessor(core_width_column, **kwargs) / plate_scale
    beta  = accessor(power_index_column, **kwargs)
    amplitude = flux * (beta - 1) / (math.pi * alpha**2)

    flux = amplitude * (dx.pow(2) + dy.pow(2)).div(alpha.pow(2)).add(1).pow(-beta)
    return flux

def sdss_model(
    dx            : torch.Tensor,
    dy            : torch.Tensor, 
    flux          : torch.Tensor, 
    plate_scale   : float,
    b_column      : str = 'psfB',
    beta_column   : str = 'psfBeta',
    p0_column     : str = 'psfP0',
    sigma1_column : str = 'psfSigma1', 
    sigma2_column : str = 'psfSigma2',
    sigmaP_column : str = 'psfSigmaP',
    **kwargs,
) -> torch.Tensor:
    """
    Analytical model for a SDSS profile. Assumes the width has units of
    pixels (0.396'' plate scale). The normalized profile is of the form

        f(r) = [exp(-r²/(2σ₁²)) + b⋅exp(-r²/(2σ₂²)) + p₀[1 + r²/(βσₚ²)]^(-β/2)] / [1+b+p₀]
 
    Parameters
    ----------
    dx : Tensor
        The vertical pixel separation from the center of the source

    dy : Tensor
        The horiztonal pixel separation from the center of the source

    flux : Tensor
        The total flux of the source

    plate_scale : float
        The plate scale of the image, in arcseconds / pixel

    b_column : str
        The name of the column associated with the ratio of the inner to
        outer PSF at the origin.

    beta_column : str
        The name of the column associated with the slope of the powerlaw.

    p0_column : str
        The name of the column associated with the value of the powerlaw
        at the origin.

    sigma1_column : str
        The name of the column associated with the inner gaussian sigma
        for the composite fit.

    sigma2_column : str
        The name of the column associated with the outer gaussian sigma
        for the composite fit.

    sigmaP_column : str
        The name of the column associated with the width of the powerlaw.

    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.spatial import coordinate, grid
    from galsyn.star.psf import sdss_model

    dx, dy = coordinate.cartesian(
        grid = grid.pixel_grid(100, 100),
        h0 = 49.5,
        w0 = 49.5,
    )

    kwargs = {
        'dataframe'  : None,
        'device'     : None,
        'dictionary' : {
            'psfSigma1': 1.41,
            'psfSigma2': 3.10,
            'psfSigmaP': 2.98,
            'psfP0'    : 0.008,
            'psfB'     : 0.034,
            'psfBeta'  : 3.25,
        },
        'filter_band': 'r',
    }

    image = sdss_model(dx, dy, 1, plate_scale=0.396, **kwargs)

    print(f'sum(image): {image.sum()}')

    fig, ax = plt.subplots()
    ax.imshow(image.squeeze())
    fig.show()
    """
    σ1_sq = accessor(sigma1_column, **kwargs)**2
    σ2_sq = accessor(sigma2_column, **kwargs)**2
    σp_sq = accessor(sigmaP_column, **kwargs)**2
    b     = accessor(b_column, **kwargs)
    β     = accessor(beta_column, **kwargs)
    p0    = accessor(p0_column, **kwargs)

    r_sq = dx.pow(2) + dy.pow(2) * (plate_scale / 0.396)**2

    num  = r_sq.div(-2*σ1_sq).exp() \
         + r_sq.div(-2*σ2_sq).exp().mul(b) \
         + (1 + r_sq.div(β * σp_sq)).pow(-β/2).mul(p0)
    den  = 1 + b + p0

    int_val = (2*math.pi/den) * (σ1_sq + b*σ2_sq + p0*β*σp_sq/(β - 2))
    amplitude = flux / int_val

    return amplitude * num / den

@dataclass
class GaussianModel:
    """
    Interface to the gaussian_model function that stores
    the names of the fwhm parameter upon initialization.
    """
    fwhm:str = 'psfWidth'

    def __call__(self, *args, **kwargs):
        return gaussian_model(
            *args,
            fwhm_column=self.fwhm,
            **kwargs
        )

@dataclass
class MoffatModel:
    """
    Interface to the moffat_model function that stores
    the names of the core_width and power_index parameters
    upon initialization.
    """
    core_width  : str = 'alpha'
    power_index : str = 'beta' 
    
    def __call__(self, *args, **kwargs):
        return moffat_model(
            *args,
            core_width_column=self.core_width,
            power_index_column=self.power_index, 
            **kwargs
        )

@dataclass
class SDSSModel:
    """
    Interface to the sdss_model function that stores the
    names of the column parameters upon initialization.
    """
    b      : str = 'psfB'
    beta   : str = 'psfBeta'
    p0     : str = 'psfP0'
    sigma1 : str = 'psfSigma1'
    sigma2 : str = 'psfSigma2'
    sigmaP : str = 'psfSigmaP'

    def __call__(self, *args, **kwargs):
        return sdss_model(
            *args,
            b_column      = self.b,
            beta_column   = self.beta,
            p0_column     = self.p0,
            sigma1_column = self.sigma1,
            sigma2_column = self.sigma2,
            sigmaP_column = self.sigmaP,
            **kwargs,
        )
