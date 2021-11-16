"""
Module for performing discrete convolutions to simulate the PSF.

Classes
-------
DoubleGaussianPowerlawConvolution
    Class interface to the double gaussian powerlaw convolution.

GaussianConvolution
    Class interface to the gaussian convolution.

MoffatConvolution
    Class interface to the moffat convolution.

Methods
-------
double_gaussian_powerlaw_convolution(input, sky_detector, plate_scale, filter_band, index,
        device, default_size, b_column, beta_column, p0_column,
        sigma1_column, sigma2_column. sigmaP_column)
    Performs a double gaussian powerlaw convolution on the input tensor,
    which is the model profile used by SDSS.

gaussian_convolution(input, sky_detector, plate_scale, filter_band, index, device, fwhm_column)
    Performs a gaussian convolution on the input tensor.

moffat_convolution(input, sky_detector, plate_scale, index, filter_band,
                   device, core_width_column, power_index_column)
    Performs a moffat convolution on the input tensor.
"""

import torch
from typing import Dict, Optional
from dataclasses import dataclass
from galkit.functional import get_kernel2d_fwhm, Gaussian, Moffat, DoubleGaussianPowerlaw
from kornia.filters import filter2d
from ..utils import accessor
from ..sky_detector import SkyDetectorGenerator

def double_gaussian_powerlaw_convolution(
    input         : torch.Tensor, 
    sky_detector  : SkyDetectorGenerator, 
    plate_scale   : float,
    filter_band   : str,
    index         : int, 
    device        : Optional[torch.device] = None,
    default_size  : int = 4,
    b_column      : str = 'psfB',
    beta_column   : str = 'psfBeta',
    p0_column     : str = 'psfP0',
    sigma1_column : str = 'psfSigma1',
    sigma2_column : str = 'psfSigma2',
    sigmaP_column : str = 'psfSigmaP',
):
    """
    Performs a double gaussian powerlaw convolution on the input tensor,
    which is the model profile used by SDSS.

    Parameters
    ----------
    input : Tensor (C × H × W)
        The Tensor to apply the convolution to.

    sky_detector
        The sky detector object that contains the parameters of the
        point spread function.

    plate_scale : float
        The plate scale of the image. Used to convert the FWHM from
        arcseconds to pixels.

    filter_band : str
        The name of the filter band. Used to find the relevant FWHM.

    index : int
        The index of the data entry in the sky detector that contains the
        relevant point spread function parameters.

    device : torch.device, optional
        Device to generate the kernel data on. Default is None.

    default_size : int
        The number of FWHM for constructing the kernel.

    b_column : str
        Name of the column corresponding to the ratio of the inner
        to outer PSF at the origin.

    beta_column : str
        Name of the column corresponding to the slope of the powerlaw.

    p0_column : str
        Name of the column corresponding to the value of the powerlaw
        at the origin.

    sigma1_column : str
        Name of the column corresponding to the inner gaussian sigma
        for the composite fit. Assumes the copula returns pixel values
        on a 0.396'' scale.
    
    sigma2_column : str
        Name of the column corresponding to the outer gaussian sigma
        for the composite fit. Assumes the copula returns pixel values
        on a 0.396'' scale.

    sigmaP_column : str
        Name of the column corresponding to the width parameter for
        the powerlaw. Assumes the copula returns pixel values on a
        0.396'' scale.

    Returns
    -------
    output : Tensor (C × H × W)
        The convolved input tensor.

    Examples
    --------
    import matplotlib.pyplot as plt
    import torch
    import math
    from galkit.spatial import coordinate, grid
    from galsyn.sky_detector import SkyDetectorGenerator
    from galsyn.galaxy.profile import Sersic
    from galsyn.galaxy.convolution import double_gaussian_powerlaw_convolution

    sky = SkyDetectorGenerator()
    sky.sample(1)

    profile = Sersic(
        amplitude = {'i': 1.0, 'r': 0.9, 'g': 0.8},
        scale     = {'i': 0.2, 'r': 0.3, 'g': 0.4},
        index     = {'i': 2.0, 'r': 1.9, 'g': 1.8},
        truncate  = None
    )

    _, r = coordinate.polar(
        grid = grid.pytorch_grid(128,128),
        q = 0.5,
        pa = math.pi/4,
        scale = 0.25,
    )

    flux = {k:profile(r,filter_band=k) for k in 'irg'}

    kwargs = {
        'sky_detector': sky,
        'plate_scale' : 0.396,
        'index': 0,
    }

    flux_psf = {k:double_gaussian_powerlaw_convolution(v, filter_band=k, **kwargs) for k,v in flux.items()}

    imag1 = torch.cat([flux[k] for k in 'irg'], dim=0).permute(1,2,0)
    imag2 = torch.cat([flux_psf[k] for k in 'irg'], dim=0).permute(1,2,0)
    imax  = imag1.max()

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(imag1 / imax)
    ax[1].imshow(imag2 / imax)
    ax[0].set_title('Without PSF')
    ax[1].set_title('With PSF')
    fig.show()
    """

    kwargs = {
        'dataframe'   : sky_detector.data,
        'dictionary'  : sky_detector.dictionary,
        'filter_band' : filter_band,
        'index'       : index,
        'view'        : None,
    }

    profile = DoubleGaussianPowerlaw(
        b      = accessor(b_column, **kwargs).squeeze(),
        beta   = accessor(beta_column, **kwargs).squeeze(),
        p0     = accessor(p0_column, **kwargs).squeeze(),
        sigma1 = accessor(sigma1_column, **kwargs).squeeze() * (0.396 / plate_scale),
        sigma2 = accessor(sigma2_column, **kwargs).squeeze() * (0.396 / plate_scale),
        sigmaP = accessor(sigmaP_column, **kwargs).squeeze() * (0.396 / plate_scale),
    )
    kernel = get_kernel2d_fwhm(profile, device=device, default_size=default_size).unsqueeze(0)

    return filter2d(input.unsqueeze(0), kernel, border_type='replicate').squeeze(0)

def gaussian_convolution(
    input        : torch.Tensor, 
    sky_detector : SkyDetectorGenerator, 
    plate_scale  : float, 
    filter_band  : str,
    index        : int,
    device       : Optional[torch.device] = None,
    default_size : int = 4,
    fwhm_column  : str = 'psfWidth',
) -> torch.Tensor:
    """
    Performs a gaussian convolution on the input tensor.

    Parameters
    ----------
    input : Tensor (C × H × W)
        The Tensor to apply the convolution to.

    sky_detector
        The sky detector object that contains the parameters of the
        point spread function.

    plate_scale : float
        The plate scale of the image. Used to convert the FWHM from
        arcseconds to pixels.

    filter_band : str
        The name of the filter band. Used to find the relevant FWHM.

    index : int
        The index of the data entry in the sky detector that contains the
        relevant point spread function parameters.

    device : torch.device, optional
        Device to generate the kernel data on. Default is None.
    
    default_size : int
        The number of FWHM for constructing the kernel.

    fwhm_column : str
        Name of the column that corresponds to the FWHM, which is assumed
        to be in arcseconds. Default is `psfWidth`.

    Returns
    -------
    output : Tensor (C × H × W)
        The convolved input tensor.

    Examples
    --------
    import matplotlib.pyplot as plt
    import torch
    import math
    from galkit.spatial import coordinate, grid
    from galsyn.sky_detector import SkyDetectorGenerator
    from galsyn.galaxy.profile import Sersic
    from galsyn.galaxy.convolution import gaussian_convolution

    sky = SkyDetectorGenerator()
    sky.sample(1)

    profile = Sersic(
        amplitude = {'i': 1.0, 'r': 0.9, 'g': 0.8},
        scale     = {'i': 0.2, 'r': 0.3, 'g': 0.4},
        index     = {'i': 2.0, 'r': 1.9, 'g': 1.8},
        truncate  = None
    )

    _, r = coordinate.polar(
        grid = grid.pytorch_grid(128,128),
        q = 0.5,
        pa = math.pi/4,
        scale = 0.25,
    )

    flux = {k:profile(r,filter_band=k) for k in 'irg'}

    kwargs = {
        'sky_detector': sky,
        'plate_scale' : 0.396,
        'index': 0,
    }

    flux_psf = {k:gaussian_convolution(v, filter_band=k, **kwargs) for k,v in flux.items()}

    imag1 = torch.cat([flux[k] for k in 'irg'], dim=0).permute(1,2,0)
    imag2 = torch.cat([flux_psf[k] for k in 'irg'], dim=0).permute(1,2,0)
    imax  = imag1.max()

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(imag1 / imax)
    ax[1].imshow(imag2 / imax)
    ax[0].set_title('Without PSF')
    ax[1].set_title('With PSF')
    fig.show()
    """
    kwargs = {
        'dataframe' : sky_detector.data,
        'dictionary': sky_detector.dictionary,
        'filter_band': filter_band,
        'index'     : index
    }

    fwhm = accessor(fwhm_column, **kwargs).squeeze() / plate_scale
    profile = Gaussian(Gaussian.fwhm2std(fwhm))
    kernel = get_kernel2d_fwhm(profile=profile, default_size=default_size, device=device).unsqueeze(0)
    return filter2d(input.unsqueeze(0), kernel, border_type='replicate').squeeze(0)

def moffat_convolution(
    input              : torch.Tensor, 
    sky_detector       : SkyDetectorGenerator, 
    plate_scale        : float,
    filter_band        : str,
    index              : int, 
    device             : Optional[torch.device] = None,
    default_size       : int = 4,
    core_width_column  : str ='alpha',
    power_index_column : str ='beta',
) -> Dict[str, torch.Tensor]:
    """
    Performs a moffat convolution on the input tensor.

    Parameters
    ----------
    input : Tensor (C × H × W)
        The Tensor to apply the convolution to.

    sky_detector
        The sky detector object that contains the parameters of the
        point spread function.

    plate_scale : float
        The plate scale of the image. Used to convert the FWHM from
        arcseconds to pixels.

    index : int
        The index of the data entry in the sky detector that contains the
        relevant point spread function parameters.

    device : torch.device, optional
        Device to generate the kernel data on. Default is None.
    
    default_size : int
        The number of FWHM for constructing the kernel.

    core_width_column : str
        The name of the column associated with the core width of the profile.
        Traditionally denoted by `alpha`, but astropy uses `gamma`. The values
        are assumed to be in arcseconds.

    power_index_column : str
        The name of the column associated with the power index of the profile.
        Traditionally denoted by `beta`, but astropy uses `alpha`.

    Returns
    -------
    output : Tensor (C × H × W)
        The convolved input tensor.

    Examples
    --------
    import matplotlib.pyplot as plt
    import torch
    import math
    from galkit.spatial import coordinate, grid
    from galsyn.sky_detector import SkyDetectorGenerator
    from galsyn.galaxy.profile import Sersic
    from galsyn.galaxy.convolution import moffat_convolution

    sky = SkyDetectorGenerator()
    sky.sample(1)

    profile = Sersic(
        amplitude = {'i': 1.0, 'r': 0.9, 'g': 0.8},
        scale     = {'i': 0.2, 'r': 0.3, 'g': 0.4},
        index     = {'i': 2.0, 'r': 1.9, 'g': 1.8},
        truncate  = None
    )

    _, r = coordinate.polar(
        grid = grid.pytorch_grid(128,128),
        q = 0.5,
        pa = math.pi/4,
        scale = 0.25,
    )

    flux = {k:profile(r,filter_band=k) for k in 'irg'}
    flux_psf = {k:moffat_convolution(
        input = v,
        sky_detector = sky,
        plate_scale = 0.396,
        filter_band = k,
        index = 0,
        device = torch.device('cpu'),
        core_width_column = 'alpha',
        power_index_column = 'beta',
    ) for k,v in flux.items()}

    imag1 = torch.cat([flux[k] for k in 'irg'], dim=0).permute(1,2,0)
    imag2 = torch.cat([flux_psf[k] for k in 'irg'], dim=0).permute(1,2,0)
    imax  = imag1.max()

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(imag1 / imax)
    ax[1].imshow(imag2 / imax)
    ax[0].set_title('Without PSF')
    ax[1].set_title('With PSF')
    fig.show()
    """

    kwargs = {
        'dataframe'   : sky_detector.data,
        'dictionary'  : sky_detector.dictionary,
        'filter_band' : filter_band,
        'index'       : index
    }

    profile = Moffat(
        core_width  = accessor(core_width_column, **kwargs).squeeze() / plate_scale,
        power_index = accessor(power_index_column, **kwargs).squeeze()
    )

    kernel = get_kernel2d_fwhm(profile, default_size=default_size, device=device).unsqueeze(0)
    return filter2d(input.unsqueeze(0), kernel, border_type='replicate').squeeze(0)

@dataclass
class DoubleGaussianPowerlawConvolution:
    """
    Class interface to the double gaussian powerlaw convolution.

    Parameters
    ----------
    b : str
        Name of the column corresponding to the ratio of the inner
        to outer PSF at the origin.

    beta : str
        Name of the column corresponding to the slope of the powerlaw.

    p0 : str
        Name of the column corresponding to the value of the powerlaw
        at the origin.

    sigma1 : str
        Name of the column corresponding to the inner gaussian sigma
        for the composite fit.
    
    sigma2 : str
        Name of the column corresponding to the outer gaussian sigma
        for the composite fit.

    sigmaP : str
        Name of the column corresponding to the width parameter for
        the powerlaw.    
    """
    b      : str = 'psfB'
    beta   : str = 'psfBeta'
    p0     : str = 'psfP0'
    sigma1 : str = 'psfSigma1'
    sigma2 : str = 'psfSigma2'
    sigmaP : str = 'psfSigmaP'
    default_size : int = 4

    def __call__(self, *args, **kwargs):
        return double_gaussian_powerlaw_convolution(
            *args,
            b_column      = self.b,
            beta_column   = self.beta,
            p0_column     = self.p0,
            sigma1_column = self.sigma1,
            sigma2_column = self.sigma2,
            sigmaP_column = self.sigmaP,
            default_size  = self.default_size,
            **kwargs
        )

@dataclass
class GaussianConvolution:
    """
    Class interface to the gaussian convolution.

    Parameters
    ----------
    fwhm : str
        Name of the column that corresponds to the FWHM, which is assumed
        to be in arcseconds. Default is `psfWidth`.

    default_size : int
        The number of FWHM for constructing the kernel.
    """
    fwhm : str = 'psfWidth'
    default_size : int = 4

    def __call__(self, *args, **kwargs):
        return gaussian_convolution(
            *args,
            fwhm_column=self.fwhm,
            default_size=self.default_size,
            **kwargs
        )

@dataclass
class MoffatConvolution:
    """
    Class interface to the moffat convolution.

    Parameters
    ----------
    core_width : str
        The name of the column associated with the core width of the profile.
        Traditionally denoted by `alpha`, but astropy uses `gamma`. The values
        are assumed to be in arcseconds.

    power_index : str
        The name of the column associated with the power index of the profile.
        Traditionally denoted by `beta`, but astropy uses `alpha`.

    default_size : int
        The number of FWHM for constructing the kernel.
    """
    core_width  : str = 'alpha'
    power_index : str = 'beta'
    default_size : int = 4

    def __call__(self, *args, **kwargs):
        return moffat_convolution(
            *args,
            core_width_column=self.core_width,
            power_index_column=self.power_index,
            default_size=self.default_size,
            **kwargs
        )
