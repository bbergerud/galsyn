"""
Methods for generating photometric models of galaxies.

Classes
-------
Exponential(amplitude, scale)
    Exponential disk profile,
        I(r) = amplitude * exp(-r / scale)

ExponentialBreak(amplitude, scale_inner, scale_outer, breakpoint)
    Exponential profile with a breakpoint,
        I(r) = amplitude1 ⋅ exp(-r / scale_inner) ⋅ Θ(r <= breakpoint) 
             + amplitude2 ⋅ exp(-r / scale_outer) ⋅ Θ(r > breakpoint)
    where Θ is the heaviside step function. The amplitude of the second
    component is calculated to ensure the function is continuous.

Ferrer(amplitude, scale, index)
    Ferrer ellipsoid used for modelling bar profiles. The functional
    form of the expression is     
        I(r) = amplitude ⋅ [1 - (r / scale)²]^(index + 0.5)     [r < scale]
    where I(r > scale) = 0.

FerrerModified(amplitude, scale, alpha, beta)
    Modified version of the Ferrer function,
        I(r) = amplitude ⋅ [1 - (r / scale)²⁻ᵝ]ᵅ        [r < scale]
    where I(r > scale) = 0.

Sersic
    Sersic profile,
        I(r) = amplitude * exp(-bₙ⋅[(r/scale)^(1/index) - 1])
    where bₙ is the solution to Γ(2n)/2 = γ(2n,bₙ). The parameters
    are specified in terms of the half-light radius. Note that this
    reduces to the exponential profile when index=1.

Profile
    Class for constructing model profiles and fluxes.
"""

import torch
from dataclasses import dataclass
from galkit.functional import sigmoid, sersic, exponential, exponential_break, ferrer, ferrer_modified
from galkit.spatial import resample
from typing import Dict, Iterable, Optional, Tuple, Union
from ..utils import accessor
from ..sky_detector import SkyDetectorGenerator

@dataclass
class Exponential:
    """
    Exponential disk profile,

        I(r) = amplitude * exp(-r / scale)

    Parameters
    ----------
    amplitude : dict, float
        Either the value of the amplitude at r=0 or a dictionary containing
        such values where the keys are the filter bands.

    scale : dict, float
        The scale length or a dictionary containing such values where the
        keys are the filter bands.
    """
    amplitude  : Union[dict, float]
    scale      : Union[dict, float]

    def __call__(self, r:torch.Tensor, filter_band:str, **kwargs) -> torch.Tensor:
        return exponential(r=r, 
            amplitude = self.amplitude if not isinstance(self.amplitude, dict) else self.amplitude[filter_band], 
            scale     = self.scale     if not isinstance(self.scale, dict)     else self.scale[filter_band],  
        )

@dataclass
class ExponentialBreak:
    """
    Exponential profile with a breakpoint,

        I(r) = amplitude1 ⋅ exp(-r / scale_inner) ⋅ Θ(r <= breakpoint) 
             + amplitude2 ⋅ exp(-r / scale_outer) ⋅ Θ(r > breakpoint)

    where Θ is the heaviside step function. The amplitude of the second
    component is calculated to ensure the function is continuous.

    Parameters
    ----------
    amplitude : dict, float
        Either the value of the amplitude at r=0 for the first component
        or a dictionary containing such values where the keys are the filter bands.

    scale_inner : dict, float
        The scale length of the inner component or a dictionary containing
        such values where the keys are the filter bands.

    scale_outer : dict, float
        The scale length of the outer component or a dictionary containing
        such values where the keys are the filter bands.

    breakpoint : dict, float
        The breakpoint at which the profile changes from the inner to the outer component.
    """
    amplitude   : Union[dict, float]
    scale_inner : Union[dict, float]
    scale_outer : Union[dict, float]
    breakpoint  : Union[dict, float]

    def __call__(self, r:torch.Tensor, filter_band:str, **kwargs) -> torch.Tensor:
        return exponential_break(r=r, 
            amplitude   = float(self.amplitude   if not isinstance(self.amplitude, dict)   else self.amplitude[filter_band]), 
            scale_inner = float(self.scale_inner if not isinstance(self.scale_inner, dict) else self.scale_inner[filter_band]),
            scale_outer = float(self.scale_outer if not isinstance(self.scale_outer, dict) else self.scale_outer[filter_band]),
            breakpoint  = float(self.breakpoint if not isinstance(self.breakpoint, dict)   else self.breakpoint[filter_band]),
        )

@dataclass
class Ferrer:
    """
    Ferrer ellipsoid used for modelling bar profiles. The functional
    form of the expression is     

        I(r) = amplitude ⋅ [1 - (r / scale)²]^(index + 0.5)

    where I(r > scale) = 0.

    Parameters
    ----------    
    amplitude : dict, float
        Either the value of the amplitude at r=0 or a dictionary containing
        such values where the keys are the filter bands.

    scale : dict, float
        Either the length of the ellipsoid or a dictionary containing such values
        where the keys are the filter bands.

    index : dict, float
        Either power index or a dictionary containing such values where the keys
        are the filter bands.
    """
    amplitude : Union[dict, float]
    scale     : Union[dict, float]
    index     : Union[dict, float] = 2

    def __call__(self, r:torch.Tensor, filter_band:str, **kwargs) -> torch.Tensor:
        return ferrer(r=r, 
            amplitude = float(self.amplitude if not isinstance(self.amplitude, dict) else self.amplitude[filter_band]), 
            scale     = float(self.scale     if not isinstance(self.scale, dict)     else self.scale[filter_band]), 
            index     = float(self.index     if not isinstance(self.index, dict)     else self.index[filter_band]), 
        )

@dataclass
class FerrerModified:
    """
    Modified version of the Ferrer function,

        I(r) = amplitude ⋅ [1 - (r / scale)²⁻ᵝ]ᵅ        [r < scale]
 
    where I(r > scale) = 0.

    Parameters
    ----------
    amplitude : dict, float
        Either the value of the amplitude at r=0 for the first component
        or a dictionary containing such values where the keys are the filter bands.

    scale : dict, float
        Either the length of the ellipsoid or a dictionary containing such values
        where the keys are the filter bands.

    alpha : dict, float
        Either the outer power index or a dictionary containing such values
        where the keys are the filter bands.

    beta : dict, float
        Either the inner power index or a dictionary containing such values
        where the keys are the filter bands.
    """
    amplitude : Union[dict, float]
    scale     : Union[dict, float]
    alpha     : Union[dict, float] = 2.5
    beta      : Union[dict, float] = 2

    def __call__(self, r:torch.Tensor, filter_band:str, **kwargs) -> torch.Tensor:
        return ferrer_modified(r=r, 
            amplitude = float(self.amplitude if not isinstance(self.amplitude, dict) else self.amplitude[filter_band]), 
            scale     = float(self.scale     if not isinstance(self.scale, dict)     else self.scale[filter_band]),
            alpha     = float(self.alpha     if not isinstance(self.alpha, dict)     else self.alpha[filter_band]),
            beta      = float(self.beta      if not isinstance(self.beta, dict)      else self.beta[filter_band]),
        )

@dataclass
class Sersic:
    """
    Sersic model profile,

        I(r) = amplitude * exp(-bₙ⋅[(r/scale)^(1/index) - 1])
    
    where bₙ is the solution to Γ(2n)/2 = γ(2n,bₙ). The parameters
    are specified in terms of the half-light radius. Note that this
    reduces to the exponential profile when index=1.

    Parameters
    ----------
    amplitude : dict, float
        The amplitude of the profile or a dictionary containing the
        (filter_band, amplitude) key-value pairings.

    scale : dict, float
        The scale of the profile or a dictionary containing the
        (filter_band, scale) key-value pairings.

    index : dict, float
        The index of the profile or a dictionary containing the
        (filter_band, index) key-value pairings. Default is 1
        to generate an exponential profile in the case of a disk.

    truncate : dict, float, optional
        The truncation radius of the profile or a dictionary containing
        the (filter_band, truncate) key-value pairings. If `None`, then
        no truncation is applied. Default is None.

    truncate_scale : dict, float, optional
        Roughly the fractional value of the truncation radius at which
        the width of the transition occurs. Used for a smoother transition
        to zero flux than a sharp dropoff.

    Examples
    --------
    import matplotlib.pyplot as plt
    import torch
    from galsyn.galaxy.profile import Sersic

    profile = Sersic(
        amplitude = {'i': 1.0, 'r': 1.0, 'g': 0.8},
        scale     = {'i': 0.2, 'r': 0.2, 'g': 0.4},
        index     = {'i': 2.0, 'r': 2.0, 'g': 1.8},
        truncate  = {'i': 0.9, 'r': 0.9, 'g': None},
        truncate_scale = {'i': 0.1, 'r': None, 'g': None},
    )

    r = torch.linspace(0.01, 1)
    flux = {k:profile(r,filter_band=k) for k in 'irg'}

    fig, ax = plt.subplots()
    for k,v in flux.items():
        ax.loglog(r, v, label=k)
    ax.legend(loc='best')
    fig.show()
    """
    amplitude      : Union[dict, float]
    scale          : Union[dict, float]
    index          : Union[dict, float] = 1
    truncate       : Optional[Union[dict, float]] = None
    truncate_scale : Optional[Union[dict, float]] = 0.1

    def function(self,
        r         : torch.Tensor, 
        amplitude : float, 
        scale     : float, 
        index     : float = 1, 
        truncate  : Optional[float] = None,
        truncate_scale : float = 0.05,
    ) -> torch.Tensor:
        flux = sersic(r=r, amplitude=amplitude, scale=scale, index=index)

        if truncate is not None:
            if truncate_scale is not None:
                flux_factor = 1 - sigmoid(r, loc=truncate, scale=truncate_scale*truncate)
                flux = flux * flux_factor
            else:
                flux[r > truncate] = 0

        return flux        

    def __call__(self, r:torch.Tensor, filter_band:str, **kwargs) -> torch.Tensor:
        return self.function(r=r, 
            amplitude = self.amplitude if not isinstance(self.amplitude, dict) else self.amplitude[filter_band], 
            scale     = self.scale     if not isinstance(self.scale, dict)     else self.scale[filter_band], 
            index     = self.index     if not isinstance(self.index, dict)     else self.index[filter_band], 
            truncate  = self.truncate  if not isinstance(self.truncate, dict)  else self.truncate[filter_band], 
            truncate_scale = self.truncate_scale if not isinstance(self.truncate_scale, dict)  else self.truncate_scale[filter_band], 
        )

class Profile:
    """
    Class containing methods for generating and manipulating the projected
    flux of a model galaxy.

    Methods
    -------
    add_component_flux(flux)
        Adds the flux from each component together to form the galaxy flux.

    apply_shot_noise_to_counts(counts, sky_detector, index)
        Applies shot/Poisson noise to the flux count rates.

    convert_counts_to_flux(counts, sky_detector, index)
        Converts counts to flux units.

    convert_flux_to_counts(counts, sky_detector, index)
        Converts flux units to counts.

    convolve(input, sky_detector, plate_scale, index)
        Convolves the input based on the provided convolution method
        and the parameters contained in the sky_detector copula.

    downscale(input, factor)
        Downscales the input tensors by the specified factor.

    get_component_flux(geometry, profile, filter_bands)
        Generates the model flux given the geometry and profiles in each
        of the filter bands.

    get_component_profile(filter_bands, index)
        Generates a dictionary containing the profiles for each component.
        If `index=None`, then a tuple is returned containing the profiles
        for each galaxy.

    Required Attributes
    -------------------
    data
        A pandas dataframe that contains the galaxy data.

    device : torch.device
        The device on which to generate tensor data.

    profile : callable, dict
        A profile model, or a dictionary of such models where the key
        is the name of the component.

    Required Methods
    ----------------
    get_components()
        Returns the names of the galaxy components associated with the model.        

    get_component_params(component, filter_band, index)
        Function that returns the parameters associated with a given component
        in the indicated filter band.

    psf_model(input, sky_detector, plate_scale, filter_band, index)
        Returns the convolved input tensor.
    """
    def add_component_flux(self,
        flux       : Union[Dict, Tuple[Dict]],
        components : Optional[Tuple[str]] = None
    ) -> Union[Dict, Tuple[Dict]]:
        """
        Adds the flux from each component together to form the galaxy flux.

        Parameters
        ----------
        flux : Dict, Tuple[Dict]
            Either a dictionary where the key is the component and the value
            is another dictionary where the key is the filter band and the value
            the flux or a tuple containing such dictionaries.

        exceptions : Tuple[str]
            A sequence of strings containing any components to ignore.

        Returns
        -------
        flux : Dict, Tuple[Dict]
            Either a dictionary where the key is the filter band and the value
            the summed flux of all the components or a tuple of such dictionaries.

        Examples
        --------
        import matplotlib.pyplot as plt
        import torch
        from galkit.spatial import grid
        from galkit.functional import arcsinh_stretch
        from galsyn.galaxy.dataset import Gadotti as Galaxy

        filter_bands = 'irg'
        size  = 5
        shape = (128,128)

        galaxy = Galaxy()
        galaxy.sample(size)
        galaxy.convert_magnitude_to_flux(plate_scale=0.396)
        isoA = galaxy.get_isoMag(value=24.5, filter_band='r', index=None)

        profile  = galaxy.get_component_profile(filter_bands)
        geometry = galaxy.get_component_geometry(grid.pytorch_grid(shape), scale=isoA)
        component_flux = galaxy.get_component_flux(geometry=geometry, profile=profile, filter_bands=filter_bands)
        total_flux = galaxy.add_component_flux(component_flux, components=['disk', 'bar'])

        def foo(i):
            image = torch.cat([total_flux[i][k] for k in filter_bands])
            image = arcsinh_stretch(image)
            
            fig, ax = plt.subplots()
            ax.imshow(image.T)
            fig.show()
        
        for i in range(size):
            foo(i)
        """
        if isinstance(flux, (tuple, list)):
            return tuple(self.add_component_flux(f, components) for f in flux)

        output = {}
        for component, band_flux in flux.items():
            if (components is None) or (component in components):
                for k,v in band_flux.items():
                    if k not in output:
                        output[k] = v.clone() if v.size(0) == 1 else v.sum(dim=0, keepdims=True)
                    else:
                        output[k] += v if v.size(0) == 1 else v.sum(dim=0, keepdims=True)
        return output

    def apply_shot_noise_to_counts(self, 
        counts       : Union[Dict, Tuple[Dict]],
        sky_detector : SkyDetectorGenerator,
        index        : Optional[int] = None
    ) -> Union[Dict, Tuple[Dict]]:
        """
        Applies shot/Poisson noise to the flux count rates.

        Parameters
        ----------
        counts : Dict, Tuple[Dict]
            Either a dictionary where the key is the filter band and the value
            the counts, a dictionary where the key is the component and the value
            a dictionary of (filter_band, counts) pairs, or a tuple of such
            dictionaries.

        sky_detector : SkyDetectorGenerator
            A sky-detector module that contains information pertaining
            to the flux-to-count conversion rate.

        index : int, optional
            The index in sky_detector to use for extracting the flux to
            count conversion rates. If flux is a tuple, then a value of
            `None` will result in matching between the flux index and
            the sky_detector index.

        Returns
        -------
        counts : Dict, Tuple[Dict]               
            Either a dictionary where the key is the filter band and the value
            the Poisson counts, a dictionary where the key is the component and
            the value a dictionary of (filter_band, Poisson counts) pairs, or a
            tuple of such dictionaries.

        Examples
        --------
        import matplotlib.pyplot as plt
        import torch
        from galkit.functional import fits2jpeg
        from galkit.spatial import grid
        from galsyn.sky_detector import SkyDetectorGenerator
        from galsyn.galaxy.dataset import Gadotti

        filter_bands = 'irg'
        size  = 5
        shape = (128,128)

        sky = SkyDetectorGenerator()
        sky.sample(size)

        galaxy = Gadotti()
        galaxy.sample(size)
        galaxy.convert_magnitude_to_flux(plate_scale=0.396)
        isoA = galaxy.get_isoMag(value=24.5, filter_band='r')

        profile  = galaxy.get_component_profile(filter_bands)
        geometry = galaxy.get_component_geometry(grid.pytorch_grid(shape), isoA)

        component_flux  = galaxy.get_component_flux(geometry=geometry, profile=profile, filter_bands=filter_bands)
        component_cnts  = galaxy.convert_flux_to_counts(component_flux, sky)
        component_noise = galaxy.apply_shot_noise_to_counts(component_cnts, sky)

        galaxy_flux  = galaxy.add_component_flux(component_flux)
        galaxy_cnts  = galaxy.convert_flux_to_counts(galaxy_flux, sky)
        galaxy_noise = galaxy.apply_shot_noise_to_counts(galaxy_cnts, sky)

        def foo(i):
            image = torch.cat([galaxy_noise[i][k] for k in filter_bands])

            fig, ax = plt.subplots()
            ax.imshow(fits2jpeg(image).T)
            fig.show()

        for i in range(size):
            foo(i)
        """
        if isinstance(counts, (tuple, list)):
            return tuple(self.apply_shot_noise_to_counts(c,sky_detector,i if index is None else index) for i,c in enumerate(counts))
        
        kwargs = {
            'column'    : sky_detector.gain_column,
            'dictionary': sky_detector.dictionary,
            'dataframe' : sky_detector.data,
            'index'     : index,
            'device'    : self.device,
        }

        output = {}
        for k,v in counts.items():
            if isinstance(v, dict):
                output[k] = {}
                for kv, vv in v.items():
                    gain = accessor(filter_band=kv, **kwargs)
                    output[k][kv] = torch.poisson(vv * gain) / gain
            else:
                gain = accessor(filter_band=k, **kwargs)
                output[k] = torch.poisson(v * gain) / gain

        return output


    def convert_counts_to_flux(self, 
        counts       : Union[Dict, Tuple[Dict]],
        sky_detector : SkyDetectorGenerator, 
        index        : Optional[int] = None
    ) -> Union[Dict, Tuple[Dict]]:
        """
        Converts counts to flux units.

        Parameters
        ----------
        counts : Dict, Tuple[Dict]
            Either a dictionary where the key is the filter band and the value
            the counts, a dictionary where the key is the component and the value
            a dictionary of (filter_band, counts) pairs, or a tuple of such
            dictionaries.

        sky_detector : SkyDetectorGenerator
            A sky-detector module that contains information pertaining
            to the flux-to-count conversion rate.

        index : int, optional
            The index in sky_detector to use for extracting the flux to
            count conversion rates. If flux is a tuple, then a value of
            `None` will result in matching between the flux index and
            the sky_detector index.

        Returns
        -------
        flux : Dict, Tuple[Dict]
            Either a dictionary where the key is the filter band and the value
            the flux, a dictionary where the key is the component and the value
            a dictionary of (filter_band, flux) pairs, or a tuple of such
            dictionaries.

        Examples
        --------
        import matplotlib.pyplot as plt
        import torch
        from galkit.spatial import grid
        from galkit.functional import fits2jpeg
        from galsyn.sky_detector import SkyDetectorGenerator
        from galsyn.galaxy.dataset import Gadotti

        filter_bands = 'irg'
        size  = 5
        shape = (128,128)

        sky = SkyDetectorGenerator()
        sky.sample(1)  # Use the same sky + detector noise for each 

        galaxy = Gadotti()
        galaxy.sample(size)
        galaxy.convert_magnitude_to_flux(plate_scale=0.396)
        isoA = 2*galaxy.get_petroRad(value=0.9, filter_band='r', index=None)

        profile  = galaxy.get_component_profile(filter_bands)
        geometry = galaxy.get_component_geometry(grid.pytorch_grid(shape), scale=isoA)

        component_flux = galaxy.get_component_flux(geometry=geometry, profile=profile, filter_bands=filter_bands)
        component_cnts = galaxy.convert_flux_to_counts(component_flux, sky, index=0)
        component_flux = galaxy.convert_counts_to_flux(component_cnts, sky, index=0)

        galaxy_flux = galaxy.add_component_flux(component_flux)
        galaxy_cnts = galaxy.convert_flux_to_counts(galaxy_flux, sky, index=0)
        galaxy_flux = galaxy.convert_counts_to_flux(galaxy_cnts, sky, index=0)

        def foo(i):
            image = torch.cat([galaxy_cnts[i][k] for k in filter_bands])
            image = fits2jpeg(image)
            
            fig, ax = plt.subplots()
            ax.imshow(image.T)
            fig.show()
        
        for i in range(size):
            foo(i)
        """
        if isinstance(counts, (tuple, list)):
            return tuple(self.convert_counts_to_flux(f,sky_detector,i if index is None else index) for i,f in enumerate(counts))

        keys = {
            'column'    : sky_detector.flux_per_count_column,
            'dictionary': sky_detector.dictionary,
            'dataframe' : sky_detector.data,
            'index'     : index,
            'device'    : self.device,
        }

        output = {}
        for k,v in counts.items():
            if isinstance(v, dict):
                output[k] = {}
                for kv, vv in v.items():
                    output[k][kv] = vv * accessor(filter_band=kv, **keys)
            else:
                output[k] = v * accessor(filter_band=k, **keys)

        return output

    def convert_flux_to_counts(self,
        flux         : Union[Dict, Tuple[Dict]],
        sky_detector : SkyDetectorGenerator, 
        index        : Optional[int] = None
    ) -> Union[Dict, Tuple[Dict]]:
        """
        Converts flux units to counts.

        Parameters
        ----------
        flux : Dict, Tuple[Dict]
            Either a dictionary where the key is the filter band and the value
            the flux, a dictionary where the key is the component and the value
            a dictionary of (filter_band, flux) pairs, or a tuple of such
            dictionaries.

        sky_detector : SkyDetectorGenerator
            A sky-detector module that contains information pertaining
            to the flux-to-count conversion rate.

        index : int, optional
            The index in sky_detector to use for extracting the flux to
            count conversion rates. If flux is a tuple, then a value of
            `None` will result in matching between the flux index and
            the sky_detector index.

        Returns
        -------
        counts : Dict, Tuple[Dict]
            Either a dictionary where the key is the filter band and the value
            the counts, a dictionary where the key is the component and the value
            a dictionary of (filter_band, counts) pairs, or a tuple of such
            dictionaries.      

        Examples
        --------
        import matplotlib.pyplot as plt
        import torch
        from galkit.spatial import grid
        from galkit.functional import fits2jpeg
        from galsyn.sky_detector import SkyDetectorGenerator
        from galsyn.galaxy.dataset import Gadotti

        filter_bands = 'irg'
        size  = 5
        shape = (128,128)

        sky = SkyDetectorGenerator()
        sky.sample(1)  # Use the same sky + detector noise for each 

        galaxy = Gadotti()
        galaxy.sample(size)
        galaxy.convert_magnitude_to_flux(plate_scale=0.396)
        isoA = 2*galaxy.get_petroRad(value=0.9, filter_band='r', index=None)

        profile  = galaxy.get_component_profile(filter_bands)
        geometry = galaxy.get_component_geometry(grid.pytorch_grid(shape), scale=isoA)

        component_flux = galaxy.get_component_flux(geometry=geometry, profile=profile, filter_bands=filter_bands)
        component_cnts = galaxy.convert_flux_to_counts(component_flux, sky, index=0)

        galaxy_flux = galaxy.add_component_flux(component_flux)
        galaxy_cnts = galaxy.convert_flux_to_counts(galaxy_flux, sky, index=0)
 
        def foo(i):
            image = torch.cat([galaxy_cnts[i][k] for k in filter_bands])
            image = fits2jpeg(image)
            
            fig, ax = plt.subplots()
            ax.imshow(image.T)
            fig.show()
        
        for i in range(size):
            foo(i)
        """
        if isinstance(flux, (tuple, list)):
            return tuple(self.convert_flux_to_counts(f,sky_detector,i if index is None else index) for i,f in enumerate(flux))

        keys = {
            'column'    : sky_detector.flux_per_count_column,
            'dictionary': sky_detector.dictionary,
            'dataframe' : sky_detector.data,
            'index'     : index,
            'device'    : self.device,
        }

        output = {}
        for k,v in flux.items():
            if isinstance(v, dict): # Components
                output[k] = {}
                for kv, vv in v.items():
                    output[k][kv] = vv / accessor(filter_band=kv, **keys)
            else:
                output[k] = v / accessor(filter_band=k, **keys)
        return output

    def convolve(self,
        input        : Union[Dict, Tuple[Dict]],
        sky_detector : SkyDetectorGenerator,
        plate_scale  : float,
        index        : Optional[int] = None,
    ) -> Union[Dict, Tuple[Dict]]:
        """
        Convolves the input based on the provided convolution method
        and the parameters contained in the sky_detector copula.

        Parameters
        ----------
        input : Dict, Tuple[Dict]
            Either a dictionary where the key is the filter band and the value
            a tensor, a dictionary where the key is the component and the value
            is a dictionary where the key is the filter band and the value a
            tensor, or a tuple of such dictionaries.

        sky_detector : SkyDetectorGenerator
            A sky-detector module that contains information pertaining
            to the flux-to-count conversion rate.

        plate_scale : float
            The plate_scale scale of the image in arcseconds / pixel

        index : int, optional
            The index in sky_detector to use for extracting the flux to
            count conversion rates. If flux is a tuple, then a value of
            `None` will result in matching between the flux index and
            the sky_detector index.        

        Returns
        -------
        output : Dict, Tuple[Dict]
            Either a dictionary where the key is the filter band and the value
            a tensor, a dictionary where the key is the component and the value
            is a dictionary where the key is the filter band and the value a
            tensor, or a tuple of such dictionaries. The value represents the
            convolved input.

        Examples
        --------
        import matplotlib.pyplot as plt
        import torch
        from galkit.functional import fits2jpeg
        from galkit.spatial import grid
        from galsyn.sky_detector import SkyDetectorGenerator
        from galsyn.galaxy.dataset import Gadotti as Galaxy

        filter_bands = 'irg'
        size  = 5
        shape = (128,128)

        sky = SkyDetectorGenerator()
        sky.sample(size)

        galaxy = Galaxy()
        galaxy.sample(size)
        galaxy.convert_magnitude_to_flux(plate_scale=0.396)
        isoA = galaxy.get_isoMag(value=24.5, filter_band='r', index=None)

        profile  = galaxy.get_component_profile(filter_bands)
        geometry = galaxy.get_component_geometry(grid.pytorch_grid(shape), isoA)
        component_flux = galaxy.get_component_flux(geometry=geometry, profile=profile, filter_bands=filter_bands)
        galaxy_flux = galaxy.add_component_flux(component_flux)
        galaxy_cnts = galaxy.convert_flux_to_counts(galaxy_flux, sky_detector=sky)
        convolved_cnts = galaxy.convolve(galaxy_cnts, sky, 0.396)

        def foo(i):
            image1 = torch.cat([galaxy_cnts[i][k] for k in filter_bands])
            image1 = fits2jpeg(image1)
            
            image2 = torch.cat([convolved_cnts[i][k] for k in filter_bands])
            image2 = fits2jpeg(image2)

            fig, ax = plt.subplots(ncols=2)
            ax[0].imshow(image1.T)
            ax[1].imshow(image2.T)
            fig.show()
        
        for i in range(size):
            foo(i)
        """
        if isinstance(input, (tuple, list)):
            return tuple(self.convolve(x, sky_detector, plate_scale, i if index is None else index) for i,x in enumerate(input))

        kwargs = {
            'sky_detector': sky_detector,
            'plate_scale' : plate_scale,
            'index'       : index,
        }

        output = {}
        for k,v in input.items():
            if isinstance(v, dict):
                output[k] = {}
                for kv, vv in v.items():
                    output[k][kv] = self.psf_model(vv, filter_band=kv, **kwargs)
            else:
                output[k] = self.psf_model(v, filter_band=k, **kwargs)
        return output


    def downscale(self, 
        input  : Union[torch.Tensor, Dict, Tuple[Dict]],
        factor : int
    ) -> Union[torch.Tensor, Dict, Tuple[Dict]]:
        """
        Downscales the input tensors by the specified factor.

        Parameters
        ----------
        input : Tensor, Dict, Tuple[Dict]
            Either a Tensor, a dictionary where the values are the Tensors to
            downscale, or a tuple of such objects.

        factor : int
            The downscaling factor. The tensors spatial dimensions
            are reduced by this factor.

        Returns
        -------
        output : Tensor, Dict, Tuple[Dict]
            Either the downsampled Tensor, a dictionary of the downsampled Tensors,
            or a tuple of such objects.
        
        Examples
        --------
        import torch
        from galsyn.galaxy.dataset import Gadotti

        input = {
            'g' : torch.empty(1,100,100),
            'r' : torch.empty(2,150,150),
            'i' : torch.empty(3,200,200)
        }

        galaxy = Gadotti()
        output = galaxy.downscale(input, 2)

        for k,v in output.items():
            print(k, v.shape)
        """
        if isinstance(input, (tuple, list)):
            return tuple(self.downscale(i,factor) for i in input)
        
        if isinstance(input, torch.Tensor):
            return resample.downscale_local_mean(input,factor) if input.size(0) != 0 else input

        else:
            output = {}
            for k,v in input.items():
                if isinstance(v, dict):
                    output[k] = {}
                    for kv,vv in v.items():
                        output[k][kv] = resample.downscale_local_mean(vv,factor) if vv.size(0) != 0 else vv
                else:
                    output[k] = resample.downscale_local_mean(v,factor) if v.size(0) != 0 else v

            return output

    def get_component_flux(self, 
        geometry     : Union[Dict, Tuple[Dict]], 
        profile      : Union[Dict, Tuple[Dict]], 
        filter_bands : Iterable
    ) -> Union[Dict, Tuple[Dict]]:
        """
        Generates the model flux given the geometry and profiles in each
        of the filter bands.

        Parameters
        ----------
        geometry : Dict, Tuple[Dict]
            Either a dictionary where the key is the component and the value
            is another dictionary where the key is the coordinate parameter
            (e.g. r, θ) and the value is a tensor containing the grid values
            or a tuple containing such dictionaries.

        profile : Dict, Tuple[Dict]
            Either a dictionary where the key is the component and the value
            is the model profile or a tuple containing such dictionaries.

        filter_bands : Iterable
            The filter bands to generate the flux for.

        Returns
        -------
        flux : Dict, Tuple[Dict]
            Either a dictionary where the key is the component and the value
            is another dictionary where the key is the filter band and the value
            the flux or a tuple containing such dictionaries.

        Examples
        --------
        import matplotlib.pyplot as plt
        import torch
        from galkit.spatial import grid
        from galkit.functional import arcsinh_stretch
        from galsyn.galaxy.dataset import Gadotti

        filter_bands = 'irg'
        size  = 5
        shape = (128,128)

        galaxy = Gadotti()
        galaxy.sample(size)
        galaxy.convert_magnitude_to_flux(plate_scale=0.396)
        isoA = galaxy.get_petroRad(value=0.9, filter_band='r', index=None)

        profile  = galaxy.get_component_profile(filter_bands)
        geometry = galaxy.get_component_geometry(grid.pytorch_grid(shape), scale=isoA)
        flux     = galaxy.get_component_flux(geometry=geometry, profile=profile, filter_bands=filter_bands)

        def foo(i, c='disk'):
            image = torch.cat([flux[i][c][k] for k in filter_bands])
            image = arcsinh_stretch(image)
            
            fig, ax = plt.subplots()
            ax.imshow(image.T)
            fig.show()
        
        for i in range(size):
            foo(i)
        """
        if isinstance(profile, (tuple, list)):
            return tuple(self.get_component_flux(geometry=g, profile=p, filter_bands=filter_bands) for g,p in zip(geometry, profile))

        components = profile.keys()

        output = {}
        for c in components:
            output[c] = {k:profile[c](**geometry[c], filter_band=k) for k in filter_bands}

        return output

    def get_component_profile(self, 
        filter_bands : Iterable, 
        index        : Optional[int] = None
    ) -> Union[Dict, Tuple[Dict]]:
        """
        Generates a dictionary containing the profiles for each component.
        If `index=None`, then a tuple is returned containing the profiles
        for each galaxy.

        Parameters
        ----------
        filter_bands : Iterable
            The filter bands for with to generate model profiles

        index : int, optional
            The element in `self.data` for which to use to generate the
            model profile. If None, then a profile is generated for each
            value in the data frame.

        Returns
        -------
        profile : Dict, Tuple[Dict]
            Either a dictionary where the key is the component and the value
            is the model profile or a tuple containing such dictionaries.

        Examples
        --------
        from galsyn.galaxy.dataset import Gadotti

        galaxy = Gadotti()
        galaxy.sample(5)
        galaxy.convert_magnitude_to_flux(plate_scale=0.396)
        profiles = galaxy.get_component_profile(filter_bands='irg', index=0)

        print(profiles)
        """
        if index is None:
            return tuple(self.get_component_profile(filter_bands, index=i) for i in range(len(self)))

        components = self.get_components()

        profiles = {}
        for c in components:
            args = {}
            for filter_band in filter_bands:
                params = self.get_component_parameters(c, filter_band, index)

                for p,v in params.items():
                    if p not in args:
                        args[p] = {}
                    args[p][filter_band] = v

            # The model profile may be a dictionary so that each component has a separate model (e.g. Sersic, Ferrer)
            profile = self.profile[c] if isinstance(self.profile, dict) else self.profile
            profiles[c] = profile(**args)

        return profiles