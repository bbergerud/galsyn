"""
Methods for generating procedual noise.

Classes
-------
PowerTransform
    Applies the transformation
        multipler ⋅ [(1 + input)/2]^power
    to the input. Useful for applying to procedural noise.

PerlinNoise
    Class interface to the perlin2d_octaves function.

Disk
    Interface to PerlinNoise where the default parameters are set up to apply
    smoother fluctuations to the disk. 

Dust
    Interface to PerlinNoise with a dust extinction transformation.
    The extinction values are renormalized to have a mean of 1 to
    help conserve the flux.

HII
    Interface to PerlinNoise where the default parameters are set up to mimic
    the appearance of star forming regions.

Perturbation
    Class for incorporating procedural noise into the flux.
"""

import random
import torch
from dataclasses import dataclass
from scipy.stats import truncnorm
from typing import Dict, Iterable, Optional, Tuple, Union
from .dust import DustModel, CCM89_OpticalIR
from ..random import perlin2d_octaves

def get_truncnorm(loc, scale, a, b):
    a, b = (a - loc) / scale, (b - loc) / scale
    return truncnorm(a=a, b=b, loc=loc, scale=scale).rvs(1).item()

@dataclass
class PowerTransform:
    """
    Applies the transformation

        multipler ⋅ [(1 + input)/2]^power
    
    to the input. Useful for applying to procedural noise.

    Parameters
    ----------
    power : float, callable
        The power index. Can be either a float or a function
        that returns a float.

    multiplier : float, callable, optional
        The multiplication factor. Can be either a float or
        a function that returns a float. If None, then no
        multiplication factor is applied to the transformation. 
    """
    power : Union[float, callable]
    multiplier : Optional[Union[float, callable]] = None

    def __call__(self, input:torch.Tensor) -> torch.Tensor:
        power = self.power() if callable(self.power) else self.power
        output = (1 + input).mul(0.5).pow(power)

        if self.multiplier is not None:
            multiplier = self.multiplier() if callable(self.multiplier) else self.multiplier
            output *= multiplier

        return output

@dataclass
class PerlinNoise:
    """
    Class interface to the perlin2d_octaves function.

    Parameters
    ----------
    octaves : callable, int
        The number of coherent noise functions to add together. A smaller
        number will have coarser features. Can be either an integer or a
        function that returns an integer.

    resolution : callable, Tuple[float, float]
        The relative resolution compared to the image size. Features
        will have  a size comparable to the this fractional value of
        the image size. Can be either a tuple of floats or a function
        that returns a tuple of floats.
    
    shear : callable, float
        The fractional value of the rotation to apply. A value of 1 will
        cause features to strongly follow the rotation pattern while a 
        value of zero will remove the rotational effect. Can be either a
        float or a function that returns a float. Only applied if a rotation
        parameter is supplied.

    transform : callable, optional
        The transformation function to apply to the noise map.
    """
    octaves     : Union[callable, int] = 10
    resolution  : Union[callable, Tuple[float,float]] = (0.05, 0.05)
    shear       : Union[callable, float] = lambda : random.uniform(0, 1)
    transform   : Optional[callable] = None

    def __call__(self, shape:Tuple[int,int], **kwargs) -> torch.Tensor:
        """
        Returns the 2D perlin noise map.

        Parameters
        ----------
        shape : Tuple[int,int]
            The image size.
        
        **kwargs
            Any additional arguments to pass into the perlin2d_octaves function.

        Returns
        -------
        output : Tensor
            The 2D noise map.
        """
        return perlin2d_octaves(
            shape      = shape,
            octaves    = self.octaves() if callable(self.octaves) else self.octaves,
            shear      = self.shear() if callable(self.shear) else self.shear,
            resolution = self.resolution() if callable(self.resolution) else self.resolution,
            transform  = self.transform,
            **kwargs,
        )

class Disk(PerlinNoise):
    """
    Interface to PerlinNoise where the default parameters are set up to apply
    smoother fluctuations to the disk. 

    Parameters
    ----------
    fraction : 
        The fractional ratio of the noise to the perturbation. Can be either
        a float or a callable function that returns a float. A value of 1 means
        that the perturbation is entirely driven by the noise, while a value of
        zero means that no perturbation is applied.

    shear : callable, float
        The fractional value of the rotation to apply. A value of 1 will
        cause features to strongly follow the rotation pattern while a 
        value of zero will remove the rotational effect. Can be either a
        float or a function that returns a float. Only applied if a rotation
        parameter is supplied.

    transform : callable, optional
        The transformation function to apply to the noise map.
    
    **kwargs
        Any additional argument to pass into the parent class constructor    
    """
    def __init__(self,
        fraction  : Union[callable, float] = lambda : random.uniform(0.00, 1.00),
        shear     : Union[callable, float] = lambda : get_truncnorm(0.5, 0.1, 0, 1),
        transform : Optional[callable]     = PowerTransform(lambda : get_truncnorm(1, 0.25, 0, 2)),
        **kwargs,
    ):
        self.fraction = fraction
        super().__init__(shear=shear, transform=transform, **kwargs)

    def __call__(self, *args, filter_bands=None, **kwargs) -> torch.Tensor:
        output = super().__call__(*args, **kwargs)
        f = self.fraction() if callable(self.fraction) else self.fraction
        return (1 - f) + output * (f / output.mean())

class Dust(PerlinNoise):
    """
    Interface to PerlinNoise with a dust extinction transformation.
    The extinction values are renormalized to have a mean of 1 to
    help conserve the flux.

    Parameters
    ----------
    dust_model : DustModel
        Dust model for generating extinction 

    shear : callable, float
        The fractional value of the rotation to apply. A value of 1 will
        cause features to strongly follow the rotation pattern while a 
        value of zero will remove the rotational effect. Can be either a
        float or a function that returns a float. Only applied if a rotation
        parameter is supplied.

    transform : callable, optional
        The transformation function to apply to the noise map.

    **kwargs
        Any additional arguments to supply to the parent class.
    """
    def __init__(self, 
        dust_model  : DustModel = CCM89_OpticalIR(),
        shear       : Union[callable, float] = lambda : get_truncnorm(0.5, 0.1, 0, 1.0),
        transform   : Optional[callable] = PowerTransform(lambda : random.uniform(0.5, 5.0), lambda : get_truncnorm(1, 0.25, 0, 3)),
        **kwargs
    ):
        self.dust_model = dust_model
        super().__init__(shear=shear, transform=transform, **kwargs)

    def __call__(self, *args, filter_bands:Iterable='zirgu', **kwargs) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary containing the perturbation tensors
        for each of the filter bands.

        Parameters
        ----------
        filter_bands : Iterable
            The sequence of filter bands to generate extinction values.
        
        *args, **kwargs
            Additional arguments to pass into the __call__ function
            from the parent class.

        Returns
        -------
        dust : Dict[str, Tensor]
            A dictionary where the keys are the filter bands and the
            values a tensor containing the perturbation. The tensors
            have a mean value of 1 to help perserve the total flux.
        """
        dust = super().__call__(*args, **kwargs)
        dust = self.dust_model(filter_bands=filter_bands, Av=dust)
        dust = {k:v/v.mean() for k,v in dust.items()}

        return dust

class HII(PerlinNoise):
    """
    Interface to PerlinNoise where the default parameters are set up to mimic
    the appearance of star forming regions.

    Parameters
    ----------
    fraction : 
        The fractional ratio of the noise to the perturbation. Can be either
        a float or a callable function that returns a float. A value of 1 means
        that the perturbation is entirely driven by the noise, while a value of
        zero means that no perturbation is applied.

    shear : callable, float
        The fractional value of the rotation to apply. A value of 1 will
        cause features to strongly follow the rotation pattern while a 
        value of zero will remove the rotational effect. Can be either a
        float or a function that returns a float. Only applied if a rotation
        parameter is supplied.

    transform : callable, optional
        The transformation function to apply to the noise map.
    
    **kwargs
        Any additional argument to pass into the parent class constructor.
    """
    def __init__(self, 
        fraction  : Union[callable, float] = lambda : get_truncnorm(0.1, 0.05, 0, 0.3),
        shear     : Union[callable, float] = lambda : get_truncnorm(0.1, 0.05, 0, 1.0),
        transform : Optional[callable]     = PowerTransform(lambda : random.uniform(5, 10)),
        **kwargs,
    ):
        self.fraction = fraction
        super().__init__(shear=shear, transform=transform, **kwargs)

    def __call__(self, *args, filter_bands=None, **kwargs) -> torch.Tensor:
        output = super().__call__(*args, **kwargs)
        f = self.fraction() if callable(self.fraction) else self.fraction
        return (1 - f) + output * (f / output.mean())

class Perturbation:
    """
    Class for incorporating procedural noise into the flux.

    Methods
    -------
    add_component_perturbation(flux, geometry, scale, rotation)
        Adds procedural noise to the image. The outputs from the procedural noise
        functions are multiplied with the flux to form the modified flux.

    Required Attributes
    -------------------
    device : torch.device
        The device on which to generate tensor data.

    perturbation : Dict[str, Iterable
        The set of perturbations. The keys should be the components to apply
        the perturbation to, separated by an underscore "_", while the values
        should be an iterable containing the different procedural noise methods.
    """
    def add_component_perturbation(self,
        flux         : Union[Dict, Tuple[Dict]],
        geometry     : Union[Dict, Tuple[Dict]],
        scale        : Union[float, torch.Tensor],
        rotation     : Union[Dict, Tuple[Dict]] = None,
    ) -> Union[Dict, Tuple[Dict]]:
        """
        Adds procedural noise to the image. The outputs from the procedural noise
        functions are multiplied with the flux to form the modified flux.

        Parameters
        ----------
        flux : Dict, Tuple[Dict]
            Either a dictionary where the key is the filter band and the value
            the flux, a dictionary where the key is the component and the value
            a dictionary of (filter_band, flux) pairs, or a tuple of such
            dictionaries.

        geometry : Dict, Tuple[Dict]
            Either a dictionary where the key is the component and the value
            is another dictionary where the key is the coordinate parameter
            (e.g. r, θ) and the value is a tensor containing the grid values
            or a tuple containing such dictionaries. 

        scale : float, tensor
            The scaling factor applied to the radial coordinates. Used to help
            normalize the noise scale.

        rotation : callable, Tuple[callable]
            Either a callable function that returns the spiral pattern of a
            tuple of such functions.

        Returns
        -------
        flux : Dict, Tuple[Dict]
            Either a dictionary where the key is the filter band and the value
            the flux, a dictionary where the key is the component and the value
            a dictionary of (filter_band, flux) pairs, or a tuple of such
            dictionaries.
        """
        if isinstance(flux, (tuple, list)):
            return tuple(self.add_component_perturbation(
                flux[i],
                geometry[i],
                scale[i] if isinstance(scale, Iterable) else scale,
                None if rotation is None else rotation[i],
            ) for i,_ in enumerate(flux))

        if self.perturbation is not None:
            for key, value in self.perturbation.items():
                components = key.split('_')
                
                for perturbation in value:
                    p = None
                    for subcomponent in components:
                        for k,v in flux.items():
                            if k.startswith(subcomponent):
                                for (kk,vv) in v.items():
                                    if p is None:
                                        p = perturbation(shape=vv.shape[-2:], filter_bands=flux[k].keys(), device=self.device, grid_kwargs={'scale': scale}, rotation=rotation, **geometry[subcomponent])
                                    flux[k][kk] = (p[kk] if isinstance(p,dict) else p) * vv

        return flux