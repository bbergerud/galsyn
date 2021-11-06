"""
Metrics for the flux distribution.

Classes
-------
IsoFlux
    Method for calculating the radius at which the profile flux
    drops to the indicated value.

IsoMag
    Method for calculating the radius at which the profile flux
    drops to the indicated magnitude.

PetroRad
    Method for calculating the Petrosian radius.

Photometric
    Class for calculating photometric quantities.
"""
import torch
import torchquad
from dataclasses import dataclass
from galkit.functional.magnitude import sdss_flux2mag
from typing import Dict, Iterable, Optional, Tuple, Union

class IsoFlux:
    """
    Method for calculating the radius at which the profile flux
    drops to the indicated value.
    """
    def __call__(self, 
        r       : torch.Tensor,
        profile : callable,
        value   : float,
        *args, **kwargs,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        r : tensor
            The radial distances over which to evaluate the function.

        profile : callable
            The profile function.

        value : float
            The desired flux value of the profile.

        Returns
        -------
        r : tensor
            The closest radial distance to the desired value.
        """
        flux = profile(r).squeeze()
        loc  = torch.argmin((flux - value).abs())
        return r[loc]

@dataclass
class IsoMag:
    """
    Method for calculating the radius at which the profile flux
    drops to the indicated magnitude.

    Parameters
    ----------
    flux2mag : callable
        A function that takes as input the flux and filter band
        and returns the magnitude.
    """
    flux2mag : callable = sdss_flux2mag

    def __call__(self, 
        r           : torch.Tensor,
        profile     : callable,
        value       : float,
        filter_band : str,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        r : tensor
            The radial distances over which to evaluate the function.

        profile : callable
            The profile function.

        value : float
            The desired magnitude of the profile.

        filter_band : str
            The filter band that the profile is evaluated in.

        Returns
        -------
        r : tensor
            The closest radial distance to the desired value.
        """
        mag = self.flux2mag(profile(r), filter_band=filter_band).squeeze()
        loc = torch.argmin((mag - value).abs())
        return r[loc]

@dataclass
class PetroRad:
    """
    Method for calculating the Petrosian radius.

    Parameters
    ----------
    method : callable
        The integration technique. Expected to be a torchquad
        element.

    N : int
        The number of sample points to use for the iteration.
    """
    method : callable = torchquad.Trapezoid()
    N      : int = 5

    def __call__(self, 
        r       : torch.Tensor,
        profile : callable,
        value   : float,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        r : tensor
            The radial distances over which to evaluate the function. For accurate values,
            it should start r=0 and end at a point that captures nearly all the flux.

        profile : callable
            The profile function.

        value : float
            The quantile representing the cumulative light distribution. A value
            of 0.9 corresponds to 90% of the light flux.

        Returns
        -------
        r : tensor
            The closest radial distance to the desired value, which corresponds to
            one of the midpoints for the provided grid due to integration.
        """
        def func(x): # ∫ p(r) r dr dθ
            return x * profile(x).view(x.shape)

        flux = torch.zeros_like(r)
        for i, (rmin,rmax) in enumerate(zip(r[:-1], r[1:])):
            flux[i] = self.method.integrate(func, dim=1, N=self.N, integration_domain=[[rmin, rmax]]).squeeze()

        cflux = torch.cumsum(flux, -1)
        cflux = cflux / cflux[-1]

        loc = torch.argmin((cflux - value).abs())
        return 0.5*(r[1:][loc] + r[:-1][loc])

class Photometric:
    """
    Class for calculating photometric quantities.

    Methods
    -------
    get_isoA(value, metric, components, index, filter_band, profile, r)
        Calculates the radius at which the light reaches the indicated value.

    get_isoFlux(*args, metric=IsoFlux(), **kwargs)
        Interface to get_isoA with the IsoFlux metric.

    get_isoMag(*args, metric=IsoMag(), **kwargs)
        Interface to get_isoA with the IsoMag metric.

    get_petroRad(*args, metric=PetroRad(), **kwargs)
        Interface to get_isoA with the PetroRad metric.
    """

    def get_isoFlux(self, *args, metric=IsoFlux(), **kwargs) -> torch.Tensor:
        """
        Interface to get_isoA with the IsoFlux metric.
        """
        return self.get_isoA(*args, **kwargs, metric=metric)

    def get_isoMag(self, *args, metric=IsoMag(), **kwargs) -> torch.Tensor:
        """
        Interface to get_isoMag with the IsoMag metric.
        """
        return self.get_isoA(*args, **kwargs, metric=metric)

    def get_petroRad(self, *args, metric=PetroRad(), **kwargs) -> torch.Tensor:
        """
        Interface to get_isoA with the PetroRad metric.
        """
        return self.get_isoA(*args, **kwargs, metric=metric)

    def get_isoA(self,
        value       : Union[float, torch.Tensor],
        metric      : callable,
        components  : Optional[Tuple[str]] = None,
        index       : Optional[int] = None,
        filter_band : str = 'r',
        profile     : Optional[Union[Dict, Tuple[Dict]]] = None,
        r           : torch.Tensor = torch.linspace(0, 10, 100).pow(2),
    ) -> torch.Tensor:
        """
        Calculates the radius at which the light reaches the indicated value.

        Parameters
        ----------
        value : float, tensor
            The value at which to equate the metric.

        metric : callable
            A function that calculates the observational metric (e.g. petrosian radius).
            Should take as input the radial values, the profile function with only a
            radial dependence, the metric value, and the filter band.

        components : Tuple[str], optional
            A tuple containing the components to consider when generating the flux. If
            set to `None`, then all the components are used. Default is None.

        index : int, optional
            The index in `self.data` at which to construct the profile. Only
            used if `profile=None`. If both `index` and `profile` are set to None,
            then the value of all the galaxies is returned.

        filter_band : str
            The filter band at which to evaluate the flux.

        profile : Dict, Tuple[Dict]
            A dictionary of tuple of dictionary containing the model profiles for each
            component. This can be passed to avoid recreating the model profile, but is
            useful to leave as None if the filter band is not part of the flux that is
            being generated (e.g., one wants to use the 'r' band for scaling purposes
            but only return the 'g' band flux.)

        r : tensor
            The sequence of radial values at which to evaluate the flux.

        Returns
        -------
        isoA : tensor
            The radial distances roughly corresponding to where the metric is satisified.

        Examples
        --------
        from galsyn.galaxy.dataset import Gadotti as Galaxy
        from galsyn.galaxy.photometric import PetroRad

        galaxy = Galaxy()
        galaxy.sample(5)
        galaxy.convert_magnitude_to_flux(plate_scale=0.396)
        isoA = galaxy.get_isoA(value=0.9, metric=PetroRad(), filter_band='r', index=None)
        print(isoA)
        """

        r = r.to(self.device)       # Move to device

        # Generate the profiles if none were passed
        if profile is None:
            profile = self.get_component_profile(
                filter_bands=filter_band,
                index=index
            )            

        # Calculate the result for each instance
        if isinstance(profile, (list, tuple)):
            return torch.as_tensor(tuple(
                self.get_isoA(
                    r = r,
                    components = components,
                    filter_band = filter_band,
                    profile = p,
                    metric = metric,
                    value = value[i] if isinstance(value, Iterable) else value,
                ) for i,p in enumerate(profile)
            ), device=self.device)

        if components is not None:
            profile = {k:v for k,v in profile.items() if k in components}

        def func(r):
            output = 0
            for p in profile.values():
                output = output + p(r, filter_band=filter_band)
            return output

        return metric(r, func, value, filter_band)