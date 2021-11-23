import torch
from typing import Dict, Optional, Tuple, Union
from ..sky_detector import SkyDetectorGenerator

class S2N:
    """
    Class container for calculating the noise and signal-to-noise ratios.

    Methods
    -------
    get_noise_level(flux, sky_detector, plate_scale, index, output_count)
        Returns the noise associated with the sky and detector noise
        as well as the source flux. Mostly an interface to the
        `sky_detector.noise_level` method.

    get_s2n(flux, noise_level, sky_detector, component, index, filter_band, operation)
        Returns the signal-to-noise ratio of the designated component.

    Required Attributes
    -------------------
    device : torch.device
        The device on which to generate tensor data.
    """
    def get_noise_level(self,
        flux          : Union[Dict, Tuple[Dict]],
        sky_detector  : SkyDetectorGenerator,
        plate_scale   : float,
        index         : Optional[int] = None,
        output_counts : bool = True,
    ) -> Union[Dict, Tuple[Dict]]:
        """
        Returns the noise associated with the sky and detector noise
        as well as the source flux. Mostly an interface to the
        `sky_detector.noise_level` method.

        Parameters
        ----------
        flux : Dict, Tuple[Dict]
            Either a dictionary where the key is the filter band and the value
            the summed flux of all the components or a tuple of such dictionaries.

        sky_detector : SkyDetectorGenerator
            A sky-detector module.

        plate_scale : float
            The plate_scale scale of the image in arcseconds / pixel

        index : int, optional
            The index in sky_detector to use for extracting the flux to
            count conversion rates. If flux is a tuple, then a value of
            `None` will result in matching between the flux index and
            the sky_detector index.            

        output_counts : bool
            Boolean indicating to output the noise in units of counts.
            Default is True.

        Returns
        ------- : 
        noise_level : Dict, Tuple[Dict]
            Either a dictionary where the key is the filter band and the value
            the noise level or a tuple of such dictionaries.

        Examples
        --------
        import matplotlib.pyplot as plt
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

        component_flux = galaxy.get_component_flux(geometry=geometry, profile=profile, filter_bands=filter_bands)
        component_cnts = galaxy.convert_flux_to_counts(component_flux, sky)
        galaxy_cnts    = galaxy.add_component_flux(component_cnts)

        noise_level = galaxy.get_noise_level(
            flux = galaxy_cnts,
            sky_detector = sky,
            plate_scale = 0.396,
        ) 

        def foo(i):
            fig, ax = plt.subplots()
            ax.imshow(noise_level[i]['r'].squeeze())
            fig.show()

        for i in range(size):
            foo(i)
        """
        if isinstance(flux, (tuple, list)):
            return tuple(self.get_noise_level(
                flux = f,
                sky_detector = sky_detector,
                plate_scale = plate_scale,
                index = i if index is None else index,
                output_counts = output_counts,
            ) for i,f in enumerate(flux))

        output = {}
        for k,v in flux.items():
            output[k] = sky_detector.noise_level(
                flux_counts = v,
                filter_band = k,
                plate_scale = plate_scale,
                index = index,
                output_counts = output_counts,
            )
        return output

    def get_s2n(self,
        flux         : Union[Dict, Tuple[Dict]],
        noise_level  : Union[Dict, Tuple[Dict]],
        sky_detector : SkyDetectorGenerator,
        component    : Optional[str] = None,
        index        : Optional[int] = None,
        filter_band  : Optional[str] = None,
        operation    : Optional[callable] = None,
    ) -> Union[Dict, Tuple[Dict]]:
        """
        Returns the signal-to-noise ratio of the designated component.

        Parameters
        ----------
        flux : Dict, Tuple[Dict]
            Either a dictionary where the key is the filter band and the value
            the flux, a dictionary where the key is the component and the value
            a dictionary of (filter_band, flux) pairs, or a tuple of such
            dictionaries.

        noise_level : Dict, Tuple[Dict]
            Either a dictionary where the key is the filter band and the value
            the noise level of a tuple of such dictionaries.

        sky_detector : SkyDetectorGenerator
            A sky-detector module.

        component : str, optional
            The name of the component to generate the S/N ratio for. If set
            to None, then the dictionary should just contain the filter band
            keys.
        
        index : int, optional
            The index in sky_detector to use for extracting the flux to
            count conversion rates. If flux is a tuple, then a value of
            `None` will result in matching between the flux index and
            the sky_detector index.

        filter_band : str, optional
            The filter band for which to calculate the S/N ratio. If None,
            then the S/N ratio is calculate for all the filter bands in
            the flux dictionary.

        operation : callable, optional
            Operation to merge the S/N ratio across multiple filter bands.

        Returns
        -------
        s2n : Tensor, Dict, Tuple[Tensor], Tuple[Dict]
            The signal to noise ratio. Can be either a tensor containing the
            merged S/N ratio, a dictionary of S/N ratios where the keys are
            the filter bands, or a tuple of such objects.

        Examples
        --------
        import matplotlib.pyplot as plt
        from galkit.spatial import grid
        from galsyn.sky_detector import SkyDetectorGenerator
        from galsyn.galaxy import Gadotti as Galaxy, load_local_generator

        filter_bands = 'irg'
        size  = 5
        shape = (128,128)

        sky = SkyDetectorGenerator()
        sky.sample(size)

        galaxy = Galaxy(load_local_generator('gadotti_2009_bar_bulge_disk.pkl'))
        galaxy.sample(size)
        galaxy.convert_magnitude_to_flux(plate_scale=0.396)
        isoA = galaxy.get_isoMag(value=24.5, filter_band='r')

        profile  = galaxy.get_component_profile(filter_bands)
        geometry = galaxy.get_component_geometry(grid.pytorch_grid(shape), isoA)

        component_flux = galaxy.get_component_flux(geometry=geometry, profile=profile, filter_bands=filter_bands)
        component_cnts = galaxy.convert_flux_to_counts(component_flux, sky)
        galaxy_cnts    = galaxy.add_component_flux(component_cnts)

        noise_level = galaxy.get_noise_level(
            flux = galaxy_cnts,
            sky_detector = sky,
            plate_scale = 0.396,
        ) 

        bar_s2n = galaxy.get_s2n(component='bar', flux=component_cnts, noise_level=noise_level, sky_detector=sky)

        def foo(i):
            fig, ax = plt.subplots()
            ax.imshow(bar_s2n[i]['r'].squeeze())
            fig.show()

        for i in range(size):
            foo(i)
        """
        if isinstance(flux, (tuple, list)):
            return tuple(self.get_s2n(
                flux = f,
                noise_level = n, 
                sky_detector = sky_detector,
                component = component, 
                index = i if index is None else index,
                operation = operation,
            ) for i, (f,n) in enumerate(zip(flux, noise_level)))

        if component is not None:
            if component not in flux:
                empty = torch.tensor([], device=self.device)
                return empty if operation is not None else {k:empty for k in flux.keys()}
            flux = flux[component]

        output = {}
        for k,v in flux.items():
            if (filter_band is None) or (k == filter_band):
                output[k] = v / noise_level[k]

        if operation is not None:
            output = torch.stack([v for v in output.values()], dim=0)
            output = operation(output, dim=0)

        return output