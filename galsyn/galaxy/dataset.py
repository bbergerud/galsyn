"""
Galaxy generators using datasets.

Classes
-------
BackgroundGalaxy
    Class that provides a useful interface for constructing
    background galaxies from multiple datasets.

Dataset
    Base class for generating synthetic galaxies. The following method should be
    implemented for child classes:
        get_component_parameters(component, filter_band, index)
    where the returned value is a dictionary where the keys are the parameter
    names and the value is a Tensor containing the values associated with each
    parameter.

Gadotti
    Galaxy generator where the default parameters are set up to
    for using a copula constructed from Gadotti (2009).

MendezAbreu
    Galaxy generator where the default parameters are set up to
    for using a copula constructed from Mendez-Abreu et al. (2017).
"""
import math
import numpy
import random
import torch
from collections import Counter
from copulas.multivariate import Multivariate
from galkit.functional import pogson_mag2flux, pogson_flux2mag
from galkit.spatial import grid
from typing  import Dict, Iterable, Tuple, Optional, Union
from .convolution import DoubleGaussianPowerlawConvolution
from .copula import Copula
from .geometry import Geometry, GeometrySampler
from .photometric import Photometric, IsoFlux
from .profile import ExponentialBreak, Ferrer, Profile, Sersic
from .procedural import Perturbation, Disk, Dust, HII
from .signal import S2N
from .spiral import Logarithmic, Spiral
from .utils import access
from ..sky_detector import SkyDetectorGenerator
from ..random import random_uniform
from ..utils import BaseGenerator, load_local_generator

class Dataset(BaseGenerator, Copula, Geometry, Perturbation, Photometric, Profile, Spiral, S2N):
    """
    Base class for generating synthetic galaxies. The following method should be
    implemented for child classes:

        get_component_parameters(component, filter_band, index)

    where the returned value is a dictionary where the keys are the parameter
    names and the value is a Tensor containing the values associated with each
    parameter.

    Methods
    -------
    __call__
        Generates the flux, masks, and S/N maps.

    sample
        Calls the sample method as part of the Copula class as well as
        geometry_sampler, the latter of which are inserted into the dataframe.
    """
    def __init__(self,
        generator         : Multivariate,
        columns           : Dict[str, Union[str,Dict]],
        profile           : Union[callable, Dict],
        flux_to_magnitude : callable = pogson_flux2mag,
        magnitude_to_flux : callable = pogson_mag2flux,
        device            : Optional[torch.device] = None,
        geometry_sampler  : callable = GeometrySampler(),
        perturbation      : Optional[Dict[str,callable]] = {'disk_arm': (Disk(), HII(), Dust())},
        psf_model         : callable = DoubleGaussianPowerlawConvolution(),
        spiral            : callable = Logarithmic(),
        spiral_fraction   : Union[callable, float] = lambda : random.uniform(0.2, 0.8),
    ):
        """
        Parameters
        ----------
        generator : Multivariate
            A Multivariate copula generator.

        columns : Dict[str, Union[str,Dict]]
            A dictionary where the keys are the parameters and the values
            are the parameter names in the copula or a dictionary where the
            string is the component and the value the parameter name.

        profile : callable, dict
            Either a class Profile object or a dictionary of such objects where
            the key is the component.

        flux_to_magnitude : callable
            A function that converts the flux in a given filter band
            to a magnitude value.

        magnitude_to_flux : callable
            A function that converts the magnitude in a given filter band
            to a flux value.

        device : torch.device
            The device to generate the data on.

        geometry_sampler : callable
            A method that generates random parameters for the different components.

        perturbation : Dict[str,callable], optional
            The set of perturbations. The keys should be the components to apply
            the perturbation to, separated by an underscore "_", while the values
            should be an iterable containing the different procedural noise methods.

        psf_model : callable
            The model PSF profile for performing convolutions.

        spiral : callable
            The model spiral pattern.

        spiral_fraction : callable, float
            The fractional amount of the disk flux that the spiral arms occupy.
            Can also be a callable function.
        """
        self.__dict__.update(**locals())

    def sample(self, *args, **kwargs):
        """
        Calls the sample method as part of the Copula class as well as
        geometry_sampler, the latter of which are inserted into the dataframe.

        Parameters
        ----------
        *args, **kwargs
            Arguments to the Copula.sample method.
        """
        super().sample(*args, **kwargs)
        components = self.get_components()
        self.projection = self.geometry_sampler(
            len(self.data),
            include_bar = 'bar' in components,
            include_bulge = 'bulge' in components,
            include_disk = 'disk' in components,
        )
        for k,v in self.projection.items():
            self.data[k] = v.cpu().tolist()

    def __call__(self,
        shape              : Tuple[int,int],
        sky_detector       : SkyDetectorGenerator,
        sky_detector_index : Optional[int] = None,
        size               : Optional[int] = None,
        apply_noise        : bool = True,
        apply_pertrubation : bool = True,
        filter_bands       : Iterable = 'ugriz',
        isoA_band          : str = 'r',
        isoA_metric        : callable = IsoFlux(),
        isoA_scale         : Union[callable, float] = 2.0,
        isoA_value         : Optional[Union[callable, float]] = None,
        output_arm_mask    : bool = False,
        output_arm_s2n     : bool = False,
        output_bar_mask    : bool = False,
        output_bar_s2n     : bool = False,
        output_counts      : bool = True,
        output_galaxy_mask : bool = False,
        output_galaxy_s2n  : bool = False,
        output_projection  : bool = False,
        oversample         : int = 1,
        plate_scale        : float = 0.396,
        s2n_mask_threshold : float = 1,
        s2n_operation      : Optional[callable] = torch.mean,
    ) -> dict:
        """
        Generates the flux, masks, and S/N maps.
        
        Parameters
        ----------
        shape : Tuple[int,int]
            The image shape.

        sky_detector : BaseGenerator
            A sky-detector module that contains information pertaining
            to the flux-to-count conversion rate.

        sky_detector_index : int, optional
            The index in sky_detector to use for extracting the flux to
            count conversion rates. If flux is a tuple, then a value of
            `None` will result in matching between the flux index and
            the sky_detector index. Useful to set it generating background
            galaxies to ensure consistent beam smearing.

        size : int
            The number of samples to generate. If None, then the current
            copula is used.

        apply_noise : bool
            Boolean indicating whether to apply shot noise to the flux counts.

        apply_perturbation : bool
            Boolean indicating whether to apply the perturbation.

        filter_bands : Iterable
            The filter bands to generate data for.

        isoA_band : str
            The band to use for calculating the semi-major axis length
            as which the flux reaches the indicated metric value.

        isoA_metric : callable
            A function that calculates the observational metric (e.g. petrosian radius).
            Should take as input the radial values, the profile function with only a
            radial dependence, the metric value, and the filter band.

        isoA_scale
            A scaling factor to use for setting the field of view. The radius
            at which the isoA_value occurs is multiplied by this value.

        isoA_value
            The value at which to equate the metric. If None, then the quantity is
            set as the sky background level. Note that the metric will need to be the
            IsoFlux for this to be valid.

        output_arm_mask : bool
            Boolean indicating whether to output a mask of each of the
            spiral arms.

        output_arm_s2n : bool
            Boolean indicating whether to output the S/N ratio of each of
            the spiral arms. 

        output_bar_mask : bool
            Boolean indicating whether to output a mask of the bar. The S/N
            threshold is set py the parameter `s2n_mask_threshold`.

        output_bar_s2n : bool
            Boolean indicating whether to output the S/N ratio of the bar.
            The threshold parameter is used for creating the mask.

        output_galaxy_mask : bool
            Boolean indicating whether to output a mask of the galaxy. The S/N
            threshold is set py the parameter `s2n_mask_threshold`.

        output_galaxy_s2n : bool
            Boolean indicating whether to output the S/N ratio of the galaxy.

        oversample : int
            The oversampling factor to use when generating the flux.
            The final image will be downsampled to the indicated shape.

        plate_scale : float
            The plate_scale scale of the image in arcseconds / pixel

        s2n_mask_threshold
            Minimum S/N value to use for constructing the galaxy and bar masks.

        s2n_operation : callable, optional
            A method that reduces the S/N ratio across multiple filters to
            a single value. If set to 'None', then the S/N ratio for each
            filter is retained.

        Returns
        -------
        output : dict
            A dictionary containing the ouptut parameters. The galaxy flux is
            stored under the key 'flux' while the value consists of a dictionary
            where the keys are the filter bands and the values are the fluxes.
                The S/N values are stored with the keys {component}_s2n and the
            masks as with the keys {component}_mask.

        Examples
        --------
        import matplotlib.pyplot as plt
        import torch
        import seaborn as sns
        from galkit.functional import fits2jpeg, to_tricolor
        from galsyn.galaxy.dataset import MendezAbreu as Galaxy
        from galsyn.sky_detector import SkyDetectorGenerator
        from galsyn.galaxy.photometric import IsoFlux

        filter_bands = 'irg'
        size = 5
        shape = (256,256)

        sky = SkyDetectorGenerator()
        galaxy = Galaxy()

        sky.sample(size)
        noise = sky(shape=shape, filter_bands=filter_bands)
        output = galaxy(
            shape=shape,
            filter_bands=filter_bands,
            size=size,
            sky_detector=sky,
            isoA_metric=IsoFlux(),
            isoA_value = sky.noise_level(0.396, 'r', output_counts=False),
            isoA_scale=1,
            output_arm_mask = True,
            output_arm_s2n = True,
            output_galaxy_mask = True,
            output_galaxy_s2n = True,
            output_projection = True,
            oversample = 2,
        )

        def foo(i):
            image = torch.cat([output['flux'][i][k] + noise[k][i] for k in filter_bands])
            image = fits2jpeg(image).permute(1,2,0)

            mask_arm = output['arm_mask'][i]
            if mask_arm.nelement() == 0:
                mask_arm = torch.zeros(256,256)
            else:
                mask_arm = to_tricolor(output['arm_mask'][i], sns.color_palette('tab10'))
                mask_arm = mask_arm / mask_arm.max()

            galaxy_s2n = output['galaxy_s2n'][i]

            fig, ax = plt.subplots(ncols=3)
            ax[0].imshow(image)
            ax[1].imshow(mask_arm.squeeze())
            ax[2].imshow(galaxy_s2n.squeeze())
            fig.show()

        for i in range(size):
            foo(i)
        """
        output = {}
        shape = [x * oversample for x in shape]
        component_output = output_arm_s2n | output_bar_s2n | output_bar_mask

        if size is not None:
            self.sample(size)
        self.convert_magnitude_to_flux(plate_scale=plate_scale)

        # Generate the scaling values.
        if callable(isoA_scale):
            isoA_scale = isoA_scale(len(self), self.device)
        if callable(isoA_value):
            isoA_value = isoA_value(len(self), self.device)
        if isoA_value is None:
            isoA_value = sky_detector.noise_level(plate_scale, isoA_band, index=sky_detector_index, output_counts=False)

        isoA = self.get_isoA(
            value = isoA_value,
            metric = isoA_metric,
            filter_band = isoA_band,
        )

        # Generate geometry
        geometry = self.get_component_geometry(
            grid  = grid.pytorch_grid(shape, device=self.device),
            scale = isoA * isoA_scale,
        )

        # Generate flux
        kwargs = {'sky_detector': sky_detector, 'index': sky_detector_index}

        profile = self.get_component_profile(filter_bands=filter_bands)
        flux = self.get_component_flux(profile=profile, geometry=geometry, filter_bands=filter_bands)
        flux = self.convert_flux_to_counts(flux=flux, **kwargs)

        # Add the perturbation and spiral arms. The perturbation is done first to avoid
        # extra computations on each of the spiral arms.
        self.spiral.sample(self, isoA)
        if apply_pertrubation:
            flux = self.add_component_perturbation(flux, geometry, scale=isoA, rotation=self.spiral.rotation())
        flux, mask_arm = self.add_spiral_flux(flux, geometry, self.spiral, self.spiral_fraction)

        if output_arm_mask:
            output['arm_mask'] = mask_arm

        # Add the fluxes together if we don't need the component fluxes
        if not component_output:
            flux = self.add_component_flux(flux)

        # Remove the oversampling
        if oversample > 1:
            flux = self.downscale(input = flux, factor = oversample)
            if output != {}:
                output = self.downscale(output, factor=oversample)

        # Convolve with the PSF
        flux = self.convolve(input=flux, plate_scale=plate_scale, **kwargs)

        # Generate S/N maps or masks
        if component_output:
            galaxy_flux = self.add_component_flux(flux)
            noise_level = self.get_noise_level(flux=galaxy_flux, plate_scale=plate_scale, **kwargs)

            kwargs.update({
                'flux': flux,
                'noise_level': noise_level,
                'operation': s2n_operation,
            })

            if output_arm_s2n:
                output['arm_s2n'] = self.get_s2n(component = 'arm', **kwargs)

            if output_bar_s2n | output_bar_mask:
                bar_s2n = self.get_s2n(component = 'bar', **kwargs)

                if output_bar_s2n:
                    output['bar_s2n'] = bar_s2n
                if output_bar_mask:
                    output['bar_mask'] = tuple((s >= s2n_mask_threshold).float() for s in bar_s2n)

            flux = kwargs['flux'] = galaxy_flux

        # Calculate the S/N of the galaxy
        if output_galaxy_s2n | output_galaxy_mask:
            if not component_output:
                kwargs.update({
                    'flux': flux,
                    'noise_level' : self.get_noise_level(flux, plate_scale=plate_scale, **kwargs),
                    'operation': s2n_operation,
                })

            galaxy_s2n = self.get_s2n(**kwargs)
            if output_galaxy_s2n:
                output['galaxy_s2n'] = galaxy_s2n
            if output_galaxy_mask:
                output['galaxy_mask'] = tuple((s >= s2n_mask_threshold).float() for s in galaxy_s2n)

        # Apply noise to the image
        if apply_noise:
            flux = self.apply_shot_noise_to_counts(flux, sky_detector=sky_detector, index=sky_detector_index)

        if not output_counts:
            flux = self.convert_counts_to_flux(flux, **kwargs)

        output['flux'] = flux

        if output_projection:
            output['projection'] = self.projection

        return output

class Gadotti(Dataset):
    """
    Galaxy generator where the default parameters are set up to
    for using a copula constructed from Gadotti (2009).

    Methods
    -------
    get_component_parameters
        Returns the parameters associated with the given component.
    """
    def __init__(self,
        generator         = load_local_generator('gadotti_2009_bulge_disk.pkl'),
        columns           = {
            'flux'      : 'flux',
            'magnitude' : 'μ',
            'index'     : 'n',
            'scale'     : 'r',
            'truncate'  : 'L',
            'h0'        : 'h0',
            'w0'        : 'w0',
            'pa'        : 'pa',
            'q'         : 'q',
            'p'         : 'p',
        },
        profile = Sersic,
        **kwargs
    ):
        super().__init__(generator=generator, columns=columns, profile=profile, **kwargs)

    def get_component_parameters(self, 
        component   : str, 
        filter_band : str,
        index       : Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Returns the parameters associated with the given component.

        Parameters
        ----------
        components : str
            The name of the parameter. Can be one of {'bar', 'bulge', 'disk'} depending
            on the generator file.

        filter_band : str
            The name of the filter band to retrive the parameter values for.

        index : int, optional
            The data index to use for retrieving the values. If set to `None`, then the
            parameters are returned for all the data samples.

        Raises
        ------
        ValueError
            If the component is not part of {'bar', 'bulge', 'disk'}, then this exception
            is raised.

        Returns
        -------
        parameters : Dict[str, Tensor]
            A dictionary where the keys are the parameter names and the value is a Tensor
            containing the values associated with each parameter.

        Examples
        --------
        from galsyn.galaxy.dataset import Gadotti

        galaxy = Gadotti()
        galaxy.sample(5)
        galaxy.convert_magnitude_to_flux(plate_scale=0.396)
        parameters = galaxy.get_component_parameters('disk', 'r')

        for k,v in parameters.items():
            print(f'{k} = {v.squeeze()}')
        """
        ckeys = {
            'dataframe'   : self.data, 
            'filter_band' : filter_band,
            'index'       : index,
            'device'      : self.device,
        }

        if component == 'disk':
            return {
                'amplitude': access(self.columns['flux'], component, **ckeys),
                'scale'    : access(self.columns['scale'], component, **ckeys)
           }

        elif component == 'bulge':
            return {
                'amplitude': access(self.columns['flux'], component, **ckeys),
                'index'    : access(self.columns['index'], component, **ckeys),
                'scale'    : access(self.columns['scale'], component, **ckeys),
            }

        elif component == 'bar':
            return {
                'amplitude': access(self.columns['flux'], component, **ckeys),
                'index'    : access(self.columns['index'], component, **ckeys),
                'scale'    : access(self.columns['scale'], component, **ckeys),                
                'truncate' : access(self.columns['truncate'], component, **ckeys),                
            }
        else:
            raise ValueError(f"{component} not found")

class MendezAbreu(Dataset):
    """
    Galaxy generator where the default parameters are set up to
    for using a copula constructed from Mendez-Abreu et al. (2017).

    Methods
    -------
    get_component_parameters
        Returns the parameters associated with the given component.
    """
    def __init__(self,
        generator         = load_local_generator('mendez-abreu_2017_bulge_disk.pkl'),
        columns           : Dict[str, Union[str,Dict]]= {
            'flux'        : 'flux',
            'magnitude'   : 'μ',
            'index'       : 'n',
            'scale'       : 'r',
            'scale_inner' : 'hi',
            'scale_outer' : 'ho',
            'breakpoint'  : 'rbreak',
            'h0'          : 'h0',
            'w0'          : 'w0',
            'pa'          : 'pa',
            'q'           : 'q',
            'p'           : 'p',
        },
        profile           : Union[callable, Dict] = {'bar': Ferrer, 'bulge': Sersic, 'disk': ExponentialBreak},
        **kwargs
    ):
        super().__init__(generator=generator, columns=columns, profile=profile, **kwargs)

    def get_component_parameters(self, 
        component   : str, 
        filter_band : str,
        index       : Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Returns the parameters associated with the given component.

        Parameters
        ----------
        components : str
            The name of the parameter. Can be one of {'bar', 'bulge', 'disk'} depending
            on the generator file.

        filter_band : str
            The name of the filter band to retrive the parameter values for.

        index : int, optional
            The data index to use for retrieving the values. If set to `None`, then the
            parameters are returned for all the data samples.

        Raises
        ------
        ValueError
            If the component is not part of {'bar', 'bulge', 'disk'}, then this exception
            is raised.

        Returns
        -------
        parameters : Dict[str, Tensor]
            A dictionary where the keys are the parameter names and the value is a Tensor
            containing the values associated with each parameter.

        Examples
        --------
        from galsyn.galaxy.dataset import MendezAbreu as Galaxy

        galaxy = Galaxy()
        galaxy.sample(5)
        galaxy.convert_magnitude_to_flux(plate_scale=0.396)
        parameters = galaxy.get_component_parameters('disk', 'r')

        for k,v in parameters.items():
            print(f'{k} = {v.squeeze()}')
        """
        ckeys = {
            'dataframe'   : self.data, 
            'filter_band' : filter_band,
            'index'       : index,
            'device'      : self.device,
        }

        if component == 'disk':
            return {
                'amplitude'  : access(self.columns['flux'], component, **ckeys),
                'scale_inner': access(self.columns['scale_inner'], component, **ckeys),
                'scale_outer': access(self.columns['scale_outer'], component, **ckeys, default=math.inf),
                'breakpoint' : access(self.columns['breakpoint'], component, **ckeys, default=math.inf),
            }

        elif component == 'bulge':
            return {
                'amplitude': access(self.columns['flux'], component, **ckeys),
                'index'    : access(self.columns['index'], component, **ckeys),
                'scale'    : access(self.columns['scale'], component, **ckeys),
            }

        elif component == 'bar':
            return {
                'amplitude': access(self.columns['flux'], component, **ckeys),
                'scale'    : access(self.columns['scale'], component, **ckeys),                
            }
        else:
            raise ValueError(f"{component} not found")

class BackgroundGalaxy:
    """
    Class that provides a useful interface for constructing
    background galaxies from multiple datasets.

    Examples
    --------
    import matplotlib.pyplot as plt
    import torch
    from galkit.functional import fits2jpeg
    from galsyn.galaxy import BackgroundGalaxy
    from galsyn.sky_detector import SkyDetectorGenerator

    bg = BackgroundGalaxy()
    sky = SkyDetectorGenerator()

    output = bg(
        size = 5,
        shape = (128,128),
        sky_detector = sky,
        filter_bands = 'irg',
        galaxies_per_pixel = 5e-4,
        output_galaxy_s2n = True,
        output_galaxy_mask = True,
    )

    def foo(i):
        image = torch.cat([output['flux'][i][k] for k in 'irg'], dim=0)
        image = fits2jpeg(image).permute(1,2,0)
        mask = output['mask'][i].squeeze()
        s2n = output['s2n'][i].squeeze()

        fig, ax = plt.subplots(ncols=3)
        ax[0].imshow(image)
        ax[1].imshow(s2n)
        ax[2].imshow(mask)
        fig.show()
    
    for i in range(len(sky)):
        foo(i)
    """
    def __init__(self,
        generators : Tuple[Dataset] = (
            Gadotti(load_local_generator('gadotti_2009_bar_bulge_disk.pkl')),
            Gadotti(load_local_generator('gadotti_2009_bulge_disk.pkl')),
            Gadotti(load_local_generator('gadotti_2009_disk.pkl')),
            MendezAbreu(load_local_generator('mendez-abreu_2017_bar_bulge_disk.pkl')),
            MendezAbreu(load_local_generator('mendez-abreu_2017_bar_bulge_disk_dbreak.pkl')),
            MendezAbreu(load_local_generator('mendez-abreu_2017_bar_disk.pkl')),
            MendezAbreu(load_local_generator('mendez-abreu_2017_bulge_disk.pkl')),
            MendezAbreu(load_local_generator('mendez-abreu_2017_bulge_disk_dbreak.pkl')),
            MendezAbreu(load_local_generator('mendez-abreu_2017_disk.pkl')),
            MendezAbreu(load_local_generator('mendez-abreu_2017_disk_dbreak.pkl')),
        ),
        device : Optional[torch.device] = None,
        geometry_sampler : callable = GeometrySampler(centroid_sampler = lambda size, device: random_uniform(-1, 1, (2,size), device=device)),
        perturbation : Optional[Dict[str,callable]] = None,
        psf_model : callable = DoubleGaussianPowerlawConvolution(),
    ):
        """
        Parameters
        ----------
        generators : Tuple[Dataset]
            A collection of Galaxy modules.
        
        device : torch.device
            The device to generate the data on.

        geometry_sampler : callable
            A method that generates random parameters for the different components.

        perturbation : Dict[str,callable], optional
            The set of perturbations. The keys should be the components to apply
            the perturbation to, separated by an underscore "_", while the values
            should be an iterable containing the different procedural noise methods.
        """
        for x in generators:
            x.geometry_sampler = geometry_sampler
            x.device = device
            x.perturbation = perturbation
            x.psf_model = psf_model
        self.generators = generators
        self.device = device

    def __call__(self,
        shape : Tuple[int,int],
        filter_bands : Iterable,
        sky_detector : SkyDetectorGenerator,
        apply_noise : bool = True,
        galaxies_per_pixel : Union[callable,float] = 2e-4,
        isoA_band = 'r',
        isoA_metric = IsoFlux(),
        isoA_value = None,
        isoA_scale : Union[callable,float] = lambda size, device : random_uniform(math.log(3), math.log(30), size, device).exp(),
        output_galaxy_mask : bool = False,
        output_galaxy_s2n : bool = False,
        oversample : Union[callable,int] = 1,
        plate_scale : Union[callable,float] = 0.396,
        size  : Optional[int] = None,
        s2n_mask_threshold : float = 1,
    ):
        """
        Parameters
        ----------
        shape : Tuple[int,int]
            The image shape.

        size : int
            The number of samples to generate. If None, then the current
            copula is used.

        filter_bands : Iterable
            The filter bands to generate data for.

        sky_detector : BaseGenerator
            A sky-detector module that contains information pertaining
            to the flux-to-count conversion rate.

        apply_noise : bool
            Boolean indicating whether to apply shot noise to the flux counts.

        galaxies_per_pixel : float, callable
            The average number of galaxies per pixel. The total number of galaxies
            is Poisson sampled based on the expectation value.

        isoA_band : str
            The band to use for calculating the semi-major axis length
            as which the flux reaches the indicated metric value.

        isoA_metric : callable
            A function that calculates the observational metric (e.g. petrosian radius).
            Should take as input the radial values, the profile function with only a
            radial dependence, the metric value, and the filter band.

        isoA_scale
            A scaling factor to use for setting the field of view. The radius
            at which the isoA_value occurs is multiplied by this value.

        isoA_value
            The value at which to equate the metric. If None, then the quantity is
            set as the sky background level. Note that the metric will need to be the
            IsoFlux for this to be valid.

        output_galaxy_mask : bool
            Boolean indicating whether to output a mask of the galaxies. The S/N
            threshold is set py the parameter `s2n_mask_threshold`.

        output_galaxy_s2n : bool
            Boolean indicating whether to output the S/N ratio of the galaxies.

        oversample : int
            The oversampling factor to use when generating the flux.
            The final image will be downsampled to the indicated shape.

        plate_scale : float
            The plate_scale scale of the image in arcseconds / pixel
        """
        if size is not None:
            sky_detector.sample(size)
        size = len(sky_detector)

        flux = []
        for i in range(size):
            gpp = galaxies_per_pixel() if callable(galaxies_per_pixel) else galaxies_per_pixel
            n = numpy.random.poisson(shape[0] * shape[1] * gpp)

            if n == 0:
                zero = torch.zeros((1,*shape), device=self.device)
                flux.append({k:zero for k in filter_bands})
                if output_galaxy_mask:
                    masks['mask'] = zero
                if output_galaxy_s2n:
                    masks['s2n'] = zero
                continue

            images  = {k:0. for k in filter_bands}
            masks   = {}
            samples = numpy.random.randint(len(self.generators), size=n)
            samples = Counter(samples)

            for index, size in samples.items():

                output = self.generators[index](
                    apply_noise = False,
                    filter_bands = filter_bands,
                    isoA_band = isoA_band,
                    isoA_metric = isoA_metric,
                    isoA_value = isoA_value,
                    isoA_scale = isoA_scale,
                    oversample = oversample,
                    plate_scale = plate_scale,
                    shape = shape,
                    size = size,
                    sky_detector = sky_detector,
                    sky_detector_index = i,
                    s2n_mask_threshold=s2n_mask_threshold,
                )

                for k in filter_bands:
                    v = torch.stack([x[k] for x in output['flux']]).sum(dim=0)
                    images[k] = v if isinstance(images[k],float) else (images[k] + v)

            flux.append(images)

        output = {}
        if output_galaxy_s2n | output_galaxy_mask:
            noise = self.generators[0].get_noise_level(flux, sky_detector=sky_detector, plate_scale=plate_scale)
            s2n = self.generators[0].get_s2n(flux, noise, sky_detector=sky_detector, operation=torch.mean)
            if output_galaxy_s2n:
                output['s2n'] = s2n
            if output_galaxy_mask:
                output['mask'] = tuple(s > s2n_mask_threshold for s in s2n)

        if apply_noise:
            flux = self.generators[0].apply_shot_noise_to_counts(flux, sky_detector=sky_detector)

        return {'flux': flux, **output}