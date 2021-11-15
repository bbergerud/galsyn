import random
import torch
from typing import Dict, Tuple, Union

class Spiral:
    """
    Class for adding spiral arms to the flux.

    Methods
    -------
    add_spiral_flux(flux, geometry, pattern, fraction)
        Adds spiral arms to the flux components and removes the contribution
        from the disk.

    sample_spiral(isoA)
        Interface to the sample method for the spiral object.
    """

    def add_spiral_flux(self, 
        flux     : Union[Dict, Tuple[Dict]],
        geometry : Union[Dict, Tuple[Dict]],
        pattern  : Union[Dict, Tuple[Dict]],
        fraction : Union[callable, float] = lambda : random.uniform(0.1, 0.8),
    ) -> Union[Union[Dict, Tuple[Dict]], Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """
        Adds spiral arms to the flux components and removes the contribution
        from the disk.

        Parameters
        ----------
        flux : Dict, Tuple[Dict]
            Either a dictionary where the key is the component and the value
            is another dictionary where the key is the filter band and the value
            the flux or a tuple containing such dictionaries.

        geometry : Dict, Tuple[Dict]
            Either a dictionary where the key is the component and the value
            is another dictionary where the key is the coordinate parameter
            (e.g. r, θ) and the value is a tensor containing the grid values
            or a tuple containing such dictionaries.

        pattern : callable, Tuple[callable]
            A function that takes as input the disk goemetry and returns the
            spiral arm perturbation as well as the spiral arm mask, or a tuple
            of such functions.

        fraction : callable, float
            The fractional amount of the disk flux that the spiral arms occupy.
            Can also be a callable function.

        Returns
        -------
        flux : Dict, Tuple[Dict]
            Either a dictionary where the key is the component and the value
            is another dictionary where the key is the filter band and the value
            the flux or a tuple containing such dictionaries. An entry for the
            arms has been added.

        mask : Tensor, Tuple[Tensor]
            The spiral arm mask, or a tuple of such masks.

        Examples
        --------
        import matplotlib.pyplot as plt
        import torch
        from galkit.spatial import grid
        from galkit.functional import arcsinh_stretch
        from galsyn.galaxy.dataset import MendezAbreu as Galaxy

        filter_bands = 'irg'
        size  = 5
        shape = (128,128)

        galaxy = Galaxy()
        galaxy.sample(size)
        galaxy.convert_magnitude_to_flux(plate_scale=0.396)
        isoA = galaxy.get_isoMag(value=24.5, filter_band='r', index=None)
        galaxy.sample_spiral(isoA)

        profile  = galaxy.get_component_profile(filter_bands)
        geometry = galaxy.get_component_geometry(grid.pytorch_grid(shape), scale=isoA)
        component_flux = galaxy.get_component_flux(geometry=geometry, profile=profile, filter_bands=filter_bands)
        component_flux, mask = galaxy.add_spiral_flux(flux=component_flux, geometry=geometry, pattern=galaxy.spiral.pattern())
        total_flux = galaxy.add_component_flux(component_flux)

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
            output = tuple(self.add_spiral_flux(
                flux = f,
                geometry = g,
                pattern = p,
                fraction = fraction,
            ) for (f,g,p) in zip(flux, geometry, pattern))
            return tuple(tuple(o[i] for o in output) for i in range(len(output[0])))

        pattern, mask = pattern(**geometry['disk'])
        f = fraction() if callable(fraction) else fraction

        flux['arm'] = {}
        for k,v in flux['disk'].items():
            flux['arm'][k] = f * pattern * v
            flux['disk'][k] = (1 - f) * v
        geometry['arm'] = geometry['disk']
        return flux, mask

    def sample_spiral(self,
        isoA : torch.Tensor,
        **kwargs
    ):
        """
        Interface to the sample method for the spiral object.
        
        Parameters
        ----------
        isoA : Tensor
            The scaling factor used for constructing the geometry.
        """
        self.spiral.sample(
            cls = self,
            isoA = isoA,
            **kwargs
        )