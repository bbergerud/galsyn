"""
Module for generating coordinate information for the various galactic components.

Classes
-------
Geometry
    Class for generating random geometric coordinates for the different components.
    Only disk, bulge, and bar are currently implemented. Assumes the presence of a
    disk component.

GeometrySampler
    Class for generating random geometry parameters for galaxies.
"""
import math
import torch
from galkit.functional import angular_distance
from galkit.spatial import coordinate
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
from .utils import access
from ..random import random_normal, random_uniform

class Geometry:
    """
    Class for generating random geometric coordinates for the different components.
    Only disk, bulge, and bar are currently implemented. Assumes the presence of a
    disk component.

    Methods
    -------
    get_component_geometry(grid, scale, index, filter_band, **kwargs)
        Returns the coordinate information for each of the components. 

    Required Attributes
    -------------------
    data : DataFrame
        A pandas dataframe that contains the parameter data.

    device : torch.device
        The device on which to generate tensor data.

    column : Dict[str, Union[str,dict]]
        A dictionary containing the following key, value pairs:

        h0 : dict, str
            The name of the column associated with the vertical (height)
            centroid component. Can also be a dictionary of such values
            where the key is the component (e.g., disk, bar).
        
        w0 : dict, str
            The name of the column associated with the horizontal (width)
            centroid component. Can also be a dictionary of such values
            where the key is the component (e.g., disk, bar).
        
        pa : dict, str
            The name of the column associated with the position angle.
            Can also be a dictionary of such values where the key is
            the component (e.g., disk, bar)

        q : dict, str
            The name of the column associated with the flattening.
            Can also be a dictionary of such values where the key is
            the component (e.g., disk, bar).
        
        p : dict, str
            The name of the column associated with ellipsoidal p value.
            Can also be a dictionary of such values where the key is
            the component (e.g., disk, bar).
    """

    def get_component_geometry(self,
        grid        : Tuple[torch.Tensor, torch.Tensor],
        scale       : torch.Tensor,
        index       : Optional[int] = None,
        filter_band : str = 'r',
        **kwargs
    ) -> Union[Dict, Tuple[Dict]]:
        """
        Returns the coordinate information for each of the components. The scaling
        factor for the bar is multiplied by an additional factor

            scale *= cos??(??pa) + sin??(??pa)/q??

        where ??pa is the angular distance between the position angle of the disk
        and the bar and q is the flattening parameter of the disk.

        Parameters
        ----------
        grid : Tuple[Tensor, Tensor]
            The base grid coordinate system.

        scale : Tensor
            The scaling factor for resizing the base grid.

        index : int, optional
            The index in `self.data` for which to use to generate the geometry profile.
            If None, then a profile is generated for each value in the data frame.
        
        filter_band : str
            The filter band to use for accessing any potential parameters related to the
            bar or bulge.

        **kwargs
            Additional parameter to pass into the galkit.spatial.coordinate.polar coordinate
            method.

        Returns
        -------
        geometry : Dict, Tuple[Dict]
            Either a dictionary where the key is the component and the value
            is another dictionary where the key is the coordinate parameter
            (e.g. r, ??) and the value is a tensor containing the grid values
            or a tuple containing such dictionaries.

        Examples
        --------
        import matplotlib.pyplot as plt
        from galkit.spatial import grid
        from galsyn.galaxy.dataset import Gadotti as Galaxy

        galaxy = Galaxy()
        galaxy.sample(5)
        galaxy.convert_magnitude_to_flux(plate_scale=0.396)
        isoA = galaxy.get_isoMag(value=24.5, filter_band='r', index=None)

        geometry = galaxy.get_component_geometry(
            grid  = grid.pytorch_grid(128, 128),
            scale = isoA,
            index = None,
        )

        def foo(i):
            data = geometry[i]
            
            n = len(data)
            fig, ax = plt.subplots(ncols=n)
            for i,(k,v) in enumerate(data.items()):
                ax[i].contour(v['r'].add(0.01).log().squeeze())
                ax[i].set_title(k)
            fig.show()

        for i in range(5):
            foo(i)
        """
        if index is None:
            return tuple(self.get_component_geometry(grid, scale=s, index=i, filter_band=filter_band, **kwargs) for i,s in enumerate(scale))

        components = self.get_components()

        keys = {
            'component'   : 'disk',
            'dataframe'   : self.data,
            'dictionary'  : {},
            'index'       : index,
            'filter_band' : filter_band,
            'device'      : self.device,
        }

        ??_disk, r_disk = coordinate.polar(
            grid = grid,
            h0 = access(self.columns['h0'], **keys),
            w0 = access(self.columns['w0'], **keys),
            pa = access(self.columns['pa'], **keys),
            q  = access(self.columns['q'], **keys),
            p  = access(self.columns['p'], **keys, default=2),
            scale = scale,
            **kwargs
        )

        geometry = {'disk': {'??': ??_disk, 'r': r_disk}}


        if 'bar' in components:
            pa_disk = access(self.columns['pa'], **keys)
            q_disk  = access(self.columns['q'], **keys)

            keys['component'] = 'bar'
            pa_bar = access(self.columns['pa'], **keys)
            ??pa = angular_distance(pa_bar, pa_disk)

            geometry['bar'] = {
                'r': coordinate.polar(
                    grid = grid,
                    h0 = access(self.columns['h0'], **keys),
                    w0 = access(self.columns['w0'], **keys),
                    pa = access(self.columns['pa'], **keys),
                    q  = access(self.columns['q'], **keys),
                    p  = access(self.columns['p'], **keys, default=2),
                    scale = scale * torch.sqrt(??pa.cos().pow(2) + ??pa.sin().div(q_disk).pow(2)),
                    **kwargs
                )[-1]
            }


        if 'bulge' in components:

            keys['component'] = 'bulge'
            geometry['bulge'] = {
                'r' : coordinate.polar(
                    grid = grid,
                    h0 = access(self.columns['h0'], **keys),
                    w0 = access(self.columns['w0'], **keys),
                    pa = access(self.columns['pa'], **keys),
                    q  = access(self.columns['q'], **keys),
                    p  = access(self.columns['p'], **keys, default=2),
                    scale = scale,
                    **kwargs
                )[-1]
            }

        return geometry

@dataclass
class GeometrySampler:
    """
    Class for generating random geometry parameters for galaxies.

    Parameters
    ----------
    centroid_sampler : callable
        Takes as input the size and device and returns the (h0,w0) centroids.

    pa_bar_sampler : callable
        Takes as input the size and device and returns the pa value of the bar.

    pa_bulge_sampler : callable
        Takes as input the size and device and returns the pa value of the bulge.

    pa_disk_sampler : callable
        Takes as input the size and device and returns the pa value of the disk.

    q_bulge_sampler : callable
        Takes as input the size and device and returns the q value of the bulge.

    q_disk_sampler : callable
        Takes as input the size and device and returns the q value of the disk.

    Methods
    -------
    __call__(size, device)
        Returns a dictionary containing the coordinate parameters. The keys are
        in the format `{parameter}_{component}`.

    Examples
    --------
    from galsyn.galaxy.geometry import GeometrySampler

    geometry = GeometrySampler()
    print(geometry(5))
    """
    centroid_sampler : callable = lambda size, device : random_normal(0, 0.025, (2, size), device)
    pa_bar_sampler   : callable = lambda size, device : random_uniform(0, math.pi, size, device)
    pa_bulge_sampler : callable = lambda size, device : torch.zeros(size, device=device)
    pa_disk_sampler  : callable = lambda size, device : random_uniform(0, math.pi, size, device)
    q_bulge_sampler  : callable = lambda size, device : torch.ones(size, device=device)
    q_disk_sampler   : callable = lambda size, device : random_uniform(0.2, 1.0, size, device)

    def __call__(self, 
        size   : int,
        device : Optional[torch.device] = None,
        include_bar : bool = True,
        include_bulge : bool = True,
        include_disk : bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary containing the projection parameters
        for the different components.

        Parameters
        ----------
        size : int
            The number of parameter sets to generate.
        
        device : torch.device, optional
            The device to generate the data on.

        include_bar : bool
            Boolean indicating whether to generate a position angle for the bar.
            The q flattening parameter is assumed to be generated by the copula.

        include_bulge : bool
            Boolean indicating whether to generate a position angle and flattening
            parameter for the bulge.

        include_disk : bool
            Boolean indicating whether to generate a position angle and flattening
            parameter for the disk.       
        """
        h0, w0 = self.centroid_sampler(size, device)
        output = {}

        if include_bar:
            output.update({
                'h0_bar': h0,
                'w0_bar': w0,
                'pa_bar': self.pa_bar_sampler(size, device)
            })

        if include_bulge:
            output.update({ 
                'h0_bulge': h0,
                'w0_bulge': w0,
                'pa_bulge': self.pa_bulge_sampler(size, device),
                'q_bulge' : self.q_bulge_sampler(size, device),
            })

        if include_disk:
            output.update({
                'h0_disk': h0,
                'w0_disk': w0,
                'pa_disk': self.pa_disk_sampler(size, device),
                'q_disk' : self.q_disk_sampler(size, device)
            })

        return output