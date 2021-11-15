"""
Methods for modulating the spiral arms.

Classes
-------
Damping
    Applies exponential damping at distances interior to r_min
    and exterior to r_max through the transformations
        exp(-(r_min - r) / σ_min)       [r < r_min]
        exp(-(r - r_max) / σ_max)       [r > r_max]
"""
import torch
from dataclasses import dataclass
from typing import Optional
from ..utils import access
from ...random import random_uniform

@dataclass
class Damping:
    """
    Applies exponential damping at distances interior to r_min
    and exterior to r_max through the transformations

        exp(-(r_min - r) / σ_min)       [r < r_min]
        exp(-(r - r_max) / σ_max)       [r > r_max]

    Parameters
    ----------
    rmin_sampler : callable, optional
        A function that takes as input the number of arms and
        the device and returns the r_min values relative to
        the base value. If None, then damping is not applied
        for r < r_min.

    rmin_sampler : callable, optional
        A function that takes as input the number of arms and
        the device and returns the r_max values relative to
        the base value. If None, then damping is not applied
        for r > r_max. 

    σmin_sampler : callable, optional
        A function that takes as input the number of arms and
        the device and returns the σ_min values relative to
        the base r_min value. If None, then the transition is
        abrupt.

    σmax_sampler : callable, optional
        A function that takes as input the number of arms and
        the device and returns the σ_max values relative to
        the base r_max value. If None, then the transition is
        abrupt.
    """
    rmin_sampler : Optional[callable] = lambda size, device: random_uniform(0.75,1.25,size,device)
    rmax_sampler : Optional[callable] = lambda size, device: random_uniform(0.50,1.50,size,device)
    σmin_sampler : Optional[callable] = lambda size, device: random_uniform(0.05,0.10,size,device)
    σmax_sampler : Optional[callable] = lambda size, device: random_uniform(0.10,0.20,size,device)

    def __call__(self,
        r:torch.Tensor, 
        index:int, 
        arm_index:int, 
        **kwargs
    ) -> torch.Tensor:
        """
        Returns a tensor containing the modulation factors.

        Parameters
        ----------
        r : Tensor
            The radial distances.
        
        index : int
            The galaxy number.
        
        arm_index : int
            The arm number.

        Returns
        -------
        output : Tensor
            A tensor where the values betweeen r_min < r < r_max are equal
            to 1 and the values outside are equal to the damping factor.
        """
        params = self[index]
        params = {k:v if v.nelement() == 1 else v[arm_index] for k,v in params.items()}

        f = torch.ones_like(r)

        if params['r_max'] is not None:
            mask = r > params['r_max']
            if params['σ_max'] is None:
                f[mask] = 0.
            else:
                z = r[mask] - params['r_max']
                f[mask] = torch.exp(-z / params['σ_max'])

        if params['r_min'] is not None:
            mask = r < params['r_min']
            if params['σ_min'] is None:
                f[mask] = 0.
            else:
                z = params['r_min'] - r[mask]
                f[mask] = torch.exp(-z / params['σ_min'])

        return f

    def __getitem__(self, index:int) -> dict:
        """
        Returns a dictionary containing the parameter values for
        the indicated galaxy.
        """
        return {k:v[index] for k,v in self.params.items()}

    def __len__(self):
        """
        Returns the number of galaxies.
        """
        return len(self.params['r_min'])

    def sample(self, 
        cls,
        isoA : torch.Tensor,
        arm_count : torch.Tensor,
        filter_band : str = 'r'
    ):
        """
        Generates the r_min, r_max, σ_min, and σ_max parameters based on
        whether a bar or bulge are present.

        If a bar is present, then r_min is set to the truncate value if it
        exist otherwise to the scale factor. 

        It a bulge is present, then r_min is set to the scale value.

        If no bar or bulge is present, then the r_min value is set to a
        random value between 0 and 0.05 times the isoA value.

        The r_max value is set to isoA, with the condition that the r_min and
        r_max values are sorted to ensure r_min <= r_max

        Parameters
        ----------
        cls : class
            The galaxy class object.

        isoA : tensor
            The scaling factors used to rescale the base grid when generating
            the geometry.

        arm_count : Tensor
            A tensor containing the number of arms in each galaxy.

        filter_band : str
            The filter band to use when extracting the bar or bulge parameters.
        """
        kwargs = {
            'dataframe'   : cls.data,
            'filter_band' : filter_band,
            'device'      : cls.device,
        }

        if 'bar' in cls.get_components():
            col = 'truncate' if 'truncate' in cls.columns else 'scale'
            r_min = access(cls.columns[col], 'bar', **kwargs)
        elif 'bulge' in cls.get_components():
            r_min = access(cls.columns['scale'], 'bulge', **kwargs)
        else:
            r_min = random_uniform(0, 0.05, len(cls), cls.device) * isoA

        r_min, r_max = torch.sort(torch.cat([r_min.view(1,-1), isoA.view(1,-1)]), 0).values

        self.params = {
            'r_min': None if self.rmin_sampler is None else tuple(r_min[i] * self.rmin_sampler(arm_count[i], cls.device) for i in range(len(cls))),
            'r_max': None if self.rmax_sampler is None else tuple(r_max[i] * self.rmax_sampler(arm_count[i], cls.device) for i in range(len(cls))),
            'σ_min': None if self.σmin_sampler is None else tuple(r_min[i] * self.σmin_sampler(arm_count[i], cls.device) for i in range(len(cls))),
            'σ_max': None if self.σmax_sampler is None else tuple(r_max[i] * self.σmax_sampler(arm_count[i], cls.device) for i in range(len(cls))),
        }