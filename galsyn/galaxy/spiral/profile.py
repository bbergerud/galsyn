"""
Profiles for the spiral arm flux.

Classes
-------
ShockProfile
    Implements a rough approximation to the density profile as the result
    of a shock through the functional form
        p(θ, θ_sp) ∝ {
            (1 - Δθ/π)^(p_shock)   pre-shock
            (1 - Δθ/π)^(p_arm)     post-shock
        }
    where Δθ is the angular distance between θ and θ_spiral.

ExponentialProfile
    Generates a profile through the functional form
        p(θ, θ_sp) ∝ exp(-(Δθ/width)^power)
"""

import math
import torch
from dataclasses import dataclass
from galkit.functional import angular_distance, sigmoid
from typing import Optional, Tuple, Union
from .modulate import Damping
from ...random import random_uniform

@dataclass
class ShockProfile:
    """
    Implements a rough approximation to the density profile as the result
    of a shock through the functional form

        p(θ, θ_sp) ∝ {
            (1 - Δθ/π)^(p_shock)   pre-shock
            (1 - Δθ/π)^(p_arm)     post-shock
        }

    where Δθ is the angular distance between θ and θ_spiral.

    Parameters
    ----------
    corotation : callable
        A function that takes as input the number of arms and the device
        and returns the corotation radius.

    transition : callable
        A function that takes as input the number of arms and the device
        and returns the fraction of the corotation about which the power
        index swaps from the pre- to post-shock regime through the sigmoid
        function.
    
    p_shock : callable
        A function that takes as input the number of arms and the device
        and returns the power index associated with the pre-shock regime.

    p_arm : callable
        A function that takes as input the number of arms and the device
        and returns the power index associated with the post-shock regime.

    modulate : callable, optional
        A class that applies damping to the spiral arm pattern. Useful
        for preventing the spiral arms from propagating into the center
        of the galaxy and extending well beyond the observable portion
        of the galaxy.

    Methods
    -------
    __getitem__(index)
        Returns a dictionary containing the set of parameters associated
        with the given index.

    To Do
    -----
    Could implement offsets such that an arm could potentially appear disconnected
    at the corotation radius.
    """
    corotation : callable = lambda size, device : random_uniform(0.1, 1, 1, device)
    transition : callable = lambda size, device : random_uniform(0.05, 0.15, 1, device)
    p_shock    : callable = lambda size, device : random_uniform(10, 15, size, device)
    p_arm      : callable = lambda size, device : random_uniform(5, 10, size, device)
    modulate   : Optional[callable] = Damping()   

    def __call__(self,
        r : torch.Tensor,
        θ : torch.Tensor,
        θ_spiral : torch.Tensor,
        sign  : torch.Tensor,
        index : int,
        arm_index : int,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the mask and a normalized profile that has a mean
        value of 1.

        Parameters
        ----------
        r : Tensor
            The radial distances.

        θ : Tensor
            The azimuthal positions.

        θ_spiral : Tensor
            The azimuthal position of the spiral arm at the
            various radial distances.

        sign : Tensor
            The sign of the rotation.

        index : int
            The galaxy index. Used to extract the appropriate values.
        
        arm_index : int
            The arm index. Used to extract the appropriate values.

        **kwargs
            Used for compatibility with other methods.

        Returns
        -------
        profile : Tensor
            The perturbation tensor for the profile

        mask : Tensor
            The spiral arm masks
        """
        params = self[index]

        # Calculate the angular offsets
        dx = angular_distance(θ, θ_spiral, absolute=False) / math.pi
        if sign < 0:
            dx = -dx

        # Generate the transition across the corotation
        loc = params['corotation'] * params['isoA']
        loc = loc if loc.nelement() == 1 else loc[arm_index]
        scale = params['transition']
        scale = scale if scale.nelement() == 1 else scale[arm_index]
        t = sigmoid(r, loc=loc, scale=scale)
  
        # Generate the power index
        p_shock = params['p_shock']
        p_shock = p_shock if p_shock.nelement() == 1 else p_shock[arm_index]
        p_arm = params['p_arm']
        p_arm = p_arm if p_arm.nelement() == 1 else p_arm[arm_index]

        s = (p_shock - p_arm) * t
        P1 = p_arm + s
        P2 = p_shock - s
        power = torch.empty_like(P2)

        # Interior
        mask = dx < 0
        power[mask] = P2[mask]

        # Exterior
        mask = ~mask
        power[mask] = P1[mask]

        mask = (1 - dx.abs()).pow(power)

        if self.modulate is not None:
            modulate = self.modulate(r=r, θ=θ, index=index, arm_index=arm_index)
            mask = mask * modulate

        profile = mask * self.norm(P1, P2)
        return profile, mask

    def __getitem__(self, index:int) -> dict:
        """
        Returns a dictionary containing the set of parameters associated
        with the given index.
        """
        return {k:v[index] for k,v in self.params.items()}   

    def sample(self, cls, isoA:torch.Tensor, arm_count:int, **kwargs):
        """
        Generates parameter values for constructing the profile.

        Parameters
        ----------
        cls : class
            The galaxy class object.

        isoA : tensor
            The scaling factors used to rescale the base grid when generating
            the geometry. The Wa values are multiplied by this.

        arm_count : int
            The number of arms for each galaxy.

        **kwargs
            Additional arguments to pass into the sample method for the modulate
            object.
        """
        self.params = {
            'isoA'      : isoA,
            'corotation': tuple(self.corotation(size=n, device=cls.device) for n in arm_count),
            'transition': tuple(self.transition(size=n, device=cls.device) for n in arm_count),
            'p_shock'   : tuple(self.p_shock(size=n, device=cls.device) for n in arm_count),
            'p_arm'     : tuple(self.p_arm(size=n, device=cls.device) for n in arm_count),
        }
        if self.modulate is not None:
            self.modulate.sample(cls=cls, isoA=isoA, arm_count=arm_count, **kwargs)


    def norm(self,
        p1:Union[float, torch.Tensor], 
        p2:Union[float, torch.Tensor]
    ) -> Union[float, torch.Tensor]:
        """
        Normalization factor such that the mean value of the profile is 1.
        """
        t1 = 1 / (p1 + 1)
        t2 = 1 / (p2 + 1)
        return 2 / (t1 + t2) 

@dataclass
class ExponentialProfile:
    """
    Generates a profile through the functional form

        p(θ, θ_sp) ∝ exp(-(Δθ/width)^power)

    Parameters
    ----------
    power : callable
        A function that takes as input the number of arms and the device
        and returns the power index associated with the arm.

    width : callable
        A function that takes as input the number of arms and the device
        and returns the width associated with the arm.

    modulate : callable, optional
        A class that applies damping to the spiral arm pattern. Useful
        for preventing the spiral arms from propagating into the center
        of the galaxy and extending well beyond the observable portion
        of the galaxy.
    """
    power    : callable = lambda size, device : random_uniform(1, 3, size, device)
    width    : callable = lambda size, device : random_uniform(0.1, 0.4, size, device)
    modulate : Optional[callable] = Damping()

    def __call__(self,
        r : torch.Tensor,
        θ : torch.Tensor,
        θ_spiral : torch.Tensor,
        index : int,
        arm_index : int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the mask and a normalized profile that has a mean
        value of 1.

        Parameters
        ----------
        r : Tensor
            The radial distances.

        θ : Tensor
            The azimuthal positions.

        θ_spiral : Tensor
            The azimuthal position of the spiral arm at the
            various radial distances.

        index : int
            The galaxy index. Used to extract the appropriate values.
        
        arm_index : int
            The arm index. Used to extract the appropriate values.

        **kwargs
            Used for compatibility with other methods.

        Returns
        -------
        profile : Tensor
            The perturbation tensor for the profile

        mask : Tensor
            The spiral arm masks
        """
        params = self[index]
        width = params['width'] if params['width'].nelement() == 1 else params['width'][arm_index]
        power = params['power'] if params['power'].nelement() == 1 else params['power'][arm_index]

        dx = angular_distance(θ, θ_spiral)

        mask = torch.exp(-(dx/width).pow(power))
        if self.modulate is not None:
            modulate = self.modulate(r=r, θ=θ, index=index, arm_index=arm_index)
            mask = mask * modulate

        profile = mask * self.norm(width=width, power=power)

        return profile, mask

    def __getitem__(self, index:int) -> dict:
        """
        Returns a dictionary containing the set of parameters associated
        with the given index.
        """
        return {k:v[index] for k,v in self.params.items()}

    def norm(self, power:torch.Tensor, width:torch.Tensor) -> torch.Tensor:
        """
        Normalization factor such that the mean value is 1.
        """
        t = torch.lgamma(1/power).exp() * (1 - torch.igammac(1/power, (math.pi / width)**power))
        return (math.pi * power) / (width * t)

    def sample(self, cls, isoA:torch.Tensor, arm_count:int, **kwargs):
        """
        Generates parameter values for constructing the profile.

        Parameters
        ----------
        cls : class
            The galaxy class object.

        isoA : tensor
            The scaling factors used to rescale the base grid when generating
            the geometry. The Wa values are multiplied by this.

        arm_count : int
            The number of arms for each galaxy.

        **kwargs
            Additional arguments to pass into the sample method for the modulate
            object.
        """
        self.params = {
            'width': tuple(self.width(n, cls.device) for n in arm_count),
            'power': tuple(self.power(n, cls.device) for n in arm_count),
        }
        if self.modulate is not None:
            self.modulate.sample(cls=cls, isoA=isoA, arm_count=arm_count, **kwargs)