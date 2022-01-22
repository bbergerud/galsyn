"""
Methods for modeling spiral arms.

Classes
-------
SpiralPattern
    Parent class for constructing spiral patterns. Child classes
    should store the parameter variables in a dictionary with the attribute
    name `params` and contain a method called function that takes as input
    the radial distances and parameter values and returns the azimuthal
    position of the spiral arm.

Logarithmic
    Spiral pattern following a logarithmic pattern,
        θ(u,α) = ∫ du' / tan(α)

LogarithmicSigmoid
    Spiral pattern following a logarithmic pattern,
        θ(u,α) = ∫ du' / tan(α(u))
    where the pitch angle has a sigmoidal radial dependence
    between two pitch angles.
        α(r) = αᵢ + (α_f - αᵢ) ⋅ sigmoid(r, loc⋅isoA, scale⋅isoA)

Ringermacher
    Spiral arm pattern drawn from Ringermacher & Mead (2009),  
        θ(r,Wa,Wb,Wn) = 2⋅Wn⋅arctan(exp(-Wa/r) / Wb)

RingermacherPitch
    Spiral arm pattern drawn from Ringermacher & Mead (2009) based on the
    turnover pitch angle,
        θ(r,Φ,RΦ) = Φ ⋅ exp{(1 - RΦ/r) / (Φ⋅tan[Φ])}
"""
import math
import torch
from galkit.functional import sigmoid
from typing import Optional, Tuple, Union
from .logarithmic import θ_logarithmic
from .profile import ShockProfile
from ...random import random_normal, random_uniform

class SpiralPattern:
    """
    Parent class for constructing spiral patterns. Child classes
    should store the parameter variables in a dictionary with the attribute
    name `params` and contain a method called function that takes as input
    the radial distances and parameter values and returns the azimuthal
    position of the spiral arm.

    Parameters
    ----------
    arm_count : callable
        A function that takes as input the number of galaxies and the device
        and generates the number of arms for each galaxy.

    amplitude : callable
        A function that takes as input the number of arms and the device and
        returns the relative strengths of the arm components.

    phase : callable
        A function that takes as input the number of spiral arms and returns
        the phase angle of the pattern for each arm. This is used to separate
        the individual arms.

    profile : callable
        Class method that generates the spiral perturbation and spiral arm mask.
        Should have the method `sample` implented, which should take as input
        the galaxy class, the number of arms, and the isoA value.

    sign : callable
        A function that takes as input the number of galaxies and the device
        and returns the sign of the rotation for each.
    """
    def __init__(self,
        amplitude : callable = lambda size, device : random_normal(0, 1, size, device).softmax(-1),
        arm_count : callable = lambda size, device : torch.multinomial(torch.tensor([2, 0, 1, 1, 1, 1], dtype=torch.float32, device=device), num_samples=size, replacement=True),
        phase     : callable = lambda size, device : random_uniform(-math.pi, math.pi, 1, device) + torch.arange(-math.pi, math.pi, 2*math.pi/size, device=device) + random_normal(0, math.pi/(4*size), size, device),
        profile   : callable = ShockProfile(),
        sign      : callable = lambda size, device : torch.randn(size, device=device).sign(),
    ):
        self.__dict__.update(**locals())

    def __call__(self, θ:torch.Tensor, r:torch.Tensor, index:int):
        """
        Computes the set of permutation tensors and spiral arm
        masks for the indicated galaxy index.

        Parameters
        ----------
        θ : Tensor
            The azimuthal positions of the disk.

        r : Tensor
            The radial distances of the disk.

        index : int
            The galaxy index to use for generating the profile.
        
        Returns
        -------
        perturbation : Tensor
            The perturbation tensor for modifying the flux.
        
        mask : Tensor
            The spiral arm masks.
        """
        data, _ = self[index]
        shared_pattern = self.shared_pattern(index)
        n = data['arm_count']
        A = data['amplitude']

        if n == 0:
            return torch.tensor(1., device=r.device), torch.tensor([], device=r.device)

        if shared_pattern:
            θ_spiral = self.rotation(index)(r=r)

        masks = []
        perturbations = []
        for i in range(n):
            if not shared_pattern:
                θ_spiral = self.rotation(index, i)(r=r)

            perturbation, mask = self.profile(
                θ = θ,
                r = r,
                θ_spiral = θ_spiral + data['phase'][i],
                index = index,
                arm_index = i,
                sign = data['sign'],
            )

            masks.append(mask)
            perturbations.append(perturbation)

        masks = torch.cat(masks)
        perturbations = A.view(-1,1,1) * torch.cat(perturbations)

        return perturbations, masks

    def __getitem__(self, index):
        """
        Returns the parameters associated with the specified index.
        """
        return (
            {k:v[index] for k,v in self.data.items()}, 
            {k:v[index] for k,v in self.params.items()},
        )

    def __iter__(self):
        """
        Iterates through the pattern.
        """
        for i in range(len(self)):
            yield self.pattern(i)

    def __len__(self):
        return len(self.data['arm_count'])

    def pattern(self, 
        index:Optional[int] = None
    ) -> Union[callable, Tuple[callable]]:
        """
        Returns a callable function that acts as an interface to __call__.
        If index is None, then a tuple of such functions is returned.

        Parameters
        ----------
        index : int, optional
            The galaxy index.

        Returns
        -------
        pattern : callable, Tuple[callable]
            A callable function that serves as an interface to __call__
            of a tuple of such functions. The index parameter is given
            the provided value.
        """
        if index is None:
            return tuple(self.pattern(i) for i in range(len(self)))

        def pattern(θ, r, index=index, **kwargs):
            return self(θ=θ, r=r, index=index, **kwargs)

        return pattern

    def rotation(self,
        index:Optional[int] = None
    ) -> Union[callable, Tuple[callable]]:
        """
        Provides a callable function that returns the spiral pattern.
        If index is None, then a tuple of such functions is returned.

        Parameters
        ----------
        index : int, optional
            The index for which to generate the rotation function. If None,
            then the function is generated for each sample.

        Returns
        -------
        rotation : callable, Tuple[callable]
            A function that takes as input the radial distances, the arm
            index with a default value of 0, and any additional keyword
            arguments, or a tuple of such functions.
        """
        if index is None:
            return tuple(self.rotation(i) for i in range(len(self)))

        data, params = self[index]
        sign = data['sign']

        def rotation(r, arm_index=0, **kwargs):
            return self.function(
                r  = r,
                **{k:v if (v.nelement() == 1) else v[arm_index] for k,v in params.items()}
            ) * (sign if sign.nelement() == 1 else sign[arm_index])

        return rotation

    def sample(self, cls, isoA:torch.Tensor, **kwargs) -> None:
        """
        Generates random values for the different parameters.

        Parameters
        ----------
        cls : class
            The galaxy class object.

        isoA : tensor
            The scaling factors used to rescale the base grid when generating
            the geometry. The Wa values are multiplied by this.

        **kwargs
            Used for possible compatibility with other classes.
        """
        arm_count = self.arm_count(size=len(cls), device=cls.device)
        self.data = {
            'arm_count' : arm_count,
            'isoA'      : isoA,
            'sign'      : self.sign(len(cls), device=cls.device),
            'amplitude' : tuple(self.amplitude(n, cls.device) for n in arm_count),
            'phase'     : tuple(self.phase(n, device=cls.device) for n in arm_count),
        }
        self.profile.sample(cls=cls, arm_count=arm_count, isoA=isoA)

    def shared_pattern(self, index:int) -> bool:
        """
        Determines whether the spiral arm patterns are the same
        (True) or different (False).
        """
        _, params = self[index]
        for v in params.values():
            if v.nelement() > 1:
                return False
        return True

class Logarithmic(SpiralPattern):
    """
    Spiral pattern following a logarithmic pattern,

        θ(u,α) = ∫ du' / tan(α)

    Parameters
    ----------
    α : callable
        A function that takes as input the size (number of arms)
        and device and returns the pitch angles.

    **kwargs
        Any additional arguments to pass into the parent class.
    """
    def __init__(self,
        α : callable = lambda size, device : random_normal(20, 5, 1, device).clip(5, 50).deg2rad(),
        **kwargs,
    ):
        self.α = α
        super().__init__(**kwargs)

    def function(self,
        r  : torch.Tensor,
        α  : Union[callable, float, torch.Tensor], 
        r0 : Union[float, torch.Tensor] = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        The functional form of the expression.
        """
        return θ_logarithmic(α=α, r=r, r0=r0)

    def sample(self, cls, isoA:torch.Tensor, **kwargs) -> None:
        """
        Generates random values for the different parameters, which are
        stored in the attribute `params`.

        Parameters
        ----------
        cls : class
            The galaxy class object.

        isoA : tensor
            The scaling factors used to rescale the base grid when generating
            the geometry. The Wa values are multiplied by this.

        **kwargs
            Used for possible compatibility with other classes.
        """
        super().sample(cls=cls, isoA=isoA, **kwargs)

        arm_count = self.data['arm_count']
        self.params = ({
            'α' : tuple(self.α(n, device=cls.device) for n in arm_count),            
        })

class LogarithmicSigmoid(SpiralPattern):
    """
    Spiral pattern following a logarithmic pattern,

        θ(u,α) = ∫ du' / tan(α(u))

    where the pitch angle has a sigmoidal radial dependence
    between two pitch angles.

        α(r) = αᵢ + (α_f - αᵢ) ⋅ sigmoid(r, loc⋅isoA, scale⋅isoA)

    Parameters
    ----------
    α_i : callable
        A function that takes as input the size (number of arms)
        and device and returns the inner pitch angles.

    α_f : callable
        A function that takes as input the size (number of arms)
        and device and returns the outer pitch angles.

    loc : callable
        A function that takes as input the size (number of arms)
        and device and returns the transition point. Note that the
        returned value is multiplied by the isoA value.

    scale : callable
        A function that takes as input the size (number of arms)
        and device and returns the scale factor. Note that the
        returned value is multiplied by the isoA value.
    """
    def __init__(self,
        α_i     : callable = lambda size, device : random_normal(20, 5, 1, device).clip(5, 50).deg2rad(),
        α_f     : callable = lambda size, device : random_normal(20, 5, 1, device).clip(5, 50).deg2rad(),
        loc     : callable = lambda size, device : random_uniform(0, 1.0, 1, device),
        scale   : callable = lambda size, device : random_uniform(0.1, 0.5, 1, device),
        **kwargs,
    ):
        self.α_i = α_i
        self.α_f = α_f
        self.loc = loc
        self.scale = scale
        super().__init__(**kwargs)

    def function(self,
        r     : torch.Tensor,
        α_i   : Union[float, torch.Tensor], 
        α_f   : Union[float, torch.Tensor],
        loc   : Union[float, torch.Tensor],
        scale : Union[float, torch.Tensor],
        r0    : Union[float, torch.Tensor] = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        The functional form of the expression.
        """
        def α(r, *args, **kwargs):
            return α_i + (α_f-α_i)*sigmoid(r, loc=loc, scale=scale)
        return θ_logarithmic(α=α, r=r, r0=r0)

    def sample(self, cls, isoA:torch.Tensor, **kwargs) -> None:
        """
        Generates random values for the different parameters, which are
        stored in the attribute `params`.

        Parameters
        ----------
        cls : class
            The galaxy class object.

        isoA : tensor
            The scaling factors used to rescale the base grid when generating
            the geometry. The r parameter values are multiplied by this.

        **kwargs
            Used for possible compatibility with other classes.
        """
        super().sample(cls=cls, isoA=isoA, **kwargs)

        arm_count = self.data['arm_count']
        self.params = ({
            'α_i'    : tuple(self.α_i(n, device=cls.device) for n in arm_count),            
            'α_f'    : tuple(self.α_f(n, device=cls.device) for n in arm_count),            
            'loc'    : tuple(self.loc(n, device=cls.device)*isoA[i] for i,n in enumerate(arm_count)),            
            'scale'  : tuple(self.scale(n, device=cls.device)*isoA[i]  for i,n in enumerate(arm_count)),            
        })

class Ringermacher(SpiralPattern):
    """
    Spiral arm pattern drawn from Ringermacher & Mead (2009),
    
        θ(r,Wa,Wb,Wn) = 2⋅Wn⋅arctan(exp(Wa/r) / Wb)

    Parameters
    ----------
    Wa : callable
        A function that takes as input the size (number of arms)
        and device and returns the Wa parameters.

    Wb : callable
        A function that takes as input the size (number of arms)
        and device and returns the Wb parameters.

    Wn : callable
        A function that takes as input the size (number of arms)
        and device and returns the Wn parameters.

    **kwargs
        Any additional arguments to pass into the parent class.

    References
    ----------
    https://ui.adsabs.harvard.edu/abs/2009MNRAS.397..164R/abstract
    """
    def __init__(self,
        Wa : callable = lambda size, device : random_uniform(.1, 1, 1, device),
        Wb : callable = lambda size, device : random_uniform(math.exp(0.10), math.exp(1.0), 1, device).log(),
        Wn : callable = lambda size, device : random_uniform(math.exp(1.00), math.exp(10.), 1, device).log(),
        **kwargs,
    ):
        self.Wa = Wa
        self.Wb = Wb
        self.Wn = Wn
        super().__init__(**kwargs)

    def sample(self, cls, isoA:torch.Tensor, **kwargs) -> None:
        """
        Generates random values for the different parameters, which are
        stored in the attribute `params`.

        Parameters
        ----------
        cls : class
            The galaxy class object.

        isoA : tensor
            The scaling factors used to rescale the base grid when generating
            the geometry. The Wa values are multiplied by this.

        **kwargs
            Used for possible compatibility with other classes.
        """
        super().sample(cls=cls, isoA=isoA, **kwargs)

        arm_count = self.data['arm_count']
        self.params = ({
            'Wa' : tuple(self.Wa(n, device=cls.device).mul(isoA[i]) for i,n in enumerate(arm_count)),
            'Wb' : tuple(self.Wb(n, device=cls.device) for n in arm_count),            
            'Wn' : tuple(self.Wn(n, device=cls.device) for n in arm_count),
        })

    def function(self,
        r  : torch.Tensor, 
        Wn : Union[float, torch.Tensor], 
        Wb : Union[float, torch.Tensor], 
        Wa : Union[float, torch.Tensor] = 1, 
        **kwargs
    ) -> torch.Tensor:
        """
        The functional form of the expression. Note that the r(θ) definition
        used by Ringermacher and Meaf (2009) returns a negative value for the
        radius, so a negative value has been added to the exponent to change the sign.
        """
        return (2*Wn)*torch.arctan(torch.exp(-Wa/r) / Wb)

class RingermacherPitch(SpiralPattern):
    """
    Spiral arm pattern drawn from Ringermacher & Mead (2009) based on the
    turnover pitch angle,
    
        θ(r,Φ,RΦ) = Φ ⋅ exp{(1 - RΦ/r) / (Φ⋅tan[Φ])}

    Parameters
    ----------
    Φ : callable
        A function that takes as input the size (number of arms)
        and device and returns the Φ parameters.

    RΦ : callable
        A function that takes as input the size (number of arms)
        and device and returns the RΦ parameters.

    **kwargs
        Any additional arguments to pass into the parent class.

    References
    ----------
    https://ui.adsabs.harvard.edu/abs/2009MNRAS.397..164R/abstract
    """
    def __init__(self,
        Φ  : callable = lambda size, device : random_uniform(0.4, 1.0, 1, device),
        RΦ : callable = lambda size, device : random_uniform(0.1,1, 1, device),
        **kwargs
    ):
        self.Φ = Φ
        self.RΦ = RΦ
        super().__init__(**kwargs)

    def sample(self, cls, isoA:torch.Tensor, **kwargs):
        """
        Generates random values for the different parameters.

        Parameters
        ----------
        cls : class
            The galaxy class object.

        isoA : tensor
            The scaling factors used to rescale the base grid when generating
            the geometry. The Wa values are multiplied by this.

         **kwargs
            Used for possible compatibility with other classes.
        """
        super().sample(cls=cls, isoA=isoA, **kwargs)

        arm_count = self.data['arm_count']
        self.params = {
            'Φ'  : tuple(self.Φ(n, cls.device) for n in arm_count),
            'RΦ' : tuple(self.RΦ(n, cls.device).mul(isoA[i]) for i,n in enumerate(arm_count)),
        }

    def function(self,
        r  : torch.Tensor,
        Φ  : torch.Tensor,
        RΦ : Union[float, torch.Tensor] = 1, 
        **kwargs
    ) -> torch.Tensor:
        return (1 - RΦ/r).div(Φ * Φ.tan()).exp().mul(Φ)