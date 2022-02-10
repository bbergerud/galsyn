"""
Methods associated with applying dust extinction.

Classes
-------
DustModel
    Base class for incorporating dust models from the dust_extinction library.

DustCCM89
    Cardelli, Clayton, & Mathis (1989) Milky Way R(V) dependent model.

DustF99
    Fitzpatrick (1999) Milky Way R(V) dependent model  
"""

import astropy.units as u
import torch
from dust_extinction.parameter_averages import CCM89, F99
from galkit.data.sdss import sdss_wavelengths
from typing import Dict, Iterable

def poly1d(p, x):
    y = 0
    for pi in p:
        y = y*x + pi
    return y

class CCM89_OpticalIR:
    """
    import torch
    import matplotlib.pyplot as plt
    from galsyn.galaxy.dust import CCM89_OpticalIR, DustCCM89

    y1 = CCM89_OpticalIR()
    y2 = DustCCM89()

    x = torch.linspace(0, 3, 1000)

    k = 'z'
    fig, ax = plt.subplots()
    ax.plot(x, y1(x, filter_bands=k)[k])
    ax.plot(x, y2(x, filter_bands=k)[k])
    fig.show()
    """
    def __init__(self,
        wavelengths : dict = sdss_wavelengths,
        Rv:float = 3.1
    ):
        wavenumber = {k:(v * 1e6)**-1 for k,v in wavelengths.items()}
        a_coeffs = (0.32999, -0.7753, 0.01979, 0.72085, -0.02427, -0.50447, 0.17699, 1)
        b_coeffs = (-2.09002, 5.3026, -0.62251, -5.38434, 1.07233, 2.28305, 1.41338, 0)

        self.Aλ_per_Av = {}
        for k,x in wavenumber.items():
            y = x - 1.82
            a = poly1d(a_coeffs, y)
            b = poly1d(b_coeffs, y)
            self.Aλ_per_Av[k] = a + b/Rv

    def __call__(self, Av, filter_bands):
        output = {}
        for k in filter_bands:
            output[k] = 10**(-0.4 * Av * self.Aλ_per_Av[k])
        return output

class DustModel:
    """
    Base class for incorporating dust models from the dust_extinction library.

    Attributes
    ----------
    model
        The dust profile model.

    wave : Dict
        Dictionary where the keys are the filter bands and the
        values the wavelength associated with the filter band.
    """
    def __init__(self, 
        model,
        wavelengths : dict = sdss_wavelengths,
        unit = u.meter,
    ):
        """
        Parameters
        ----------
        model
            A model of the dust profile. Should have an `extinguish`
            method that takes as input the wavelength in astropy units
            followed by the dust extinction Av array.

        wavelengths : dict
            A dictionary where the keys are the filter bands and the
            values are the wavelengths.
        
        unit
            The wavelength unit. Should be an astropy.unit item.
            The values in the wavelength dictionary are multiplied
            by this unit.
        """ 
        self.model = model
        self.wave  = {k:v * unit for k,v in wavelengths.items()}

    def __call__(self,
        Av : torch.Tensor, 
        filter_bands : Iterable,
    ) -> Dict[str,torch.Tensor]:
        """
        Returns the fractional flux of light that passes through the dust
        at each pixel position.

        Parameters
        ----------
        Av : tensor
            A tensor containing the Av extinction magnitudes.

        filter_bands : Iterable
            The sequence of filter bands to generate extinction values for.

        Returns
        -------
        output : dict
            A dictionary where the keys are the filter bands and the values are
            the fractional amount of light that passes through the dust.
        """
        numpy_input = Av.cpu().numpy()
        output = {}
        for k in filter_bands:
            result = self.model.extinguish(self.wave[k], numpy_input)
            output[k] = Av.new(result)

        return output

class DustCCM89(DustModel):
    """
    Cardelli, Clayton, & Mathis (1989) Milky Way R(V) dependent model.
    """
    def __init__(self, Rv:float=3.1, *args, **kwargs):
        super().__init__(CCM89(Rv=Rv), *args, **kwargs)

class DustF99(DustModel):
    """
    Fitzpatrick (1999) Milky Way R(V) dependent model    
    """
    def __init__(self, Rv:float=3.1, *args, **kwargs):
        super().__init__(F99(Rv=Rv), *args, **kwargs)