"""
A collection of methods that are commonly used.

Classes
-------
BaseGenerator
    Base generator for the SkyDetector, Star, and Galaxy classes.

Methods
-------
accessor(column, dataframe, dictionary, filter_band, index, **kwargs)
    Utility for accessing data that may be stored in a dataframe or
    a dictionary.   

local_file(filename)
    Returns the path to the file stored in the local __files__ directory.

load_generator(filename)
    Returns a Gaussian Multivariate instance of the data stored in
    the provided filename.
"""

import os
from copulas.multivariate import GaussianMultivariate
from galkit.utils import to_tensor
from pandas import DataFrame
from typing import Optional

class BaseGenerator:
    """
    Base generator for the SkyDetector, Star, and Galaxy classes.

    Attributes
    ----------
    data : DataFrame
        Once the method `sample` is run, then the generated data
        is stored in the `data` attribute as a pandas DataFrame.

    Methods
    -------
    __len__
        Returns the number of row elements in the attribute `data`.
    
    sample(size)
        Generates a sample of the specified size from the generator.

    Required Attributes
    ----------------
    generator : Multivariate
        A Multivariate copula for generating sythetic data.
    """
    def __len__(self):
        return len(self.data)

    def sample(self, size):
        self.data = self.generator.sample(size)

def accessor(
    column      : str,
    dataframe   : DataFrame,
    dictionary  : dict = {},
    filter_band : Optional[str] = None,
    index       : Optional[int] = None,
    **kwargs
):
    """
    Utility for accessing data that may be stored in a dataframe or
    a dictionary.    

    Parameters
    ----------
    column : str
        The base name of the column. Any filter band extensions are assumed
        to be on the end and will be added if needed.

    dataframe : DataFrame
        A dataframe object containing parameter values. If any of the parameters
        are filter dependent, then the names of these parameters are assumed to
        be of the form `{column}_{filter_band}`.

    dictionary : dict
        A dictionary containing parameter values. The format of the dictionary
        should be such that the `column` key is associated with a value or is
        associated with a dictionary where the keys of that dictionary correspond
        to the filter bands.

    filter_band : str
        The name of the filter band. Only needed if the `column` parameter depends
        on the filter band. The filter band is assumed to be appended to the end
        of the column name and preceeded by an underscore.

    index : int, optional
        If the parameter is contained in the dictionary, then this
        parameter is not used.

    **kwargs
        Any additional keyword arguments to pass into the `to_tensor` method.

    Returns
    -------
    value : Tensor
        The output values are cast into a tensor

    Raises
    ------
    ValueError
        If the column isn't found in the dictionary or dataframe, then this
        exception is raised.

    Examples
    --------
    import pandas
    import numpy
    from galsyn.utils import accessor

    size = 5
    dataframe = pandas.DataFrame({
        'flux_g': numpy.random.rand(size),
        'flux_r': numpy.random.rand(size) * 2,
        'flux_i': numpy.random.rand(size) * 3,
    })

    dictionary = {
        'fluxPerCount' : {'g': 1, 'r': 0.5, 'i': 0.25},
        'darkVariance' : 1,
    }

    kwargs = {
        'dataframe': dataframe, 
        'dictionary': dictionary, 
        'view': (-1,)
    }

    flux_g = accessor('flux', filter_band='g', **kwargs)
    print(f'flux_g = {flux_g}')

    flux_g_0 = accessor('flux', filter_band='g', index=0, **kwargs)
    print(f'flux_g[0] = {flux_g_0}')

    fpc = accessor('fluxPerCount', filter_band='r', **kwargs)
    print(f'flux_per_count_r = {fpc}')

    dark = accessor('darkVariance', filter_band=None, **kwargs)
    print(f'dark_variance = {dark}')
    """
    # ======================================================================
    # Check if column is in the dictionary. If so, then check if it
    # depends on the filter band and return
    # ======================================================================
    if column in dictionary:
        data = dictionary[column]
        return to_tensor(data[filter_band] if isinstance(data, dict) else data,**kwargs)

    # ======================================================================
    # Check if column is in the DataFrame. If so, then check if it
    # depends on the filter band and return. If index is passed, then just
    # return the specified index
    # ======================================================================
    if column in dataframe:
        data = dataframe[column].values
        return to_tensor(data if index is None else data[index], **kwargs)

    column = f'{column}_{filter_band}'
    if column in dataframe:
        data = dataframe[column].values
        return to_tensor(data if index is None else data[index], **kwargs)

    # ======================================================================
    # If haven't found yet, raise error
    # ======================================================================
    raise ValueError(f"Not able to find {column}")

def local_file(filename:str) -> str:
    """
    Returns the path to the file stored in the local __files__ directory.
    """
    return os.path.join(os.path.dirname(__file__), '__file__', filename)

def load_generator(filename:str) -> GaussianMultivariate:
    """
    Returns a Gaussian Multivariate instance of the data stored in
    the provided filename.
    """
    return GaussianMultivariate.load(filename)