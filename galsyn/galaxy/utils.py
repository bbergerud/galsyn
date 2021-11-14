"""
Utility functions associated with the galaxy data.

Functions
---------
access(column, component, **kwargs)
    Interface to the accessor method that allows for
    the column element to be a dictionary where the
    keys are the components.
"""
from typing import Union
from ..utils import accessor

def access(
    column : Union[str, dict],
    component : str,
    **kwargs
):
    """
    Interface to the accessor method that allows for
    the column element to be a dictionary where the
    keys are the components.

    Parameters
    ----------
    column : str, dict
        Either a string representing the column or a dictionary
        where the keys are the components and the values are the
        strings representing the column.

    component : str
        The component to access the data for.

    **kwargs
        Additional arguments for the accessor function.
    """
    if isinstance(column, dict):
        column = column[component]
    return accessor(f'{column}_{component}', **kwargs)