import numpy as np


def fractional_transformation(param):
    """
    Generates the fractional transformation z=x/(x+param), its inverse and its derivate for a given parameter.
    Useful for space compression in the case [0, np.inf], cannot be used for spaces which include x = -param.
    
    Args:
        param (float, > 0): The parameter to be used in the fractional transformations
    Returns:
        tuple: A tuple containing three functions:
            - frac_transformation (function): Transforms an input x using the formula x / (x + param).
            - inverse_frac_transformation (function): Computes the inverse of the fractional transformation.
            - frac_transformation_derivative (function): Computes the derivative of the fractional transformation.
    
    """

    def frac_transformation(x):
        if np.isinf(x):
            return 1
        else:
            return x / (x+param)
        
    frac_transformation.parameters = {'param': param}

    def inverse_frac_transformation(z):
        if z == 1:
            return np.inf
        else:
            return -param*z / (z-1)
        
    def frac_transformation_derivative(x):
        if np.isinf(x):
            return 0
        else:
            return param / (x+param)**2
        
    
    return (frac_transformation, inverse_frac_transformation, frac_transformation_derivative)