from enum import Enum
import torch
from data_generation.simulations.grid import tangent_transformation, logistic_transformation

class SystemType(Enum):
    TECH_SUBSTITUTION = 'tech_substitution'
    SADDLE_SYSTEM = 'saddle_system'

SYSTEM_CONFIGS = {
    'tech_substitution': {
        'control_format': 'continuous',
        'control_dim': 1,
        'value_types': ['market_share', '90% market share', 'identity'],
        'default_value_type': 'market_share',
        'value_activation': {
            'market_share': 'sigmoid',  # [0,1] range for probabilities
            '90% market share': 'sigmoid',  # binary output
            'identity': None  # No activation for identity
        },
        'value_dim': {
            'market_share': 1,
            '90% market share': 1,
            'identity': 2
        },
        'default_value_loss': {
            'market_share': 'mse',
            '90% market share': 'binary_cross_entropy',
            'identity': 'mse'
        },
        'transformation': {
            'type': 'tangent',
            'params': {'x0': 3.0, 'alpha': 0.5}
        },
        'value_sorting_functions': {
            'market_share': lambda values: values[:, 0],  # Single value, sort directly
            'identity': lambda values: values[:, 1]  / values[:, 0] # similar to angle / market share sorting
        }
    },
    'saddle_system': {
        'control_format': 'categorical',
        'control_dim': None,
        'value_types': ['angle', 'identity'],
        'default_value_type': 'angle',
        'value_activation': {
            'angle': 'tanh',  # [-1,1] range for sin/cos
            'identity': None  # No activation for identity
        },
        'value_dim': {
            'angle': 2,
            'identity': 2
        },
        'default_value_loss': {
            'angle': 'angular',
            'identity': 'mse'
        },
        'transformation': {
            'type': 'logistic',
            'params': {'k': 0.5, 'x_0': 0.0}
        },
        'value_sorting_functions': {
            'angle': lambda values: (torch.atan2(values[:, 0], values[:, 1]) * 180 / torch.pi) % 360,
            'identity': lambda values: values[:, 0] + values[:, 1]  # Sum of transformed components
        }
    }
}

def get_system_config(system_type: SystemType) -> dict:
    """
    Get configuration for a specific system type
    
    Args:
        system_type: Type of system from SystemType enum
        
    Returns:
        Dictionary containing system configuration
    """
    return SYSTEM_CONFIGS[system_type.value]

def get_transformation(system_type: SystemType):
    """
    Get the transformation function for a given system
    
    Args:
        system_type: Type of system from SystemType enum
        
    Returns:
        Transformation function tuple (forward, inverse, derivative)
    """
    config = SYSTEM_CONFIGS[system_type.value]
    trans_config = config['transformation']
    
    if trans_config['type'] == 'tangent':
        params = trans_config['params']
        return tangent_transformation(params['x0'], params['alpha'])
    elif trans_config['type'] == 'logistic':
        params = trans_config['params']
        return logistic_transformation(params)
    else:
        raise ValueError(f"Unknown transformation type: {trans_config['type']}")
    
def get_value_sorting_function(system_type: SystemType, value_method: str):
    """
    Get the value sorting function for a given system and value method
    
    Args:
        system_type: Type of system from SystemType enum
        value_method: Value method name
        
    Returns:
        Function that takes values tensor and returns sorting keys
    """
    config = SYSTEM_CONFIGS[system_type.value]
    if 'value_sorting_functions' not in config:
        raise ValueError(f"No value sorting functions defined for {system_type.value}")
    
    if value_method not in config['value_sorting_functions']:
        raise ValueError(f"No sorting function for value method '{value_method}' in {system_type.value}")
    
    return config['value_sorting_functions'][value_method]