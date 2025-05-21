from enum import Enum
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