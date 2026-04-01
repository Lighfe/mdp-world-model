import numpy as np

class FitzHughNagumoModel:

    def __init__(self, a = 0.7, b = 2.0, epsilon = 0.08, I = 0.35, control_params=[]):
        """
        Create the FitzHughNagumo Model corresponding to the given default values and later control parameters.
        Later, it will be important to hand-over the control parameters always in the same order as specified here.

        Args: a, b, epsilon, I (scalars): default parameters
              control_params (list of strings): list of strings with expected entries in ["a", "b", "epsilon", "I" ]
        """
    # https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model
    # in wikipedia 1/epsilon = tau = 12.5 and I = R*I_ext
    # Experiment with 3 actions 
    # 1) b=2, I = 0.35 ---- two basins of attraction
    # 2) b=1.5, I=0.0  ---- single stable fix point
    # 3) b=0.8, I=0.5  ---- limit cycle

        # dimensions
        self.x_dim = 2 # v, w
        self.control_dim = len(control_params) 
        self.control_params = control_params
        
        # Params
        self.params = dict()
        self.params['a'] = a if 'a' not in control_params else None
        self.params['b'] = b if 'b' not in control_params else None
        self.params['epsilon'] = epsilon if 'epsilon' not in control_params else None
        self.params['I'] = I if 'I' not in control_params else None


    def get_config(self):
        config = {
            'model': self.__class__.__name__,
            'x_dim': self.x_dim,
            'control_dim': self.control_dim,
            'control_params': self.control_params,
            'a': self.params['a'],
            'b':  self.params['b'],
            'epsilon': self.params['epsilon'],
            'I': self.params['I'] ,
        }
        return config
        
    def odes(self, t, z, control_params):
        """
        Compute the derivatives for a 2-dimensional vector [v,w] and a dictionary of parameter.
        
        Args:
            t (float): The current time. (Doesn't matter here, the system is time-independet)
            z (2-dim list or array-like): The current state of the system, where z[0] is v and z[1] is w.
            control_params (dict): A dictionary containing control parameters with possible keys:
                - 'a': Parameter a (default is self.a)
                - 'b': Parameter b (default is self.b)
                - 'epsilon': Parameter epsilon (default is self.epsilon)
                - 'I': Parameter I (default is self.I)
        Returns:
            np.array (2dim): A np.array containing the derivatives [dvdt, dwdt]
        """

        # Extract control parameters with default values
        a = control_params.get('a', self.params['a'])
        b = control_params.get('b', self.params['b'])
        epsilon = control_params.get('epsilon',self.params['epsilon'])
        I = control_params.get('I', self.params['I'])
        
        v, w = z
        dvdt = v - (v**3) / 3 - w + I
        dwdt = epsilon * (v + a - b * w)
        return np.array([dvdt, dwdt])
    
    
    def odes_vectorized(self, t, Z, control_params):
        """
        Compute the derivatives of the state variables v and w for a system of ODEs in a vectorized manner.
        Args:
            t(float): The current time (not used in the computation but required by the ODE solver interface).
            Z ( 1D np.array): state variables, structured as [v1, v2, ..., vn, w1, w2, ..., wn].
            control_params(dict) : 
                A dictionary containing the system parameters. Each entry is a numpy array of length n (number of samples).
                Expected keys are: 'a', 'b', 'epsilon' and 'I' with default values in self.params
        Returns:
        
        np.array
            A 1D numpy array containing the derivatives of the state variables, structured as [dv1/dt, dv2/dt, ..., dvn/dt, dw1/dt, dw2/dt, ..., dwn/dt].
        """

        v, w = Z[:int(len(Z)/2)], Z[int(len(Z)/2):]  
        n_samples = len(v)
        
        # Extract control parameters with default values
        a = control_params.get('a', np.full(n_samples, self.params['a']))
        b = control_params.get('b', np.full(n_samples, self.params['b']))
        epsilon = control_params.get('epsilon', np.full(n_samples, self.params['epsilon']))
        I = control_params.get('I', np.full(n_samples, self.params['I']))

        #ODEs
        dvdt = v - (v**3) / 3 - w + I
        dwdt = epsilon * (v + a - b * w)

        
        return np.append(dvdt, dwdt)
    