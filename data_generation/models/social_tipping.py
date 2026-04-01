import numpy as np

class SocialTipping:
    """
    ODE System for Interacting social tipping elements:social tipping of opinions and behaviours (toy model)
        x = fraction of population displaying green behaviour
        y = fraction of population holding green opinion

    Defalut Values: a = e = 1,  d = h = 0.1
    Possible modes: 'default', "subsidy", "tax", "campaign"
        mode = "default":   c = f = g = 0.65  b = 0.6
        mode = "subsidy":   b = c = f = g = 0.65
        mode = "tax":       f = g = 0.65,  b = c = 0.6 
        mode = "campaign":  c = f = 0.65,  b = g = 0.6
    """    

    def __init__(self, a=1, b=0.6, c=0.65, d=0.1, e=1, f=0.65, g=0.65, h=0.1, control_params=['b','c','f','g']):
        """
        ODE System for social tipping of opinions and behaviours (toy model)
        Args: 
              a,b,c,e,f,g,h (scalars): parameters 
              control_params (list of strings): list of strings with expected entries in ['b','c','f','g']
        """
        # dimensions
        self.x_dim = 2 # v, w
        self.control_dim = len(control_params) 
        self.control_params = control_params
        
        # Params
        self.params = dict()
        self.params['a'] = a  if 'a' not in control_params else None
        self.params['b'] = b if 'b' not in control_params else None
        self.params['c'] = c if 'c' not in control_params else None
        self.params['d'] = d if 'd' not in control_params else None
        self.params['e'] = e if 'e' not in control_params else None
        self.params['f'] = f if 'f' not in control_params else None
        self.params['g'] = g if 'g' not in control_params else None
        self.params['h'] = h if 'h' not in control_params else None

    def get_config(self):
        config = {
            'model': self.__class__.__name__,
            'x_dim': self.x_dim,
            'control_dim': self.control_dim,
            'control_params': self.control_params,
            'a': self.params['a'],
            'b': self.params['b'],
            'c': self.params['c'],
            'd': self.params['d'],
            'e': self.params['e'],
            'f': self.params['f'],
            'g': self.params['g'],
            'h': self.params['h']
        }
        return config

    def get_control_params(self, action):
        if action == "default":
            control_params = {
                'b': 0.6,
                'c': 0.65,
                'f': 0.65,
                'g': 0.65
            }
        elif action == "subsidy":
            control_params = {
                'b': 0.65,
                'c': 0.65,
                'f': 0.65,
                'g': 0.65
            }
        elif action == "tax":
            control_params = {
                'b': 0.6,
                'c': 0.6,
                'f': 0.65,
                'g': 0.65
            }
        elif action == "campaign":
            control_params = {
                'b': 0.6,
                'c': 0.65,
                'f': 0.65,
                'g': 0.6
            }
        else:
            raise ValueError(f"Unknown action: '{action}'. Valid actions are: 'default', 'subsidy', 'tax', 'campaign'")

        return control_params

    def odes(self, t, z, control_params):
        """
        Compute the derivatives for a 2-dimensional vector [v,w] and a dictionary of parameter.
        
        Args:
            t (float): The current time. (Doesn't matter here, the system is time-independet)
            z (2-dim list or array-like): The current state of the system, where z[0] is v and z[1] is w.
            control_params (dict): A dictionary containing control parameters with possible keys:
                - p in ['b','c','f','g']: Parameter p 
                
        Returns:
            np.array (2dim): A np.array containing the derivatives [dvdt, dwdt]
        """

        # Extract control parameters with default values
        a = control_params.get('a', self.params['a'])
        b = control_params.get('b', self.params['b'])
        c = control_params.get('c', self.params['c'])
        d = control_params.get('d', self.params['d'])
        e = control_params.get('e', self.params['e'])
        f = control_params.get('f', self.params['f'])
        g = control_params.get('g', self.params['g'])
        h = control_params.get('h', self.params['h'])
        
        x, y = z
        dxdt = a * (1-x)*(x**2) *(b + (1-b)*y) - a * x*((1-x)**2) *(c + (1-c)*(1-y)) + d*(1-2*x)
        dydt = e * (1-y)*(y**2) *(f + (1-f)*(1-x)) - e * y*(1-y)**2 *(g + (1-g)*x) + h*(1-2*y)
        return np.array([dxdt, dydt])


    def odes_vectorized(self, t, Z, control_params):
        """
        Vectorized computation of derivatives for n samples simultaneously.

        Args:
            t (float): Current time (not used — autonomous system, required by solver interface).
            Z (1D np.array): Flattened state array of shape (2*n,),
                             structured as [x1, ..., xn, y1, ..., yn].
            control_params (dict): Parameters for the system. Each value is a numpy array
                                   of length n. Control params (typically b, c, f, g) must
                                   be provided; fixed params (a, d, e, h) fall back to
                                   self.params if absent.

        Returns:
            np.array: 1D array of shape (2*n,) with derivatives,
                      structured as [dx1/dt, ..., dxn/dt, dy1/dt, ..., dyn/dt].
        """

        x, y = Z[:int(len(Z)/2)], Z[int(len(Z)/2):]  
        n_samples = len(x)
        
        # Extract control parameters with default values
        a = control_params.get('a', np.full(n_samples, self.params['a']))
        b = control_params.get('b', np.full(n_samples, self.params['b']))
        c = control_params.get('c', np.full(n_samples, self.params['c']))
        d = control_params.get('d', np.full(n_samples, self.params['d']))
        e = control_params.get('e', np.full(n_samples, self.params['e']))
        f = control_params.get('f', np.full(n_samples, self.params['f']))
        g = control_params.get('g', np.full(n_samples, self.params['g']))
        h = control_params.get('h', np.full(n_samples, self.params['h']))

        #ODEs
        dxdt = a * (1-x)*(x**2) *(b + (1-b)*y) - a * x*((1-x)**2) *(c + (1-c)*(1-y)) + d*(1-2*x)
        dydt = e * (1-y)*(y**2) *(f + (1-f)*(1-x)) - e * y*(1-y)**2 *(g + (1-g)*x) + h*(1-2*y)

        
        return np.append(dxdt, dydt)
