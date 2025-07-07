import numpy as np


class UniformFlow:
    """
    ODE System for a steady parallel flow from the left to the right for c = 0.
        For 0 < c <= 10 it changes it's direction from horizontal to vertical.
    """

    def __init__(self, c = 0, control_params=[]):
        """
        
        Args: c (scalars): default parameters
              control_params (list of strings): list of strings with expected entries in ["c"]
        """
        # dimensions
        self.x_dim = 2 # v, w
        self.control_dim = len(control_params) 
        self.control_params = control_params
        
        # Params
        self.params = dict()
        self.params['c'] = c if 'c' not in control_params else None

    def get_config(self):
        config = {
            'model': self.__class__.__name__,
            'x_dim': self.x_dim,
            'control_dim': self.control_dim,
            'control_params': self.control_params,
            'c': self.params['c'],
        }
        return config
        
    def odes(self, t, z, control_params):
        """
        Compute the derivatives for a 2-dimensional vector [v,w] and a dictionary of parameter.
        
        Args:
            t (float): The current time. (Doesn't matter here, the system is time-independet)
            z (2-dim list or array-like): The current state of the system, where z[0] is v and z[1] is w.
            control_params (dict): A dictionary containing control parameters with possible keys:
                - 'c': Parameter c (default is self.c)
        Returns:
            np.array (2dim): A np.array containing the derivatives [dvdt, dwdt]
        """

        # Extract control parameters with default values
        c= control_params.get('c', self.params['c'])
        
        v, w = z
        dvdt = 10 -c
        dwdt = c
        return np.array([dvdt, dwdt])
    
    
    def odes_vectorized(self, t, Z, control_params):
        """
        Compute the derivatives of the state variables v and w for a system of ODEs in a vectorized manner.
        Args:
            t(float): The current time (not used in the computation but required by the ODE solver interface).
            Z ( 1D np.array): state variables, structured as [v1, v2, ..., vn, w1, w2, ..., wn].
            control_params(dict) : 
                A dictionary containing the system parameters. Each entry is a numpy array of length n (number of samples).
                Expected keys are: 'c'  with default values in self.params
        Returns:
        
        np.array
            A 1D numpy array containing the derivatives of the state variables, structured as [dv1/dt, dv2/dt, ..., dvn/dt, dw1/dt, dw2/dt, ..., dwn/dt].
        """

        v, w = Z[:int(len(Z)/2)], Z[int(len(Z)/2):]  
        n_samples = len(v)
        
        # Extract control parameters with default values
        c = control_params.get('c', np.full(n_samples, self.params['c']))

        #ODEs
        dvdt = np.full(n_samples, 10) - c
        dwdt = c

        
        return np.append(dvdt, dwdt)
    


class StableNode:
    """
    ODE System for a stable node at (0,0)
    Default c = 1 is a stable star
    For c>0, the higher c, the more attraction towards the y-axis
    For c<0 the system becomes unstable
    """

    def __init__(self, c = 1, control_params=[]):
        """
    
        Args: c (scalars): default parameters
              control_params (list of strings): list of strings with expected entries in ["c"]
        """
        # dimensions
        self.x_dim = 2 # v, w
        self.control_dim = len(control_params) 
        self.control_params = control_params
        
        # Params
        self.params = dict()
        self.params['c'] = c if 'c' not in control_params else None

    def get_config(self):
        config = {
            'model': self.__class__.__name__,
            'x_dim': self.x_dim,
            'control_dim': self.control_dim,
            'control_params': self.control_params,
            'c': self.params['c'],
        }
        return config
        
    def odes(self, t, z, control_params):
        """
        Compute the derivatives for a 2-dimensional vector [v,w] and a dictionary of parameter.
        
        Args:
            t (float): The current time. (Doesn't matter here, the system is time-independet)
            z (2-dim list or array-like): The current state of the system, where z[0] is v and z[1] is w.
            control_params (dict): A dictionary containing control parameters with possible keys:
                - 'c': Parameter c (default is self.c)
        Returns:
            np.array (2dim): A np.array containing the derivatives [dvdt, dwdt]
        """

        # Extract control parameters with default values
        c= control_params.get('c', self.params['c'])
        
        v, w = z
        dvdt = - c * v
        dwdt = - w
        return np.array([dvdt, dwdt])
    
    
    def odes_vectorized(self, t, Z, control_params):
        """
        Compute the derivatives of the state variables v and w for a system of ODEs in a vectorized manner.
        Args:
            t(float): The current time (not used in the computation but required by the ODE solver interface).
            Z ( 1D np.array): state variables, structured as [v1, v2, ..., vn, w1, w2, ..., wn].
            control_params(dict) : 
                A dictionary containing the system parameters. Each entry is a numpy array of length n (number of samples).
                Expected keys are: 'c'  with default values in self.params
        Returns:
        
        np.array
            A 1D numpy array containing the derivatives of the state variables, structured as [dv1/dt, dv2/dt, ..., dvn/dt, dw1/dt, dw2/dt, ..., dwn/dt].
        """

        v, w = Z[:int(len(Z)/2)], Z[int(len(Z)/2):]  
        n_samples = len(v)
        
        # Extract control parameters with default values
        c = control_params.get('c', np.full(n_samples, self.params['c']))

        #ODEs
        dvdt = - c * v
        dwdt = - w 

        
        return np.append(dvdt, dwdt)
    

class TwoStableNodesOneSaddlePoint:
    """
    ODE System for two stable nodes at (-1,c) and (1,c) as well as a saddle point at (0,c)
    Standard control at c = 0, changing c changes the vertical position of the fixpoints, moving them up/down
    """    

    def __init__(self, c = 0, control_params=[]):
        """
        ODE System for 
        Args: c (scalars): default parameters
                control_params (list of strings): list of strings with expected entries in ["c"]
        """
        # dimensions
        self.x_dim = 2 # v, w
        self.control_dim = len(control_params) 
        self.control_params = control_params
        
        # Params
        self.params = dict()
        self.params['c'] = c if 'c' not in control_params else None

    def get_config(self):
        config = {
            'model': self.__class__.__name__,
            'x_dim': self.x_dim,
            'control_dim': self.control_dim,
            'control_params': self.control_params,
            'c': self.params['c'],
        }
        return config
        
    def odes(self, t, z, control_params):
        """
        Compute the derivatives for a 2-dimensional vector [v,w] and a dictionary of parameter.
        
        Args:
            t (float): The current time. (Doesn't matter here, the system is time-independet)
            z (2-dim list or array-like): The current state of the system, where z[0] is v and z[1] is w.
            control_params (dict): A dictionary containing control parameters with possible keys:
                - 'c': Parameter c (default is self.c)
        Returns:
            np.array (2dim): A np.array containing the derivatives [dvdt, dwdt]
        """

        # Extract control parameters with default values
        c= control_params.get('c', self.params['c'])
        
        v, w = z
        dvdt = - v * (v-1) * (v+1)
        dwdt = - w + c
        return np.array([dvdt, dwdt])


    def odes_vectorized(self, t, Z, control_params):
        """
        Compute the derivatives of the state variables v and w for a system of ODEs in a vectorized manner.
        Args:
            t(float): The current time (not used in the computation but required by the ODE solver interface).
            Z ( 1D np.array): state variables, structured as [v1, v2, ..., vn, w1, w2, ..., wn].
            control_params(dict) : 
                A dictionary containing the system parameters. Each entry is a numpy array of length n (number of samples).
                Expected keys are: 'c'  with default values in self.params
        Returns:
        
        np.array
            A 1D numpy array containing the derivatives of the state variables, structured as [dv1/dt, dv2/dt, ..., dvn/dt, dw1/dt, dw2/dt, ..., dwn/dt].
        """

        v, w = Z[:int(len(Z)/2)], Z[int(len(Z)/2):]  
        n_samples = len(v)
        
        # Extract control parameters with default values
        c = control_params.get('c', np.full(n_samples, self.params['c']))

        #ODEs
        dvdt = - v * (v-1) * (v+1)
        dwdt = - w + c

        
        return np.append(dvdt, dwdt)


class Movable2Stable1SaddlePoint:
    """
    ODE System for two stable nodes at (-1+h,c) and (1+h,c) as well as a saddle point at (h,c)
    Standard control at h= 0, c = 0, 
        changing c changes the vertical position of the fixpoints, moving them up/down
        changing h changes the horizontal position of the fixpoints
    """    

    def __init__(self, c = 0, h= 0, control_params=[]):
        """
        ODE System for 
        Args: c (scalars): default parameters
              h (scalars): default parameters
              control_params (list of strings): list of strings with expected entries in ["c"]
        """
        # dimensions
        self.x_dim = 2 # v, w
        self.control_dim = len(control_params) 
        self.control_params = control_params
        
        # Params
        self.params = dict()
        self.params['c'] = c if 'c' not in control_params else None
        self.params['h'] = h if 'h' not in control_params else None

    def get_config(self):
        config = {
            'model': self.__class__.__name__,
            'x_dim': self.x_dim,
            'control_dim': self.control_dim,
            'control_params': self.control_params,
            'c': self.params['c'],
            'h': self.params['h']
        }
        return config
        
    def odes(self, t, z, control_params):
        """
        Compute the derivatives for a 2-dimensional vector [v,w] and a dictionary of parameter.
        
        Args:
            t (float): The current time. (Doesn't matter here, the system is time-independet)
            z (2-dim list or array-like): The current state of the system, where z[0] is v and z[1] is w.
            control_params (dict): A dictionary containing control parameters with possible keys:
                - 'c': Parameter c (default is self.c)
                - 'h': Parameter h (default is self.h)
        Returns:
            np.array (2dim): A np.array containing the derivatives [dvdt, dwdt]
        """

        # Extract control parameters with default values
        c= control_params.get('c', self.params['c'])
        h= control_params.get('h', self.params['h'])
        
        v, w = z
        dvdt = - (v-h) * (v -h -1) * (v- h +1)
        dwdt = - w + c
        return np.array([dvdt, dwdt])


    def odes_vectorized(self, t, Z, control_params):
        """
        Compute the derivatives of the state variables v and w for a system of ODEs in a vectorized manner.
        Args:
            t(float): The current time (not used in the computation but required by the ODE solver interface).
            Z ( 1D np.array): state variables, structured as [v1, v2, ..., vn, w1, w2, ..., wn].
            control_params(dict) : 
                A dictionary containing the system parameters. Each entry is a numpy array of length n (number of samples).
                Expected keys are: 
                        'c' and 'h'  with default values in self.params
        Returns:
        
        np.array
            A 1D numpy array containing the derivatives of the state variables, structured as [dv1/dt, dv2/dt, ..., dvn/dt, dw1/dt, dw2/dt, ..., dwn/dt].
        """

        v, w = Z[:int(len(Z)/2)], Z[int(len(Z)/2):]  
        n_samples = len(v)
        
        # Extract control parameters with default values
        c = control_params.get('c', np.full(n_samples, self.params['c']))
        h = control_params.get('h', np.full(n_samples, self.params['h']))

        #ODEs
        dvdt = - (v-h) * (v -h -1) * (v- h +1)
        dwdt = - w + c

        
        return np.append(dvdt, dwdt)


class Stretchable2Stable1SaddlePoint:
    """
    ODE System for two stable nodes at (-h,c) and (h,c) as well as a saddle point at (0,c)
    Standard control at h= 0, c = 0, 
        changing c changes the vertical position of the fixpoints, moving them up/down
        changing h changes the horizontal position of the fixpoints, moving them further apart larger |h|
    """    

    def __init__(self, c = 0, h= 1, control_params=[]):
        """
        ODE System for 
        Args: c (scalars): default parameters
              h (scalars): default parameters
              control_params (list of strings): list of strings with expected entries in ["c"]
        """
        # dimensions
        self.x_dim = 2 # v, w
        self.control_dim = len(control_params) 
        self.control_params = control_params
        
        # Params
        self.params = dict()
        self.params['c'] = c if 'c' not in control_params else None
        self.params['h'] = h if 'h' not in control_params else None

    def get_config(self):
        config = {
            'model': self.__class__.__name__,
            'x_dim': self.x_dim,
            'control_dim': self.control_dim,
            'control_params': self.control_params,
            'c': self.params['c'],
            'h': self.params['h']
        }
        return config
        
    def odes(self, t, z, control_params):
        """
        Compute the derivatives for a 2-dimensional vector [v,w] and a dictionary of parameter.
        
        Args:
            t (float): The current time. (Doesn't matter here, the system is time-independet)
            z (2-dim list or array-like): The current state of the system, where z[0] is v and z[1] is w.
            control_params (dict): A dictionary containing control parameters with possible keys:
                - 'c': Parameter c (default is self.c)
                - 'h': Parameter h (default is self.h)
        Returns:
            np.array (2dim): A np.array containing the derivatives [dvdt, dwdt]
        """

        # Extract control parameters with default values
        c= control_params.get('c', self.params['c'])
        h= control_params.get('h', self.params['h'])
        
        v, w = z
        dvdt = - (v) * (v -h) * (v+ h)
        dwdt = - w + c
        return np.array([dvdt, dwdt])


    def odes_vectorized(self, t, Z, control_params):
        """
        Compute the derivatives of the state variables v and w for a system of ODEs in a vectorized manner.
        Args:
            t(float): The current time (not used in the computation but required by the ODE solver interface).
            Z ( 1D np.array): state variables, structured as [v1, v2, ..., vn, w1, w2, ..., wn].
            control_params(dict) : 
                A dictionary containing the system parameters. Each entry is a numpy array of length n (number of samples).
                Expected keys are: 
                        'c' and 'h'  with default values in self.params
        Returns:
        
        np.array
            A 1D numpy array containing the derivatives of the state variables, structured as [dv1/dt, dv2/dt, ..., dvn/dt, dw1/dt, dw2/dt, ..., dwn/dt].
        """

        v, w = Z[:int(len(Z)/2)], Z[int(len(Z)/2):]  
        n_samples = len(v)
        
        # Extract control parameters with default values
        c = control_params.get('c', np.full(n_samples, self.params['c']))
        h = control_params.get('h', np.full(n_samples, self.params['h']))

        #ODEs
        dvdt = - (v) * (v -h) * (v + h)
        dwdt = - w + c

        
        return np.append(dvdt, dwdt)






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
        b = control_params.get('b', self.params['cb'])
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
        Compute the derivatives of the state variables x and y for a system of ODEs in a vectorized manner.
        Args:
            t(float): The current time (not used in the computation but required by the ODE solver interface).
            Z ( 1D np.array): state variables, structured as [x1, x2, ..., xn, y1, y2, ..., yn].
            control_params(dict) : 
                A dictionary containing the system parameters. Each entry is a numpy array of length n (number of samples).
                Expected keys are: 
                        '?' and '?'  with default values in self.params
        Returns:
        
        np.array
            A 1D numpy array containing the derivatives of the state variables, structured as [dx1/dt, dx2/dt, ..., dxn/dt, dy1/dt, dy2/dt, ..., dyn/dt].
        """

        x, y = Z[:int(len(Z)/2)], Z[int(len(Z)/2):]  
        n_samples = len(x)
        
        # Extract control parameters with default values
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




