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
