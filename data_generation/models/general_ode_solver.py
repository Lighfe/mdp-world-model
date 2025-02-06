import numpy as np
from scipy.integrate import solve_ivp



class FitzHughNagumoModel:
    def __init__(self, a = 0.7, b = 1.6, epsilon = 0.08, I = 0.35, control_params=[]):
        
        # dimensions
        self.x_dim = 2 # v, w
        self.control_dim = len(control_params) 
        self.control_params = control_params
        
        # Params
        self.a = a if 'a' not in control_params else None
        self.b = b if 'b' not in control_params else None
        self.epsilon = epsilon if 'epsilon' not in control_params else None
        self.I = I if 'I' not in control_params else None
        self.params = {'a': self.a, 'b': self.b, 'epsilon':self.epsilon, 'I':self.I}

    def get_config(self):
        config = {
            'model': self.__class__.__name__,
            'x_dim': self.x_dim,
            'control_dim': self.control_dim,
            'control_params': self.control_params,
            'a': self.a,
            'b': self.b,
            'epsilon': self.epsilon,
            'I': self.I,
        }
        return config
        
    def odes(self, t, z, control_params):
        """
        Compute the derivatives for a system of ordinary differential equations (ODEs).
        Parameters:
        t (float): The current time.
        z (list or array-like): The current state of the system, where z[0] is v and z[1] is w.
        control_params (dict): A dictionary containing control parameters with possible keys:
            - 'a': Parameter a (default is self.a)
            - 'b': Parameter b (default is self.b)
            - 'epsilon': Parameter epsilon (default is self.epsilon)
            - 'I': Parameter I (default is self.I)
        Returns:
        list: A list containing the derivatives [dvdt, dwdt], where:
            - dvdt is the derivative of v with respect to time.
            - dwdt is the derivative of w with respect to time.
        """

        # Extract control parameters with default values
        a = control_params.get('a', self.a)
        b = control_params.get('b', self.b)
        epsilon = control_params.get('epsilon',self.epsilon)
        I = control_params.get('I', self.I)
        
        v, w = z
        dvdt = v - (v**3) / 3 - w + I
        dwdt = epsilon * (v + a - b * w)
        return [dvdt, dwdt]
    
    
    def odes_vectorized(self, t, Z, control_params):
        # Z has to be provided as np.array([v1, v2, ..., vn, w1, w2, ..., wn])
        # control_params_dict as a dictionary where each entry is a scalar or a np.array of length n (number samples)
        
        v, w = Z[:int(len(Z)/2)], Z[int(len(Z)/2):]  
        n_samples = len(v)
        
        # Extract control parameters with default values
        a = control_params.get('a', np.full(n_samples, self.a))
        b = control_params.get('b', np.full(n_samples, self.b))
        epsilon = control_params.get('epsilon', np.full(n_samples, self.epsilon))
        I = control_params.get('I', np.full(n_samples, self.I))

        dvdt = v - (v**3) / 3 - w + I
        dwdt = epsilon * (v + a - b * w)

        
        return np.append(dvdt, dwdt)
    

class NumericalSolver:
    def __init__(self, model):
        self.model = model

    
    def create_params_dict(self, n_samples):
        
        params_dict = dict()

        for p in self.model.params.keys():
            if p not in self.model.control_params:
                params_dict[p] = np.full(n_samples, self.model.params[p])
            else:
                params_dict[p] = None
       
        return params_dict

    def update_params_dict(self, current_dict, controlstep):
        
        for i, p in enumerate(self.model.control_params):
            current_dict[p] = controlstep[:,i]
            
        return current_dict



    def get_derivative(self, X, control):
        """
        Return derivative for a sample of points X
        
        Args:   
        X:         array of shape (n_samples, n_dim) containing observations [x1, x2]
        control:   dictionary where each entry is a scalar or a np.array of length n (number samples)
        """
        current_params_dict = self.create_params_dict(X.shape[0])
        current_params_dict = self.update_params_dict(current_params_dict, control)
        derivative = self.model.odes_vectorized(0, X.transpose().flatten(), current_params_dict)

        return derivative.reshape(X.shape).transpose()
    
    


    def step(self, X, control, delta_t, num_steps=1, steady_control=False):
        """
        Integrate system forward using rk4

        Parameters:
        X: array of shape (n_samples, n_dim) containing observations [x1, x2]
        control: np.array of shape(num_steps, n_samples, control_dim)
        delta_t: duration of timestep
        num_steps: number of timesteps to integrate

        Return:
        Trajectory: array of shape (num_steps+1, n_samples, n_dim) containing all observations

        """

        # Create empty trajectory
        trajectory = np.zeros((num_steps + 1,) + X.shape)
        # initialize starting observation
        trajectory[0] = X
     
        current_params_dict = self.create_params_dict(X.shape[0])
        initcon = X.transpose().flatten()

        # TODO: steady_control special case (for performance)

        t_span_start = 0        
        for i in range(0, num_steps):

            current_params_dict = self.update_params_dict(current_params_dict, control[i,:,:])
            t_span = [t_span_start, t_span_start+delta_t]
            
            sol_vect = solve_ivp(self.model.odes_vectorized, t_span, initcon, args=(current_params_dict,), t_eval = t_span)
            
            trajectory[i+1] = sol_vect.y[:,-1].reshape(X.shape).transpose()

            #Update Initial Condition
            initcon = sol_vect.y[:,-1]
            t_span_start += delta_t
            
        return trajectory
