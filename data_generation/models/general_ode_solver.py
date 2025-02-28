import numpy as np
from scipy.integrate import solve_ivp



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
    

class GeneralODENumericalSolver:
    """
    General numerical integration of a system of ordinary differential equations (ODEs) 
        using the Runge-Kutta method via scipy.integrate.solve_ivp()
    """
    def __init__(self, model):
        self.model = model

        # Params
        self.rtol = 1e-3 
        self.atol = 1e-6
        self.ivp_method = 'scipy.solve_ivp(RK45)'


    def get_config(self):
        config = {
            'solver': self.__class__.__name__,
            'model': self.model.get_config(),
            'rtol': self.rtol,
            'atol': self.atol,
            'ivp_method': self.ivp_method
        }
        return config
    
    def create_params_dict(self, n_samples):
        """
        Create a dictionary of the current parameters for the model.
        For control parameters add value none, for default parameters create array of length n_samples.
        Args:
            n_samples (int): The number of samples for which each parameter vector has to be created
        Returns:
            dict: A dictionary containing the parameters and their corresponding values.
        """
        
        params_dict = dict()

        for p in self.model.params.keys():
            if p not in self.model.control_params:
                params_dict[p] = np.full(n_samples, self.model.params[p])
            else:
                params_dict[p] = None
       
        return params_dict

    def update_params_dict(self, current_dict, controlstep):
        """
        Update the params dict for a given control.
        Args:
            current_dict (dict): as in self.create_params_dict
            controlstep (2dim np.array): controlarray of shape (n_samples, control_dim), 
                                            has to be in the correct order corresponding to self.model.control_params
        """
        
        for i, p in enumerate(self.model.control_params):
            current_dict[p] = controlstep[:,i]
            
        return current_dict



    def get_derivative(self, X, control):
        """
        Return derivative for a sample of points X
        
        Args:   
        X:         array of shape (n_samples, n_dim) containing observations [x1, x2]
        control:   scalar, list or array of shape (control_dim)
        """
        
        control = np.array(control).reshape(X.shape[0],self.model.control_dim)
        current_params_dict = self.create_params_dict(X.shape[0])
        current_params_dict = self.update_params_dict(current_params_dict, control)
        derivative = self.model.odes_vectorized(0, X.transpose().flatten(), current_params_dict)

        return derivative.reshape(X.shape[1], X.shape[0]).transpose()
    
    

    def step(self, X, control, delta_t, num_steps=1, steady_control=False):
        """
        Integrate system forward using rk4

        Parameters:
        X: array of shape (n_samples, n_dim) containing observations [x1, x2]
        control: np.array of shape(num_steps, n_samples, control_dim)
        delta_t: duration of timestep
        num_steps: number of timesteps to integrate
        steady_control (bool): If the control is the same throughout the whole simulation

        Return:
        Trajectory: array of shape (num_steps+1, n_samples, n_dim) containing all observations

        """

        # Create empty trajectory
        trajectory = np.zeros((num_steps + 1,) + X.shape)
        # initialize starting observation
        trajectory[0] = X
     
        current_params_dict = self.create_params_dict(X.shape[0])
        initcon = X.flatten('F') 

        if steady_control:
            params_dict = self.update_params_dict(current_params_dict, control[0]) # dim (n_samples, control_dim), 0 arbitrary since all same
            # Time points to evaluate at
            t_eval = np.linspace(0, delta_t * num_steps, num_steps + 1)

            solution = solve_ivp(
                self.model.odes_vectorized,
                t_span = (0, delta_t * num_steps),
                y0 = initcon,
                t_eval = t_eval,
                args = (params_dict,),
                method='RK45') 
            n_samples = X.shape[0]
            n_dim = X.shape[1]         
            trajectory = solution.y.reshape(n_dim, n_samples, num_steps +1).transpose(2,1,0)    
            
        else:
            t_span_start = 0        
            for i in range(0, num_steps):

                current_params_dict = self.update_params_dict(current_params_dict, control[i,:,:])
                t_span = [t_span_start, t_span_start+delta_t]
                                
                sol_vect = solve_ivp(self.model.odes_vectorized, 
                                     t_span, 
                                     initcon, 
                                     args=(current_params_dict,), 
                                     t_eval = t_span,
                                     method = 'RK45')

                n_samples = X.shape[0]
                n_dim = X.shape[1]             
                trajectory[i+1] = sol_vect.y[:,-1].reshape(n_dim, n_samples).T #changed test it
                
                #Update Initial Condition
                initcon = sol_vect.y[:,-1]
                t_span_start += delta_t
                
        return trajectory
