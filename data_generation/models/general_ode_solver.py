import numpy as np
from scipy.integrate import solve_ivp

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
        
        control = self.create_control_array_for_derivative(np.array(control), X.shape[0])#np.array(control).reshape(X.shape[0],self.model.control_dim)
        current_params_dict = self.create_params_dict(X.shape[0])
        current_params_dict = self.update_params_dict(current_params_dict, control)
        derivative = self.model.odes_vectorized(0, X.transpose().flatten(), current_params_dict)

        return derivative.reshape(X.shape[1], X.shape[0]).transpose()
    


    def create_control_array_for_derivative(self, control, n_samples):
        '''
        Args:
            control (np.ndarray): Input control array of scalar, list or (n_samples, control_dim)
            n_samples (int): Number of samples being considered
            
        Returns:
            np.ndarray: Control array of shape (num_steps, n_samples, control_dim)
        '''
        # scalar input
        if control.ndim == 0:
            if self.model.control_dim != 1:
                raise ValueError(f"Scalar control input requires control_dim=1, got {self.model.control_dim}")
            # Create full array with scalar value
            control = np.full((n_samples, 1), control)

        # Case list input (as np.array)
        elif control.ndim == 1:
            if len(control) != self.model.control_dim:
                raise ValueError(f"1D control input length must match control_dim={self.model.control_dim}")
            # Repeat control values across all timesteps and samples
            control = control.reshape(1, self.model.control_dim)
            control = np.broadcast_to(control, (n_samples, self.model.control_dim)) 

        # check input
        else:
            required_shape = (n_samples, self.model.control_dim)
            if control.shape != required_shape:
                raise ValueError(
                    f"Control array must have shape {required_shape}, got {control.shape}"
                )      
            
        return control



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
