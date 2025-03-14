import numpy as np
from scipy.optimize import least_squares
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp

class TechnologySubstitution:
    def __init__(self, D0=1.0, delta=1.0, sigma=0.2, alpha=0.5, gamma1=1.0):
        # dimensions
        self.x_dim = 2 # x1, x2
        self.control_dim = 1 # gamma2
        # fixed params
        self.D0 = D0
        self.delta = delta
        self.sigma = sigma 
        self.alpha = alpha
        self.gamma1 = gamma1
        self.control_params = ['gamma2']

    def get_config(self):
        # TODO add git commit number
        config = {
            'model': self.__class__.__name__,
            'x_dim': self.x_dim,
            'control_dim': self.control_dim,
            'control_params': self.control_params,
            'D0': self.D0,
            'delta': self.delta,
            'sigma': self.sigma,
            'alpha': self.alpha,
            'gamma1': self.gamma1
        }

        return config
      
    
class TechSubNumericalSolver:
    """"Root solver from scipy"""
    def __init__(self, model):
        self.model = model
        # Params
        self.ls_method = 'trf'
        self.ftol = 1e-6
        self.xtol = 1e-6
        self.ivp_method = 'RK4' # not using scipy currently

    def get_config(self):
        config = {
            'solver': self.__class__.__name__,
            'model': self.model.get_config(),
            'ls_method': 'trf',
            'ftol': self.ftol,
            'xtol': self.xtol,
            'ivp_method': self.ivp_method
        }
        return config

    def equilibrium_equations(self, vars, X, control):
        """
        System of equations for multiple samples
        Params:
        vars: array of shape (n_samples * 3,) containing [y1, y2, p] for each sample
        X: array of shape (n_samples, 2) containing states [x1, x2]
        control: array of shape (n_samples,) containing control values gamma2
        Returns:
        Array of constraint equations, shape (n_samples * 3,)
        """
        n_samples = X.shape[0]
        
        # Reshape vars into y1, y2, p arrays
        y1, y2, p = vars.reshape(n_samples, 3).T
        
        # Extract observations
        x1, x2 = X[:, 0], X[:, 1]
        x1 = np.maximum(x1, 1e-10)
        x2 = np.maximum(x2, 1e-10)

        # extract gamma2
        gamma2 = control[0]
        
        # Compute equations for all samples
        eq = np.empty(n_samples * 3)
        eq[0::3] = self.model.gamma1 * y1**self.model.sigma / x1**self.model.alpha - p
        eq[1::3] = control * y2**self.model.sigma / x2**self.model.alpha - p
        eq[2::3] = y1 + y2 - (self.model.D0 - self.model.delta * p)
        
        return eq
    
    def solve_single(self, i, x, c, initial_guess):
        """
        Solve equilibrium for a single sample
        
        Args:
            i: Sample index
            x: Single sample state of shape (1, x_dim)
            c: Single sample control of shape (control_dim,)
            initial_guess: Initial guess for solver
        """

        result = least_squares(
            lambda vars: self.equilibrium_equations(vars, x, c),
            initial_guess,
            bounds=([0,0,0], [np.inf,np.inf,np.inf]),
            method=self.ls_method,
            ftol=self.ftol,
            xtol=self.xtol
        )
        
        if not result.success:
            raise ValueError(f"Failed to find market equilibrium: {result.message}")
        
        return result.x[:2]
    

    def solve_equilibrium(self, X, control):
        """
        Numerical solve for production rates given observation and control
        
        Args:
            X: array of shape (n_samples, 2) containing observation [x1, x2]
            control: array of shape (n_samples, control_dim)
        
        Returns:
            Y: array of shape (n_samples, 2) containing production rates [y1, y2]
        """
        n_samples = X.shape[0]
        assert control.shape == (n_samples, self.model.control_dim), \
            f"Control shape {control.shape} doesn't match (n_samples={n_samples}, control_dim={self.model.control_dim})"
    
        results = np.zeros((n_samples, 2))   

        # Initial guess for each sample
        p_guess = self.model.D0 / (2 * self.model.delta)
        y_guess = self.model.D0 / 4
        initial_guess = np.array([y_guess, y_guess, p_guess])
        
        # Use parallel processing
        #with Pool() as pool:
        #    results = pool.map(self.solve_single, range(n_samples))

        # CPU paralllization using joblib
        results = Parallel(n_jobs=-1)(
            delayed(self.solve_single)(i, X[i:i+1], control[i], initial_guess)
            for i in range(n_samples)
        )

        return np.array(results)
    

    def get_derivative(self, X, control):
        # NOTE: in this case this is exactly the same as solve_equilibrium, but not necessarily in other models
        """
        Return derivative for a sample of points X
        
        Args:   
        X:         array of shape (n_samples, n_dim) containing observations [x1, x2]
        control:   scalar, list or array of shape (control_dim)
        """
        
        control = self.create_control_array_for_derivative(np.array(control), X.shape[0])#np.array(control).reshape(X.shape[0],self.model.control_dim)
        derivative = self.solve_equilibrium(X, control)

        return derivative
    

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
        control: array of shape (num_steps, n_samples, 1) 
        delta_t: duration of timestep
        num_steps: number of timesteps to integrate

        Return:
        Trajectory: array of shape (num_steps+1, n_samples, n_dim) containing all observations

        """
        # NOTE: The Simulator needs to take care that all inputs come in the correct format
        # possible TODO: step function can assume correct input, but maybe should add some asserts anyway

        # Create empty trajectory
        trajectory = np.zeros((num_steps + 1,) + X.shape)
        # initialize starting observation
        trajectory[0] = X
        n_samples = X.shape[0]

        if steady_control: 
            control = control[0] # dim (n_samples, control_dim), 0 arbitrary since all same
            # Time points to evaluate at
            t_eval = np.linspace(0, delta_t * num_steps, num_steps + 1)

            # Create wrapper function for solve_ivp
            def dxdt(t, x_flat, control):
                # Reshape flat array back to (n_samples, 2) for solve_equilibrium
                x_2d = x_flat.reshape(n_samples, 2)
                dx = self.solve_equilibrium(x_2d, control)
                return dx.flatten()  # Return flattened for solve_ivp
            
            # Solve IVP
            solution = solve_ivp(
                dxdt,
                t_span=(0, delta_t * num_steps),
                y0=X.flatten(), # y is solve_ivp naming, NOT y from equations
                t_eval=t_eval, 
                args=(control,), 
                method='RK45'
                # rtol=1e-6, # @Karolin, what accuracy is needed here and in general? Maybe we should talk about this with Jobst, I guess it depends on the problem
                # atol=1e-6 # @Karolin, what accuracy is needed here and in general?
            )
            
            # Reshape solution to match expected output format
            trajectory = solution.y.T.reshape(num_steps + 1, -1, 2)

        else:
            # for time-varying controls I'm using less accurate integration method
            # https://de.wikipedia.org/wiki/Klassisches_Runge-Kutta-Verfahren
            # Didn't manage yet to make scipy implementation faster than this:
            for step in range(num_steps):
                x = trajectory[step]
                c = control[step] # shape (n_samples, control_dim)
                
                # RK4 stages
                k1 = self.solve_equilibrium(x, c)
                k2 = self.solve_equilibrium(x + 0.5 * delta_t * k1, c)
                k3 = self.solve_equilibrium(x + 0.5 * delta_t * k2, c)
                k4 = self.solve_equilibrium(x + delta_t * k3, c)
                
                # Update
                trajectory[step + 1] = x + (delta_t / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return trajectory

    def f_v(self, y):
        # returns market share of technology 2
        # NOTE: "real" market share would be how much is produced at one time step (derivative). But x or y does not have this information, lacks c and delta_t.
            # If y is a tuple or list, convert to an array for consistent handling
        if isinstance(y, (tuple, list)):
            y = np.array(y)

        # Now y is a NumPy array.
        if y.ndim == 1:
            # Expecting a single pair: (y1, y2)
            y1, y2 = y
        elif y.ndim == 2 and y.shape[1] == 2:
            # Expecting an array of shape (n, 2)
            y1 = y[:, 0]
            y2 = y[:, 1]
        else:
            raise ValueError("Input must be a tuple of two values or an array of shape (n, 2)")
        return y2 / (y1+y2 +1e-10)