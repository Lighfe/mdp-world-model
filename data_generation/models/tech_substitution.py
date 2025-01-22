import numpy as np
from scipy.optimize import least_squares

class TechnologySubstitution:
    def __init__(self, D0=1.0, delta=1.0, sigma=0.2, alpha=0.5, gamma1=1.0):
        # dimensions
        self.x_dim = 2 # x1, x2
        self.control_dim = 1 # gamma2
        # Params
        self.D0 = D0
        self.delta = delta
        self.sigma = sigma 
        self.alpha = alpha
        self.gamma1 = gamma1
    
      
    
class NumericalSolver:
    """"Root solver from scipy"""
    def __init__(self, model):
        self.model = model

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
        vars = vars.reshape(n_samples, 3)
        y1, y2, p = vars[:, 0], vars[:, 1], vars[:, 2]
        
        # Extract states
        x1, x2 = X[:, 0], X[:, 1]
        x1 = np.maximum(x1, 1e-10)
        x2 = np.maximum(x2, 1e-10)
        
        # Compute equations for all samples
        eq1 = self.model.gamma1 * y1**self.model.sigma / x1**self.model.alpha - p
        eq2 = control * y2**self.model.sigma / x2**self.model.alpha - p
        eq3 = y1 + y2 - (self.model.D0 - self.model.delta * p)
        
        # Stack equations back into 1D array
        return np.column_stack([eq1, eq2, eq3]).flatten()
    
    def solve_equilibrium(self, X, control):
        """
        Numerical solve for production rates given observation and control
        
        Parameters:
        X: array of shape (n_samples, 2) containing observation [x1, x2]
        control: control parameter gamma2
        
        Returns:
        Y: array of shape (n_samples, 2) containing production rates [y1, y2]
        """
        n_samples = X.shape[0]
        assert control.shape == (n_samples,), \
            f"Control shape {control.shape} doesn't match n_samples={n_samples}"
        
        # Initial guess for each sample
        p_guess = self.model.D0 / (2 * self.model.delta)
        y_guess = self.model.D0 / 4
        initial_guess = np.tile([y_guess, y_guess, p_guess], n_samples)
        
        # Bounds
        lb = np.tile([0.0, 0.0, 0.0], n_samples)
        ub = np.tile([np.inf, np.inf, np.inf], n_samples)
        
        result = least_squares(
            self.equilibrium_equations,
            initial_guess,
            args=(X, control),
            bounds=(lb, ub),
            method='trf',
            ftol=1e-6,
            xtol=1e-6
        )
        
        if not result.success:
            raise ValueError(f"Failed to find market equilibrium: {result.message}")
        
        solution = result.x.reshape(n_samples, 3)
        return solution[:, :2]
    
     
    def step(self, X, control, delta_t, num_steps=1):
        """
        Integrate system forward using rk4

        Parameters:
        X: array of shape (n_samples, n_dim) containing observations [x1, x2]
        control: array of shape (num_steps, n_samples) 
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

        # https://de.wikipedia.org/wiki/Klassisches_Runge-Kutta-Verfahren
        # using simple version first because it's probably most performant
        # NOTE: Might want to try more sophisticated scipy.integrate.solve_ivp
        # related possible TODO: check if the calculations are accurate enough
        
        for step in range(num_steps):
            x = trajectory[step]
            c = control[step] # shape (n, samples)
            
            # RK4 stages
            k1 = self.solve_equilibrium(x, c)
            k2 = self.solve_equilibrium(x + 0.5 * delta_t * k1, c)
            k3 = self.solve_equilibrium(x + 0.5 * delta_t * k2, c)
            k4 = self.solve_equilibrium(x + delta_t * k3, c)
            
            # Update
            trajectory[step + 1] = x + (delta_t / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return trajectory
