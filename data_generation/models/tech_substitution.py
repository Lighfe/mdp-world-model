import numpy as np
from scipy.optimize import least_squares

class TechnologySubstitution:
    def __init__(self, D0=1.0, delta=1.0, sigma=0.2, alpha=0.5, gamma1=1.0):
        # 
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

    def equilibrium_equations(self, vars, X, gamma2):
        """
        System of equations for multiple samples
        Params:
        vars: array of shape (n_samples * 3,) containing [y1, y2, p] for each sample
        X: array of shape (n_samples, 2) containing states [x1, x2]
        gamma2: control parameter
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

        # TODO: Should gamma be a list of size n_samples, too?
        
        # Compute equations for all samples
        eq1 = self.model.gamma1 * y1**self.model.sigma / x1**self.model.alpha - p
        eq2 = gamma2 * y2**self.model.sigma / x2**self.model.alpha - p
        eq3 = y1 + y2 - (self.model.D0 - self.model.delta * p)
        
        # Stack equations back into 1D array
        return np.column_stack([eq1, eq2, eq3]).flatten()
    
    def solve_equilibrium(self, X, gamma2):
        """
        Numerical solve for production rates given observation and control
        
        Parameters:
        X: array of shape (n_samples, 2) containing observation [x1, x2]
        gamma2: control parameter
        
        Returns:
        Y: array of shape (n_samples, 2) containing production rates [y1, y2]
        """
        n_samples = X.shape[0]
        
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
            args=(X, gamma2),
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
        Generic step function working with any solver

        Parameters:
        X: array of shape (n_samples, n_dim) or (n_dim,) containing observations [x1, x2]
        gamma2: control parameter # TODO vector instead of single value
        delta_t: duration of timestep
        solver: Solver class connected to a model

        Return:
        Trajectory TODO

        """
        X = np.atleast_2d(X)
        Y = self.solve_equilibrium(X, control)

        # https://de.wikipedia.org/wiki/Klassisches_Runge-Kutta-Verfahren
        # using simple version first
        # TODO try more sophisticated scipy.integrate.solve_ivp

        trajectory = np.zeros((num_steps + 1,) + X.shape)
        trajectory[0] = X
        
        for step in range(num_steps):
            x = trajectory[step]
            
            # RK4 stages
            k1 = self.solve_equilibrium(x, control)
            k2 = self.solve_equilibrium(x + 0.5 * delta_t * k1, control)
            k3 = self.solve_equilibrium(x + 0.5 * delta_t * k2, control)
            k4 = self.solve_equilibrium(x + delta_t * k3, control)
            
            # Update
            trajectory[step + 1] = x + (delta_t / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return trajectory

def manual_test():
    """Simple check if different inputs work"""
    # TODO: Write a more sophisticated test, that also checks if the values are correct
    model = TechnologySubstitution(D0=1.0, delta=1.0, sigma=0.2, alpha=0.5, gamma1=1.0)
    solver = NumericalSolver(model)

    test_cases = [
        (np.array([[0.05, 0.01],[0.2, 0.05]]), 0.5),  # baseline case
        (np.array([[2.0, 2.0],[2.0, 1.0]]), 1.0), # different gamma 
        (np.array([2.0, 1.0]), 0.5), # (2,) dimensional
        (np.array([[2.0, 1.0],[10.0, 0.5], [20.0, 18.0]]), 0.5) # (3,2) dimensional  
    ]

    for test in test_cases:
        X, gamma2 = test
        X_next = solver.step(X, control=gamma2, delta_t=1.0)
        print(X_next)

def test_integration():
    """
    Test the integration implementation
    """
    # Initialize model and solver with realistic parameters
    model = TechnologySubstitution(D0=1.0, delta=1.0, sigma=0.2, alpha=0.5, gamma1=1.0)
    solver = NumericalSolver(model)

    # Test 1: Basic shape tests with realistic initial conditions
    X_single = np.array([[0.5, 0.3]])  # Single sample, well above zero
    X_multi = np.array([[0.5, 0.3], [0.7, 0.4], [0.2, 0.6]])  # Multiple samples
    control = 0.5
    dt = 0.1
    
    try:
        # Single step trajectories
        traj_single_step = solver.step(X_single, control, dt, num_steps=1)
        print("Single step shape:", traj_single_step.shape)
        print("Single step values:", traj_single_step)
        
        traj_multi_step = solver.step(X_multi, control, dt, num_steps=1)
        print("Multi step shape:", traj_multi_step.shape)
        
        # Multiple timesteps
        num_steps = 5
        traj_multiple = solver.step(X_single, control, dt, num_steps=num_steps)
        print("Multiple timesteps final value:", traj_multiple[-1])
        
        print("All tests passed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        raise
