import numpy as np
from scipy.optimize import root

class TechnologySubstitution:
    def __init__(self, D0=1.0, delta=1.0, sigma=0.2, alpha=0.5, gamma1=1.0):
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
        
        # Solve system
        result = root(
            self.equilibrium_equations,
            initial_guess,
            args=(X, gamma2),
            # not sure which method is best
            # NOTE: there is no constraint here regarding negative values
            method='lm'
        )
        
        if not result.success:
            raise ValueError("Failed to find market equilibrium")
        
        # Reshape result and extract production rates
        solution = result.x.reshape(n_samples, 3)
        Y = solution[:, :2]  # just y1, y2
        
        # Ensure non-negative production
        return np.maximum(0, Y)
    
     
def step(X, control, dt, solver):
    """
    Generic step function working with any solver

    Parameters:
    X: array of shape (n_samples, n_dim) or (n_dim,) containing observations [x1, x2]
    gamma2: control parameter # TODO vector instead of single value
    dt: duration of timestep
    solver: Solver class connected to a model

    Return:
    Y: array of same shape as X containing observations [y1, y2]

    """
    X = np.atleast_2d(X)
    Y = solver.solve_equilibrium(X, control)
    return X + Y*dt

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
        X_next = step(X, control=gamma2, dt=1.0, solver=solver)
        print(X_next)

