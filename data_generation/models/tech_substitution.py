import numpy as np
from scipy.optimize import root
from diffeqpy import de #, setup

class TechnologySubstitution:
    def __init__(self, D0=1.0, delta=1.0, sigma=0.2, alpha=0.5, gamma1=1.0):
        self.D0 = D0
        self.delta = delta
        self.sigma = sigma 
        self.alpha = alpha
        self.gamma1 = gamma1
    
    @staticmethod
    def x_to_z(X):
        """
        Transform from x-space to z-space
        Parameter x: array of shape (n_samples, n_dim) or (n_dim,)
        Returns: Z array of same shape as X
        """
        # TODO 0.3 is a hyper-parameter here, has influence on our data sampling
        return np.round(X / (0.3 + X),5)
    
    @staticmethod
    def z_to_x(Z):
        """
        Transform from z-space back to x-space
        Parameter Z: array of shape (n_samples, n_dim) or (n_dim,)
        Returns: X array of same shape as Z
        """
        # TODO 0.3 is a hyper-parameter here, has influence on our data sampling
        return np.round(0.3 * Z / (1 - Z),5)
      
    
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
    
class DAESolver:
    """Solves equilibrium equations using DAE system"""
    def __init__(self, model):
        self.model = model
        # setup()  # I cannot import setup from diffeqpy
        
    
    def dae_system(self, du, u, p, t): 
    # TODO: results might be wrong, were not the same with Numerical Solver, need to test
    # TODO: this is extremely unstable, throws an [IDAS ERROR]  IDACalcIC "Newton/Linesearch algorithm failed to converge."
    # NOTE: it fails to converge even when there are solutions a numerical solver can find
    # TODO: need to scale this to work with multiple samples using de.EnsembleProblem
        """
        DAE system definition where:
        u[0], u[1]: x1, x2 (cumulative production - differential)
        u[2], u[3]: dx1/dt, dx2/dt (production rates - algebraic) 
        u[4]: p (price - algebraic)
        p[0]: gamma2 (control parameter)
        
        The system has 2 differential equations:
        - dx1/dt = y1
        - dx2/dt = y2
        
        And 3 algebraic constraints:
        - y1**0.2/x1**0.5 = price (MC1 = price)
        - gamma2 * y2**0.2/x2**0.5 = price (MC2 = price)
        - y1 + y2 = 1 - price (market clearing)
        """
        x1, x2, y1, y2, price = u
        gamma2 = p[0]

        # positive values
        x1 = max(x1, 1e-10)
        x2 = max(x2, 1e-10)
        y1 = max(y1, 1e-10)
        y2 = max(y2, 1e-10)
        
        # Differential equations
        res1 = du[0] - y1  # dx1/dt = y1
        res2 = du[1] - y2  # dx2/dt = y2
        
        # Algebraic constraints
        res3 = self.model.gamma1 * y1**self.model.sigma / x1**self.model.alpha - price  # MC1 = price
        res4 = gamma2 * y2**self.model.sigma / x2**self.model.alpha - price  # MC2 = price
        res5 = y1 + y2 - (self.model.D0 - self.model.delta * price)  # Market clearing
        
        return [res1, res2, res3, res4, res5]

        """
        Example use:
        # Initial conditions 
        X = [2.0, 1.0]
        p_guess = 0.4 # between 0 and 1
        y_total = 1.0 - p_guess
        y_guess1 = y_total * 0.3
        y_guess2 = y_total * 0.7

        u0 = [X[0], X[1], y_guess1, y_guess2, p_guess]  # States: [x1, x2, y1, y2, price]
        du0 = [y_guess1, y_guess2, 0.0, 0.0, 0.0]  # Only need derivatives for differential variables
        tspan = (0.0, 1.0)
        gamma2 = 0.5

        # Only x1 and x2 are differential variables
        differential_vars = [True, True, False, False, False]

        prob = de.DAEProblem(dae_system, du0, u0, tspan, [gamma2], differential_vars=differential_vars)
        sol = de.solve(prob)
        y1, y2 = sol.u[-1][2:4]
        """

    def solve_equilibrium(self, X, gamma2):
        """
        Solve for production rates at equilibrium given observation and control
        
        Parameters:
        X: array of shape (n_samples, 2) containing observation [x1, x2]
        gamma2: control parameter
        
        Returns:
        Y: production rates [y1, y2], NOT the next observation
        """
        # TODO: handle n_samples 
        x1, x2 = X
        param = (gamma2,)
        # Initial guesses
        price_guess = self.model.D0 / 2 
        y_total = self.model.D0 - self.model.delta * price_guess
        total_gamma = self.model.gamma1 + gamma2
        y_guess1 = y_total * (self.model.gamma1 / total_gamma)
        y_guess2 = y_total * (gamma2 / total_gamma)
        

        u0 = [x1, x2, y_guess1, y_guess2, price_guess] 
        du0 = [y_guess1, y_guess2, 0.0, 0.0, 0.0]  # Only need derivatives for differential variables
        tspan = (0.0, 1.0)
        differential_vars = [True, True, False, False, False]

        prob = de.DAEProblem(self.dae_system, du0, u0, tspan, param, differential_vars=differential_vars)

        # TODO: Create ensemble problem

        sol = de.solve(prob, dt=1.0, adaptive=False, reltol=1e-3, abstol=1e-3) # reltol for better performance
        y1, y2 = sol.u[-1][2:4]
        
        return np.array([y1, y2])

# Generic step functions    
    
def step_x_space(X, gamma2, dt, solver):
    """
    Generic step function working with any solver
    """
    X = np.atleast_2d(X)
    Y = solver.solve_equilibrium(X, gamma2)
    return X + Y*dt

def step_z_space(Z, gamma2, dt, model, solver):
    """
    Step in z-space using x-space step
    """
    X = model.z_to_x(Z)
    X_next = step_x_space(X, gamma2, dt, solver)
    return model.x_to_z(X_next)

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
        x_next = step_x_space(X, gamma2, dt=1.0, solver=solver)
        print(x_next)

