import numpy as np
from diffeqpy import de

# This tries to make DAE from diffeqpy work for the Tech Substitution model
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
     