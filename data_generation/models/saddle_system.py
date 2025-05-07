import numpy as np

class MultiSaddleSystem:
    """
    ODE System with k different saddle node dynamics, selectable via control parameter.
    
    Each saddle node has:
    - An equilibrium point (randomly placed or explicitly specified)
    - A stable manifold direction (randomly oriented or explicitly specified)
    - Fixed Lyapunov exponents (λ₁ > 0, λ₂ < 0) - same for all saddles
    
    Control parameter selects which saddle dynamics is currently active.
    """

    def __init__(self, k=2, saddle_points=None, angles=None, lambda1=1.0, lambda2=-1.0):
        """
        Initialize a system with k saddle nodes.
        
        Args:
            k (int): Number of saddle dynamics (default 2)
            saddle_points (list of arrays): Optional explicit saddle point coordinates
                                           If None, will generate random points in unit square
            angles (list of floats): Optional explicit angles in degrees [0-360] for each manifold
                                    If None, will generate random directions
            lambda1 (float): Positive Lyapunov exponent, used for all saddles
            lambda2 (float): Negative Lyapunov exponent, used for all saddles
        """
        # Verify Lyapunov exponents
        if lambda1 <= 0 or lambda2 >= 0:
            raise ValueError(f"Lyapunov exponents must satisfy λ₁ > 0 and λ₂ < 0, got ({lambda1}, {lambda2})")
            
        # Dimensions
        self.x_dim = 2
        self.control_dim = 1  # Control selects which saddle dynamics to use
        self.control_params = ['selected_saddle']
        self.k = k
        
        # Store Lyapunov exponents (same for all saddles)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        # Initialize arrays for saddle configurations
        self.saddle_points = []
        self.directions = []
        self.dynamics_matrices = []
        
        # Generate or assign saddle points
        if saddle_points is None:
            self.saddle_points = [np.random.rand(2) for _ in range(k)]
        else:
            if len(saddle_points) != k:
                raise ValueError(f"Must provide exactly {k} saddle points, got {len(saddle_points)}")
            self.saddle_points = [np.array(p) for p in saddle_points]
            
        # Generate or assign manifold directions
        if angles is None:
            random_angles = np.random.uniform(0, 2*np.pi, k)
            self.directions = [np.array([np.cos(angle), np.sin(angle)]) for angle in random_angles]
            self.angles_degrees = [(angle * 180 / np.pi) % 360 for angle in random_angles]
        else:
            if len(angles) != k:
                raise ValueError(f"Must provide exactly {k} angles, got {len(angles)}")
            # Store original angles in degrees
            self.angles_degrees = angles
            # Convert degrees to radians
            angles_rad = [np.radians(angle) for angle in angles]
            self.directions = [np.array([np.cos(angle), np.sin(angle)]) for angle in angles_rad]
        
        # Default parameter is the first saddle
        self.params = {
            'selected_saddle': 0
        }
        
        # Compute dynamics matrices for all saddles
        self._compute_all_dynamics_matrices()
    
    def _compute_dynamics_matrix(self, idx):
        """
        Compute the dynamics matrix A for the saddle at index idx.
        
        Args:
            idx (int): Index of the saddle to compute dynamics for
        
        Returns:
            np.array: 2x2 dynamics matrix
        """
        # Stable manifold direction = eigenvector for λ₂
        v2 = self.directions[idx]
        
        # Generate an orthogonal vector for unstable manifold
        v1 = np.array([-v2[1], v2[0]])  # Rotated 90 degrees
        
        # Create the matrix V with eigenvectors as columns
        V = np.column_stack((v1, v2))
            
        # Create diagonal matrix D with eigenvalues
        D = np.diag([self.lambda1, self.lambda2])
        
        # Compute A = V·D·V⁻¹
        return V @ D @ np.linalg.inv(V)
    
    def _compute_all_dynamics_matrices(self):
        """Compute dynamics matrices for all saddle nodes"""
        self.dynamics_matrices = [self._compute_dynamics_matrix(i) for i in range(self.k)]
    
    def get_config(self):
        """Return the model configuration"""
        config = {
            'model': self.__class__.__name__,
            'x_dim': self.x_dim,
            'control_dim': self.control_dim,
            'control_params': self.control_params,
            'k': self.k,
            'saddle_points': [point.tolist() for point in self.saddle_points],
            'angles_degrees': self.angles_degrees,
            'lambda1': self.lambda1,
            'lambda2': self.lambda2,
        }
        return config
        
    def odes(self, t, z, control_params):
        """
        Compute the derivatives for the selected saddle dynamics.
        
        Args:
            t (float): Current time (not used - autonomous system)
            z (array-like): Current state [x1, x2]
            control_params (dict): Dictionary with 'selected_saddle' specifying which dynamics to use
            
        Returns:
            np.array: Derivatives [dx1/dt, dx2/dt]
        """
        # Get selected saddle
        saddle_idx = control_params.get('selected_saddle', self.params['selected_saddle'])
        
        # Ensure valid index
        if saddle_idx < 0 or saddle_idx >= self.k:
            raise ValueError(f"Invalid saddle index: {saddle_idx}. Must be between 0 and {self.k-1}")
        
        # Get saddle point and dynamics matrix
        saddle_point = self.saddle_points[saddle_idx]
        A = self.dynamics_matrices[saddle_idx]
        
        # Compute using matrix A: dx/dt = A·(x - x*)
        return A @ (np.array(z) - saddle_point)
    
    def odes_vectorized(self, t, Z, control_params):
        """
        Vectorized computation of derivatives for multiple states.
        
        Args:
            t (float): Current time (not used)
            Z (1D np.array): Flattened states [x1_1,...,x1_n,x2_1,...,x2_n]
            control_params (dict): Dictionary containing 'selected_saddle' array
            
        Returns:
            np.array: Flattened derivatives
        """
        # Extract state components
        n_samples = len(Z) // 2
        x1 = Z[:n_samples]
        x2 = Z[n_samples:]
        
        # Get selected saddle indices for each sample
        saddle_indices = control_params.get('selected_saddle')
        
        # Ensure valid selections
        if saddle_indices is None or len(saddle_indices) != n_samples:
            raise ValueError(f"Must provide 'selected_saddle' control for all {n_samples} samples")
        
        # Compute derivatives for each state
        dx1 = np.zeros(n_samples)
        dx2 = np.zeros(n_samples)
        
        for i in range(n_samples):
            saddle_idx = int(saddle_indices[i])
            
            if saddle_idx < 0 or saddle_idx >= self.k:
                raise ValueError(f"Invalid saddle index: {saddle_idx} for sample {i}")
                
            # Get saddle point and dynamics matrix
            saddle_point = self.saddle_points[saddle_idx]
            A = self.dynamics_matrices[saddle_idx]
            
            # Calculate delta from saddle point
            delta = np.array([x1[i], x2[i]]) - saddle_point
            
            # Apply dynamics
            deriv = A @ delta
            dx1[i] = deriv[0]
            dx2[i] = deriv[1]
            
        return np.concatenate([dx1, dx2])
    
    def halfspace_values(self, x):
        """
        Return binary vector indicating halfspace membership for all saddle dynamics.
        
        Args:
            x (np.array): Point coordinates [x1, x2]
            
        Returns:
            np.array: k-dimensional binary vector [h₁, h₂, ..., hₖ] where:
                     hᵢ = 0 if x is in H0 (negative halfspace) for saddle i
                     hᵢ = 1 if x is in H1 (positive halfspace) for saddle i
        """
        # Initialize result vector
        result = np.zeros(self.k)
        
        # Calculate halfspace value for each saddle
        for i in range(self.k):
            # Vector from saddle_point to x
            v = x - self.saddle_points[i]
            
            # Normal vector to the stable manifold (perpendicular to direction)
            normal = np.array([-self.directions[i][1], self.directions[i][0]])
            
            # Signed distance from point to manifold
            signed_dist = np.dot(v, normal)
            
            # Assign 0 for negative halfspace, 1 for positive halfspace
            result[i] = 0.0 if signed_dist < 0 else 1.0
            
        return result
    
    def calculate_angle(self, point):
        """
        Calculate the angle in degrees [0-360] from the positive x1-axis to the given point.
        
        Args:
            point (np.array): Point coordinates [x1, x2]
            
        Returns:
            float: Angle in degrees [0-360]
        """
        # Extract coordinates
        x1, x2 = point
        
        # Calculate angle in radians
        angle_rad = np.arctan2(x2, x1)
        
        # Convert to degrees in [0, 360] range
        angle_deg = (angle_rad * 180 / np.pi) % 360
        
        return angle_deg