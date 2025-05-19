import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class StateDiversityLoss(nn.Module):
    """
    Loss that encourages diverse usage of states across a batch by minimizing correlations 
    between different state activations.
    """
    def __init__(self, normalize=True):
        super(StateDiversityLoss, self).__init__()
        self.normalize = normalize  # Whether to normalize state activations
        
    def forward(self, state_batch, epsilon=1e-8):
        """
        Calculate state diversity loss
        
        Args:
            state_batch: Batch of state probability distributions (batch_size, num_states)
            epsilon: Small constant for numerical stability
            
        Returns:
            Loss value - smaller means more diverse state usage
        """
        batch_size = state_batch.size(0)
        
        # Skip if batch is too small
        if batch_size <= 1:
            return torch.tensor(0.0, device=state_batch.device)
        
        # Center and standardize each state dimension across the batch
        if self.normalize:
            # Center (subtract mean)
            centered_states = state_batch - state_batch.mean(dim=0, keepdim=True)
            
            # Standardize (divide by standard deviation)
            std = torch.std(centered_states, dim=0, keepdim=True) + epsilon
            normalized_states = centered_states / std
        else:
            normalized_states = state_batch - state_batch.mean(dim=0, keepdim=True)
        
        # Calculate correlation matrix: (num_states, num_states)
        corr_matrix = torch.matmul(normalized_states.t(), normalized_states) / batch_size
        
        # Create mask to select only off-diagonal elements
        num_states = corr_matrix.size(0)
        mask = 1.0 - torch.eye(num_states, device=corr_matrix.device)
        
        # Sum squared correlations (both positive and negative hurt diversity)
        off_diag_correlations = (corr_matrix * mask).pow(2).sum()
        
        return off_diag_correlations

class StableDRMLoss(nn.Module):
    def __init__(self, state_loss_weight=0.5, value_loss_weight=1.5, 
                 use_state_diversity=False, diversity_weight=0.001,
                 use_entropy_reg=True, entropy_weight=3.0, 
                 use_entropy_decay=True, entropy_decay_proportion=0.2,
                 state_loss_type="kl_div", value_loss_type="mse"):
        """
        Loss function for the Discrete Representations Model with optional entropy and state diversity regularization.
        
        Args:
            state_loss_weight: Weight for state prediction loss
            value_loss_weight: Weight for value prediction loss
            use_state_diversity: Whether to use diversity regularization
            diversity_weight: Weight for diversity regularization
            use_entropy_reg: Whether to use entropy regularization
            entropy_weight: Initial weight for entropy regularization
            use_entropy_decay: Whether to decay entropy weight during training
            entropy_decay_proportion: Proportion of training after which entropy weight 
                                        reaches its minimum value (0.2 = 20% of training)
            state_loss_type: Type of state loss ("kl_div", "cross_entropy", "mse", "js_div")
            value_loss_type: Type of value loss ("mse", "angular", "binary_cross_entropy")
        """
        super(StableDRMLoss, self).__init__()
        self.state_loss_weight = state_loss_weight
        self.value_loss_weight = value_loss_weight

        self.diversity_weight = diversity_weight
        self.use_state_diversity = use_state_diversity
        self.diversity_weight = diversity_weight if use_state_diversity else 0.0
        self.state_diversity = StateDiversityLoss()
        
        # New entropy regularization parameters
        self.use_entropy_reg = use_entropy_reg
        self.initial_entropy_weight = entropy_weight if use_entropy_reg else 0.0
        self.current_entropy_weight = entropy_weight
        self.min_entropy_weight = 0.01 # hardcoded for now
        self.use_entropy_decay = use_entropy_decay
        self.entropy_decay_proportion = entropy_decay_proportion

        # State loss type
        self.state_loss_type = state_loss_type
        if state_loss_type not in ["kl_div", "cross_entropy", "mse", "js_div"]:
            raise ValueError(f"Unsupported state loss type: {state_loss_type}. Must be one of 'kl_div', 'mse', or 'js_div'")
        
        # Value loss type
        self.value_loss_type = value_loss_type
        if value_loss_type not in ["mse", "angular", "binary_cross_entropy"]:
            raise ValueError(f"Unsupported value loss type: {value_loss_type}. Must be one of 'mse', 'angular', or 'binary_cross_entropy'")
    
    def _kl_div_loss(self, s_y, s_y_pred):
        """KL divergence loss (original implementation)"""
        epsilon = 1e-8
        s_y = torch.clamp(s_y, epsilon, 1.0 - epsilon)
        s_y_pred = torch.clamp(s_y_pred, epsilon, 1.0 - epsilon)
        return F.kl_div(torch.log(s_y_pred), s_y, reduction='batchmean')
    
    def _cross_entropy_loss(self, s_y, s_y_pred):
        """Cross entropy loss with soft targets"""
        epsilon = 1e-8
        s_y_pred = torch.clamp(s_y_pred, epsilon, 1.0 - epsilon)
        return -torch.sum(s_y * torch.log(s_y_pred), dim=1).mean()
    
    def _mse_loss(self, s_y, s_y_pred):
        """Mean Squared Error between probability distributions"""
        return torch.mean(torch.sum((s_y - s_y_pred) ** 2, dim=1))
    
    def _js_div_loss(self, s_y, s_y_pred):
        """Jensen-Shannon divergence: 0.5 * KL(P || M) + 0.5 * KL(Q || M) where M = 0.5 * (P + Q)"""
        epsilon = 1e-8
        s_y = torch.clamp(s_y, epsilon, 1.0 - epsilon)
        s_y_pred = torch.clamp(s_y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate the mixture distribution M = 0.5 * (P + Q)
        m = 0.5 * (s_y + s_y_pred)
        m = torch.clamp(m, epsilon, 1.0 - epsilon)
        
        # Calculate KL(P || M)
        kl_p_m = torch.sum(s_y * torch.log(s_y / m), dim=1).mean()
        
        # Calculate KL(Q || M)
        kl_q_m = torch.sum(s_y_pred * torch.log(s_y_pred / m), dim=1).mean()
        
        # JS divergence is 0.5 * (KL(P || M) + KL(Q || M))
        return 0.5 * (kl_p_m + kl_q_m)
    # TODO: KS divergence turned around
    
    def _angular_loss(self, v_pred, v_true):
        """
        Angular loss for angle predictions represented as (sin(θ), cos(θ))
        
        Args:
            v_pred: Predicted values (batch_size, 2) - [sin(θ), cos(θ)]
            v_true: True values (batch_size, 2) - [sin(θ), cos(θ)]
            
        Returns:
            Angular loss value
        """
        # Ensure inputs are normalized (they should be if using tanh activation)
        # Compute dot product between predicted and true unit vectors
        dot_product = torch.sum(v_pred * v_true, dim=1)
        
        # Clamp to avoid numerical issues with arccos
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
        # Angular distance in radians
        angular_distance = torch.acos(dot_product)
        
        # You can also square it for smoother gradients
        return torch.mean(angular_distance)
    
    def _angular_mse_loss(self, v_pred, v_true):
        """
        MSE loss on the sine-cosine representation
        This is simpler and often works well for angle prediction
        
        Args:
            v_pred: Predicted values (batch_size, 2) - [sin(θ), cos(θ)]
            v_true: True values (batch_size, 2) - [sin(θ), cos(θ)]
            
        Returns:
            MSE loss on sine-cosine components
        """
        return torch.mean((v_pred - v_true) ** 2)
    
    def _binary_cross_entropy_loss(self, v_pred, v_true):
        """
        Binary cross entropy loss for binary predictions (e.g., 90% market share)
        
        Args:
            v_pred: Predicted values (batch_size, 1) - probabilities
            v_true: True values (batch_size, 1) - binary targets
            
        Returns:
            Binary cross entropy loss
        """
        epsilon = 1e-8
        v_pred = torch.clamp(v_pred, epsilon, 1.0 - epsilon)
        return F.binary_cross_entropy(v_pred, v_true)
    
    def forward(self, s_y, s_y_pred, v_true, v_pred_for_all_states, s_x=None):
        """
        Compute the combined loss with optional diversity and entropy regularization.
        
        Args:
            s_y: True next state probabilities
            s_y_pred: Predicted next state probabilities
            v_true: True value (computed from environment)
            v_pred_for_all_states: Predicted values for all one-hot encoded states (num_states, value_dim)
            s_x: Current state probabilities (for diversity/entropy regularization)
        """

        # dimension checking
        expected_dims = {
            'mse': None,  # Can handle any dimension
            'angular': 2,  # Must be 2D for sin/cos
            'binary_cross_entropy': 1  # Must be 1D
        }
        if self.value_loss_type in expected_dims and expected_dims[self.value_loss_type] is not None:
            expected_dim = expected_dims[self.value_loss_type]
            actual_dim = v_true.shape[1]
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Value loss type '{self.value_loss_type}' expects {expected_dim}D values, "
                    f"but got {actual_dim}D. Check if value_method matches value_loss_type."
                )
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        
        # Clamp probabilities to avoid numerical issues
        s_y = torch.clamp(s_y, epsilon, 1.0 - epsilon)
        s_y_pred = torch.clamp(s_y_pred, epsilon, 1.0 - epsilon)
        
        # Compute appropriate state loss based on type
        if self.state_loss_type == "kl_div":
            state_loss = self._kl_div_loss(s_y, s_y_pred)
        elif self.state_loss_type == "cross_entropy":
            state_loss = self._cross_entropy_loss(s_y, s_y_pred)
        elif self.state_loss_type == "mse":
            state_loss = self._mse_loss(s_y, s_y_pred)
        elif self.state_loss_type == "js_div":
            state_loss = self._js_div_loss(s_y, s_y_pred)
        
        # Check for NaN in state loss and replace with zero if any
        if torch.isnan(state_loss):
            print("WARNING: NaN detected in state loss, setting to 0")
            state_loss = torch.tensor(0.0, device=s_y.device)
        
        # Value loss calculation based on type
        value_loss = self._calculate_expected_value_loss(
            s_y_pred, v_pred_for_all_states, v_true, loss_type=self.value_loss_type
        )
        
        # Correlation-based diversity
        diversity_loss = torch.tensor(0.0, device=s_y.device)

        # Calculate batch and individual entropy metrics
        batch_entropy = torch.tensor(0.0, device=s_y.device)
        individual_entropy = torch.tensor(0.0, device=s_y.device)
        
        if s_x is not None:
            # always calculate for tracking
            diversity_loss = self.state_diversity(s_x)
            batch_entropy = self.batch_entropy(s_x)
            individual_entropy = self.individual_entropy(s_x)
            batch_weight = 0.8 # hard coded
            individual_weight = 0.2 # hard coded
            entropy_loss = batch_weight * (1.0 - batch_entropy) + individual_weight * individual_entropy

        
        # Combined loss
        total_loss = (
            self.state_loss_weight * state_loss + 
            self.value_loss_weight * value_loss +
            self.diversity_weight * diversity_loss + # only when enabled
            self.current_entropy_weight * entropy_loss # only when enabled
        )
        
        # Return loss components and entropy metrics
        return total_loss, state_loss, value_loss, diversity_loss, entropy_loss, batch_entropy, individual_entropy
    
    def _calculate_expected_value_loss(self, s_y_pred, v_pred_for_all_states, v_true, loss_type="mse"):
        """
        Calculate expected value loss under the predicted state distribution.
        
        Args:
            s_y_pred: Predicted next state probabilities (batch_size, num_states)
            v_pred_for_all_states: Values for all one-hot states (num_states, value_dim)
            v_true: True value (batch_size, value_dim)
            loss_type: Type of loss to use ("mse", "angular", "binary_cross_entropy")
            
        Returns:
            Expected value loss
        """
        # First, calculate the expected value under the predicted distribution
        # Expected value: sum_i P(state_i) * value(state_i)
        # Shape: (batch_size, value_dim)
        expected_v_pred = torch.matmul(s_y_pred, v_pred_for_all_states)
        
        # Now calculate the loss based on type
        if loss_type == "mse":
            # Standard MSE loss
            value_loss = torch.mean((expected_v_pred - v_true) ** 2)
        elif loss_type == "angular":
            # Use angular MSE loss for stability
            value_loss = self._angular_mse_loss(expected_v_pred, v_true)
        elif loss_type == "binary_cross_entropy":
            # Binary cross entropy for binary predictions
            value_loss = self._binary_cross_entropy_loss(expected_v_pred, v_true)
        else:
            raise ValueError(f"Unknown value loss type: {loss_type}")
        
        return value_loss
    
    def update_diversity_weight(self, epoch, max_epochs):
        """Gradually decrease diversity weight as training progresses"""
        if not self.use_diversity_loss:
            return 0.0  # Return zero if diversity loss is disabled
            
        # Decay to minimum weight by 60% of training
        progress = min(1.0, epoch / (0.6 * max_epochs))
        self.current_diversity_weight = self.initial_diversity_weight - progress * (
            self.initial_diversity_weight - self.min_diversity_weight)
        return self.current_diversity_weight
    
    def update_entropy_weight(self, epoch, max_epochs):
        """Gradually decrease entropy weight as training progresses"""
        
        if not self.use_entropy_decay:
            return self.current_entropy_weight # Don't think this if will be needed
        
        # Decay to minimum weight by 20% of training
        progress = min(1.0, epoch / (self.entropy_decay_proportion * max_epochs))
        self.current_entropy_weight = self.initial_entropy_weight - progress * (
            self.initial_entropy_weight - self.min_entropy_weight)
        return self.current_entropy_weight
    
    def batch_entropy(self, state_probs):
        """
        Calculate normalized entropy of average state usage across batch.
        Higher values (closer to 1.0) indicate more uniform state usage across the batch.
        
        Args:
            state_probs: Batch of state probability distributions (batch_size, num_states)
        
        Returns:
            Normalized entropy (0.0 to 1.0)
        """
        # Calculate average state usage across batch
        avg_state_usage = torch.mean(state_probs, dim=0)
        
        # Add small epsilon to prevent log(0)
        eps = 1e-10
        avg_state_usage = torch.clamp(avg_state_usage, eps, 1.0)
        
        # Calculate entropy and normalize by maximum possible entropy
        num_states = avg_state_usage.size(0)
        max_entropy = torch.log(torch.tensor(float(num_states), device=state_probs.device))
        entropy = -torch.sum(avg_state_usage * torch.log(avg_state_usage))
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def individual_entropy(self, state_probs):
        """
        Calculate normalized entropy for each individual state probability distribution,
        then average across the batch. Lower values indicate more discrete/peaky distributions.
        
        Args:
            state_probs: Batch of state probability distributions (batch_size, num_states)
        
        Returns:
            Average normalized entropy across samples (0.0 to 1.0)
        """
        eps = 1e-10
        state_probs = torch.clamp(state_probs, eps, 1.0)
        
        # Calculate entropy for each sample
        num_states = state_probs.size(1)
        max_entropy = torch.log(torch.tensor(float(num_states), device=state_probs.device))
        
        # Calculate entropy for each sample: -sum(p_i * log(p_i))
        sample_entropies = -torch.sum(state_probs * torch.log(state_probs), dim=1)
        
        # Normalize by maximum possible entropy
        normalized_entropies = sample_entropies / max_entropy
        
        # Return average across batch
        return torch.mean(normalized_entropies)
