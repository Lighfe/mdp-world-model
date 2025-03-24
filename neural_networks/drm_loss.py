import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class StableDRMLoss(nn.Module):
    def __init__(self, state_loss_weight=1.0, value_loss_weight=1.0, 
                 initial_diversity_weight=1.0, min_diversity_weight=0.1,
                 use_diversity_loss=True, 
                 use_entropy_reg=False, entropy_weight=2.0):
        """
        Modified loss function for the Discrete Representations Model with optional diversity regularization.
        
        Args:
            state_loss_weight: Weight for the state prediction loss
            value_loss_weight: Weight for the value prediction loss
            initial_diversity_weight: Starting weight for diversity regularization
            min_diversity_weight: Minimum weight for diversity regularization after decay
            use_diversity_loss: Whether to apply diversity regularization (default: True)
            use_entropy_reg: Whether to use entropy regularization to prevent state collapse
            entropy_weight: Weight for the entropy regularization term
        """
        super(StableDRMLoss, self).__init__()
        self.state_loss_weight = state_loss_weight
        self.value_loss_weight = value_loss_weight
        self.initial_diversity_weight = initial_diversity_weight
        self.min_diversity_weight = min_diversity_weight
        self.current_diversity_weight = initial_diversity_weight
        self.use_diversity_loss = use_diversity_loss
        
        # New entropy regularization parameters
        self.use_entropy_reg = use_entropy_reg
        self.entropy_weight = entropy_weight
    
    def entropy_loss(self, state_probs):
        """
        Enhanced entropy regularization that actively pushes toward higher entropy
        rather than just preventing collapse below a threshold.
        """
        # Calculate average state usage across batch
        avg_state_usage = torch.mean(state_probs, dim=0)
        
        # Add small epsilon to prevent log(0)
        eps = 1e-10
        avg_state_usage = torch.clamp(avg_state_usage, eps, 1.0)
        
        # Calculate entropy and normalize by maximum possible entropy
        num_states = avg_state_usage.size(0)
        max_entropy = torch.log(torch.tensor(float(num_states)))
        entropy = -torch.sum(avg_state_usage * torch.log(avg_state_usage))
        normalized_entropy = entropy / max_entropy
        
        # Instead of just penalizing below threshold, actively push toward maximum entropy
        # The closer to 1.0, the smaller the penalty
        entropy_loss = 1.0 - normalized_entropy
        
        return entropy_loss
    
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
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        
        # Clamp probabilities to avoid numerical issues
        s_y = torch.clamp(s_y, epsilon, 1.0 - epsilon)
        s_y_pred = torch.clamp(s_y_pred, epsilon, 1.0 - epsilon)
        
        # Manual calculation of KL divergence for better stability
        state_loss = torch.sum(s_y * (torch.log(s_y) - torch.log(s_y_pred)), dim=1).mean()
        
        # Check for NaN in KL divergence and replace with zero if any
        if torch.isnan(state_loss):
            print("WARNING: NaN detected in KL divergence, setting to 0")
            state_loss = torch.tensor(0.0, device=s_y.device)
        
        # Value loss calculation (new implementation)
        value_loss = self._calculate_expected_value_loss(s_y_pred, v_pred_for_all_states, v_true)
        
        # Add diversity loss if s_x is provided and diversity loss is enabled
        div_loss = torch.tensor(0.0, device=s_y.device)
        if s_x is not None and self.use_diversity_loss:
            div_loss = self.diversity_loss(s_x)
        
        # Add entropy regularization if enabled
        entropy_loss = torch.tensor(0.0, device=s_y.device)
        if s_x is not None and self.use_entropy_reg:
            entropy_loss = self.entropy_loss(s_x) * self.entropy_weight
        
        # Combined loss
        total_loss = (
            self.state_loss_weight * state_loss + 
            self.value_loss_weight * value_loss +
            div_loss + 
            entropy_loss
        )
        
        return total_loss, state_loss, value_loss, div_loss, entropy_loss
    
    def _calculate_expected_value_loss(self, s_y_pred, v_pred_for_all_states, v_true):
        """
        Calculate expected value loss under the predicted state distribution.
        
        Args:
            s_y_pred: Predicted next state probabilities (batch_size, num_states)
            v_pred_for_all_states: Values for all one-hot states (num_states, value_dim)
            v_true: True value (batch_size, value_dim)
            
        Returns:
            Expected value loss
        """
        # Reshape for broadcasting using unsqueeze
        v_pred_expanded = v_pred_for_all_states.unsqueeze(0)  # [1, n_states, n_values]
        v_true_expanded = v_true.unsqueeze(1)                 # [n_batch, 1, n_values]
        
        # Squared differences in each value dimension --> shape: (n_batch, n_states, n_values)
        value_losses_by_successor_and_value_direction = (v_pred_expanded - v_true_expanded)**2
        
        # Sum over all value directions, giving squared value loss for each possible successor state
        # --> shape: (n_batch, n_states)
        value_losses_by_successor = value_losses_by_successor_and_value_direction.sum(dim=2)
        
        # Expected value over all possible successor states according to predicted distribution
        # --> shape: (n_batch)
        expected_value_losses = (s_y_pred * value_losses_by_successor).sum(dim=1)
        
        # Finally, take the mean over all samples in the batch
        value_loss = expected_value_losses.mean()
        
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