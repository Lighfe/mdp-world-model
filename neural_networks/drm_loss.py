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
    def __init__(self, state_loss_weight=1.0, value_loss_weight=1.0, 
                 use_state_diversity=False, diversity_weight=0.001,
                 use_entropy_reg=False, entropy_weight=5.0, 
                 use_entropy_decay=True, entropy_decay_proportion=0.2):
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
        # TODO set weights for batch_entropy and individual entropy
        self.current_entropy_weight = entropy_weight
        self.min_entropy_weight = 0.01 # hardcoded for now
        self.use_entropy_decay = use_entropy_decay
        self.entropy_decay_proportion = entropy_decay_proportion
    
    
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
        state_loss = F.kl_div(torch.log(s_y_pred), s_y, reduction='batchmean')
        
        # Check for NaN in KL divergence and replace with zero if any
        if torch.isnan(state_loss):
            print("WARNING: NaN detected in KL divergence, setting to 0")
            state_loss = torch.tensor(0.0, device=s_y.device)
        
        # Value loss calculation
        value_loss = self._calculate_expected_value_loss(s_y_pred, v_pred_for_all_states, v_true)
        
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
            batch_weight = 0.9 # hard coded
            individual_weight = 0.1 # hard coded
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
