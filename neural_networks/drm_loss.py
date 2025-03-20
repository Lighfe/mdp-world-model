import torch
import torch.nn as nn
import torch.nn.functional as F

class StableDRMLoss(nn.Module):
    def __init__(self, state_loss_weight=1.0, value_loss_weight=1.0, 
                 initial_diversity_weight=1.0, min_diversity_weight=0.1,
                 use_diversity_loss=True):
        """
        Modified loss function for the Discrete Representations Model with optional diversity regularization.
        
        Args:
            state_loss_weight: Weight for the state prediction loss
            value_loss_weight: Weight for the value prediction loss
            initial_diversity_weight: Starting weight for diversity regularization
            min_diversity_weight: Minimum weight for diversity regularization after decay
            use_diversity_loss: Whether to apply diversity regularization (default: True)
        """
        super(StableDRMLoss, self).__init__()
        self.state_loss_weight = state_loss_weight
        self.value_loss_weight = value_loss_weight
        self.initial_diversity_weight = initial_diversity_weight
        self.min_diversity_weight = min_diversity_weight
        self.current_diversity_weight = initial_diversity_weight
        self.use_diversity_loss = use_diversity_loss
    
    # ... rest of methods remain the same ...
    
    def forward(self, s_y, s_y_pred, v_true, v_pred, s_x=None):
        """
        Compute the combined loss with optional diversity regularization.
        
        Args:
            s_y: True next state probabilities
            s_y_pred: Predicted next state probabilities
            v_true: True value (computed from environment)
            v_pred: Predicted value
            s_x: Current state probabilities (for diversity regularization)
        
        Returns:
            total_loss: Weighted sum of all losses
            state_loss: KL divergence between state probabilities
            value_loss: MSE between true and predicted values
            div_loss: Diversity regularization loss (0 if s_x not provided or disabled)
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
        
        # MSE for value loss
        value_loss = nn.functional.mse_loss(v_pred, v_true)
        
        # Add diversity loss if s_x is provided and diversity loss is enabled
        div_loss = torch.tensor(0.0, device=s_y.device)
        if s_x is not None and self.use_diversity_loss:
            div_loss = self.diversity_loss(s_x)
        
        # Combined loss
        total_loss = (
            self.state_loss_weight * state_loss + 
            self.value_loss_weight * value_loss +
            div_loss
        )
        
        return total_loss, state_loss, value_loss, div_loss
    
    def update_diversity_weight(self, epoch, max_epochs):
        """Gradually decrease diversity weight as training progresses"""
        if not self.use_diversity_loss:
            return 0.0  # Return zero if diversity loss is disabled
            
        # Decay to minimum weight by 60% of training
        progress = min(1.0, epoch / (0.6 * max_epochs))
        self.current_diversity_weight = self.initial_diversity_weight - progress * (
            self.initial_diversity_weight - self.min_diversity_weight)
        return self.current_diversity_weight