import torch
import torch.nn as nn

class StableDRMLoss(nn.Module):
    def __init__(self, state_loss_weight=1.0, value_loss_weight=1.0):
        """
        Modified loss function for the Discrete Representations Model with numerical stability improvements.
        
        Args:
            state_loss_weight: Weight for the state prediction loss (D loss in diagram)
            value_loss_weight: Weight for the value prediction loss (L loss in diagram)
        """
        super(StableDRMLoss, self).__init__()
        self.state_loss_weight = state_loss_weight
        self.value_loss_weight = value_loss_weight
    
    def forward(self, s_y, s_y_pred, v_true, v_pred):
        """
        Compute the combined loss with improved numerical stability.
        
        Args:
            s_y: True next state probabilities
            s_y_pred: Predicted next state probabilities
            v_true: True value (computed from environment)
            v_pred: Predicted value
        
        Returns:
            total_loss: Weighted sum of state loss and value loss
            state_loss: KL divergence between state probabilities
            value_loss: MSE between true and predicted values
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        
        # Clamp probabilities to avoid numerical issues
        s_y = torch.clamp(s_y, epsilon, 1.0 - epsilon)
        s_y_pred = torch.clamp(s_y_pred, epsilon, 1.0 - epsilon)
        
        # Manual calculation of KL divergence for better stability
        state_loss = torch.sum(s_y * torch.log(s_y / s_y_pred), dim=1).mean()
        
        # Check for NaN in KL divergence and replace with zero if any
        if torch.isnan(state_loss):
            print("WARNING: NaN detected in KL divergence, setting to 0")
            state_loss = torch.tensor(0.0, device=s_y.device)
        
        # MSE for value loss
        value_loss = nn.functional.mse_loss(v_pred, v_true)
        
        # Combined loss
        total_loss = (
            self.state_loss_weight * state_loss + 
            self.value_loss_weight * value_loss
        )
        
        return total_loss, state_loss, value_loss