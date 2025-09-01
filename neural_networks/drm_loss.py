import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StableDRMLoss(nn.Module):
    def __init__(self, state_loss_weight=0.5, value_loss_weight=1.5, 
                 use_entropy_reg=True, entropy_weight=3.0, 
                 use_entropy_decay=True, entropy_decay_proportion=0.2,
                 state_loss_type="kl_div", value_loss_type="mse", value_method=None):
        """
        Loss function for the Discrete Representations Model with optional entropy regularization.
        
        Args:
            state_loss_weight: Weight for state prediction loss
            value_loss_weight: Weight for value prediction loss
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

        # Entropy regularization parameters
        self.use_entropy_reg = use_entropy_reg
        self.initial_entropy_weight = entropy_weight if self.use_entropy_reg else 0.0
        self.current_entropy_weight = entropy_weight
        self.min_entropy_weight = 0.01 # hardcoded for now
        self.use_entropy_decay = use_entropy_decay
        self.entropy_decay_proportion = entropy_decay_proportion

        # State loss type
        self.state_loss_type = state_loss_type
        if state_loss_type not in ["kl_div", "cross_entropy", "mse", "js_div"]:
            raise ValueError(f"Unsupported state loss type: {state_loss_type}. "
                           f"Choose from ['kl_div', 'cross_entropy', 'mse', 'js_div']")
            
        # Value loss type
        self.value_loss_type = value_loss_type  
        self.value_method = value_method
        if value_loss_type not in ["mse", "angular", "binary_cross_entropy"]:
            raise ValueError(f"Unsupported value loss type: {value_loss_type}. "
                           f"Choose from ['mse', 'angular', 'binary_cross_entropy']")
        
    def forward(self, s_y, s_y_pred, v_true, v_pred_for_all_states, s_x=None, 
                embed_x=None, embed_y=None, epoch=0, max_epochs=100):
        """
        Forward pass of the loss function
        
        Args:
            s_y: True state probabilities
            s_y_pred: Predicted state probabilities
            v_true: True values
            v_pred_for_all_states: Predicted values for all states
            s_x: Current state probabilities (for entropy calculation)
            epoch: Current epoch (for scheduling)
            max_epochs: Total epochs (for scheduling)
        
        Returns:
            Tuple of (total_loss, state_loss, value_loss, 0.0, entropy_loss, batch_entropy, individual_entropy)
        """

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
            s_y_pred, v_pred_for_all_states, v_true, 
            loss_type=self.value_loss_type, value_method=self.value_method
        )
        
        # Calculate batch and individual entropy metrics
        batch_entropy = torch.tensor(0.0, device=s_y.device)
        individual_entropy = torch.tensor(0.0, device=s_y.device)
        entropy_loss = torch.tensor(0.0, device=s_y.device)
        
        if s_x is not None:
            batch_entropy = self.batch_entropy(s_x)
            individual_entropy = self.individual_entropy(s_x)
            if self.use_entropy_reg:
                batch_weight = 0.8 # hard coded
                individual_weight = 0.2 # hard coded
                entropy_loss = batch_weight * (1.0 - batch_entropy) + individual_weight * individual_entropy
        
        
        # Combine losses
        total_loss = (self.state_loss_weight * state_loss + 
                     self.value_loss_weight * value_loss + 
                     self.current_entropy_weight * entropy_loss)
        
        return (total_loss, state_loss, value_loss,  
                entropy_loss, batch_entropy, individual_entropy)
    
    def update_entropy_weight(self, epoch, max_epochs):
        """Gradually decrease entropy weight as training progresses"""
        
        if not self.use_entropy_decay:
            return self.current_entropy_weight
        
        # Decay to minimum weight by entropy_decay_proportion of training
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
            Average normalized entropy across batch (0.0 to 1.0)
        """
        # Add small epsilon to prevent log(0)
        eps = 1e-10
        state_probs = torch.clamp(state_probs, eps, 1.0)
        
        # Calculate entropy for each sample in the batch
        individual_entropies = -torch.sum(state_probs * torch.log(state_probs), dim=1)
        
        # Normalize by maximum possible entropy
        num_states = state_probs.size(1)
        max_entropy = torch.log(torch.tensor(float(num_states), device=state_probs.device))
        normalized_entropies = individual_entropies / max_entropy
        
        # Return average across batch
        return torch.mean(normalized_entropies)
    
    def _kl_div_loss(self, s_y, s_y_pred):
        """KL divergence loss"""
        # KL divergence: sum(s_y * log(s_y / s_y_pred))
        kl_loss = torch.sum(s_y * torch.log(s_y / s_y_pred), dim=1)
        return torch.mean(kl_loss)
    
    def _cross_entropy_loss(self, s_y, s_y_pred):
        """Cross entropy loss"""
        # Cross entropy: -sum(s_y * log(s_y_pred))
        ce_loss = -torch.sum(s_y * torch.log(s_y_pred), dim=1)
        return torch.mean(ce_loss)
    
    def _mse_loss(self, s_y, s_y_pred):
        """Mean squared error loss"""
        mse_loss = torch.mean((s_y - s_y_pred) ** 2, dim=1)
        return torch.mean(mse_loss)
    
    def _js_div_loss(self, s_y, s_y_pred):
        """Jensen-Shannon divergence loss"""
        # JS divergence: 0.5 * (KL(s_y || M) + KL(s_y_pred || M))
        # where M = 0.5 * (s_y + s_y_pred)
        M = 0.5 * (s_y + s_y_pred)
        kl1 = torch.sum(s_y * torch.log(s_y / M), dim=1)
        kl2 = torch.sum(s_y_pred * torch.log(s_y_pred / M), dim=1)
        js_loss = 0.5 * (kl1 + kl2)
        return torch.mean(js_loss)
    
    def _calculate_expected_value_loss(self, s_y_pred, v_pred_for_all_states, v_true, loss_type="mse", value_method=None):
        """Calculate value loss based on expected value computation"""
        # Compute expected value: sum over states of (state_prob * value_for_state)
        expected_v_pred = torch.matmul(s_y_pred, v_pred_for_all_states)  # <-- REVERT TO THIS
        
        if loss_type == "mse":
            value_loss = F.mse_loss(expected_v_pred, v_true)
        elif loss_type == "angular":
            value_loss = self._circular_mse_loss(expected_v_pred, v_true)
        elif loss_type == "binary_cross_entropy":
            value_loss = self._binary_cross_entropy_loss(expected_v_pred, v_true)
        else:
            raise ValueError(f"Unknown value loss type: {loss_type}")
        return value_loss
    
    def _circular_mse_loss(self, pred, true):
        """MSE loss for sine/cosine representations"""
        # Convert back to angles
        angle_pred = torch.atan2(pred[..., 0], pred[..., 1])  # sin, cos
        angle_true = torch.atan2(true[..., 0], true[..., 1])
        
        # Circular difference
        diff = angle_pred - angle_true
        circular_diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        
        return torch.mean(circular_diff ** 2)
    
    def _binary_cross_entropy_loss(self, pred, true):
        """Binary cross entropy loss"""
        return F.binary_cross_entropy_with_logits(pred, true)
    
    