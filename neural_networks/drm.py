import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sqlalchemy import create_engine, select, Table, MetaData
from sqlalchemy.orm import Session

# Import here to avoid circular imports
# Define project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from neural_networks.system_registry import SystemType, get_value_sorting_function



class DiscreteRepresentationsModel(nn.Module):
    def __init__(self, obs_dim=2, control_dim=1, value_dim=1, input_transform='none', num_states=4, hidden_dim=64, 
                 predictor_type='bilinear', use_gumbel=False, initial_temp=5.0, min_temp=0.5,
                 use_target_encoder=False, ema_decay=0.9, value_activation="sigmoid"):
        """
        Initialize the Discrete Representations architecture
        
        Args:
            obs_dim: Dimension of the observations x and y
            control_dim: Dimension of the control input c
            num_states: Number of discrete states to model (number of logits)
            hidden_dim: Hidden layer size
            predictor_type: Type of predictor ('standard', 'control_gate', or 'bilinear')
            use_gumbel: Whether to use Gumbel softmax for state encoding
            initial_temp: Initial temperature for Gumbel softmax
            min_temp: Minimum temperature for Gumbel softmax
            use_target_encoder: Whether to use a target encoder with EMA updates
            ema_decay: EMA decay rate for target encoder (higher = slower updates)
            control_format: Format of control input ('continuous' or 'categorical')
            value_activation: Activation function to use for value output
        """
        super(DiscreteRepresentationsModel, self).__init__()
        
        self.input_transform = input_transform

        # Store Gumbel softmax parameters
        self.use_gumbel = use_gumbel
        self.current_temp = initial_temp
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        # Store EMA parameters
        self.use_target_encoder = use_target_encoder
        self.ema_decay = ema_decay
        
        # initialize
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_states)  # Logits for state probabilities
        )

        # Create target encoder - initially a copy of the encoder
        if use_target_encoder:
            self.target_encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_states)
            )
            # Copy weights from encoder to target encoder
            self._copy_weights(self.encoder, self.target_encoder)
            # Disable gradient computation for target encoder
            for param in self.target_encoder.parameters():
                param.requires_grad = False
        
        # Create appropriate predictor based on type
        if predictor_type == 'standard':
            self.predictor = StandardPredictor(num_states, control_dim, hidden_dim)
        elif predictor_type == 'control_gate':
            self.predictor = ControlGatePredictor(num_states, control_dim, hidden_dim)
        elif predictor_type == 'bilinear':
            self.predictor = BilinearPredictor(num_states, control_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(num_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, value_dim)
        )

        # Set value activation function
        if value_activation == "sigmoid":
            self.value_activation = nn.Sigmoid()
        elif value_activation == "tanh":
            self.value_activation = nn.Tanh()
        else:
            self.value_activation = nn.Identity()
        
        self.num_states = num_states
        self.value_dim = value_dim

    def _copy_weights(self, src_model, tgt_model):
        """Helper method to copy weights from source to target model"""
        for src_param, tgt_param in zip(src_model.parameters(), tgt_model.parameters()):
            tgt_param.data.copy_(src_param.data)
    
    def update_target_encoder(self):
        """Update target encoder using exponential moving average"""
        if not self.use_target_encoder:
            return
            
        with torch.no_grad():
            for src_param, tgt_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                tgt_param.data.mul_(self.ema_decay).add_(src_param.data, alpha=1 - self.ema_decay)
    
    def get_embeddings_and_logits(self, x, use_target=False):
        """Get both hidden embeddings and final logits"""
        encoder_to_use = self.target_encoder if self.use_target_encoder and use_target else self.encoder
        
        # Transform raw inputs first
        x_transformed = self.transform_inputs(x)

        # Extract embeddings (everything except final layer)
        embeddings = x_transformed
        for layer in encoder_to_use[:-1]:  # All layers except the last
            embeddings = layer(embeddings)
        
        # Get final logits
        logits = encoder_to_use[-1](embeddings)
        
        return embeddings, logits
    
    def transform_inputs(self, x):
        if self.input_transform == 'log':
            return torch.log(x + 1e-10)
        elif self.input_transform == 'asinh':
            return torch.asinh(x)
        else:
            return x  # No transformation
    
    def get_state_probs(self, x, training=True, hard=False, use_target=False, soft=False):
        # Choose encoder
        encoder_to_use = self.target_encoder if self.use_target_encoder and use_target else self.encoder
        
        # Get logits
        _, logits = self.get_embeddings_and_logits(x, use_target=use_target)
        
        if hard:
            # Always use hard argmax when requested regardless of other settings
            index = torch.argmax(logits, dim=1).unsqueeze(1)
            prob_x = torch.zeros_like(logits).scatter_(1, index, 1.0)
        elif soft:
            # To visualize with normal softmax
            prob_x = F.softmax(logits, dim=1)
        elif self.use_gumbel and training and not use_target:
            # Only main encoder during training uses Gumbel noise
            prob_x = F.gumbel_softmax(logits, tau=self.current_temp, hard=False)
        else:
            # Target encoder OR validation: deterministic softmax with temperature
            if self.use_gumbel:
                # Temperature scaled softmax (no Gumbel noise)
                prob_x = F.softmax(logits / self.current_temp, dim=1)
            else:
                # Regular softmax
                prob_x = F.softmax(logits, dim=1)
        
        return prob_x
    
    def compute_value(self, s_y):
        """
        Extract value from state probabilities with appropriate activation.
        
        Args:
            s_y: State probabilities (predicted)
        
        Returns:
            val: Predicted value with activation applied
        """
        # Apply value network
        val = self.value_net(s_y)
        
        # Apply activation function
        return self.value_activation(val)
    
    def update_temperature(self, epoch, total_epochs, annealing_proportion=0.8, delay_epochs=5):
        """Much slower temperature annealing"""
        if not self.use_gumbel:
            return
            
        # Calculate annealing factor (from 0 to 1)
        annealing_end = int(total_epochs * annealing_proportion)
        # Higher starting temperature
        if epoch <= delay_epochs:
            self.current_temp = self.initial_temp
        elif epoch >= annealing_end:
            self.current_temp = self.min_temp
        else:
            # Use exponential decay instead of linear
            progress = (epoch - delay_epochs) / (annealing_end - delay_epochs)
            self.current_temp = self.initial_temp * (self.min_temp / self.initial_temp) ** progress
            
        return self.current_temp
    
    def forward(self, x, c, y, v_true, training=True):
        """Forward pass with embedding extraction for VICReg"""
        
        # Get embeddings and state probs for current observation
        embed_x, logits_x = self.get_embeddings_and_logits(x, use_target=False)
        s_x = self._logits_to_probs(logits_x, training=training, use_target=False, hard=False, soft=False)
        
        # Get embeddings and state probs for next observation  
        embed_y, logits_y = self.get_embeddings_and_logits(y, use_target=True)
        s_y = self._logits_to_probs(logits_y, training=training, use_target=True, hard=False, soft=False)
        
        # Predict next state probabilities
        s_y_pred = self.predict_next_state(s_x, c)
        
        # Compute values for all possible states
        v_pred_for_all_states = self.compute_values_for_all_states()
        
        return s_x, s_y, s_y_pred, v_pred_for_all_states, embed_x, embed_y

    def _logits_to_probs(self, logits, training, use_target, hard, soft):
        """Convert logits to probabilities using existing logic"""
        if hard:
            index = torch.argmax(logits, dim=1).unsqueeze(1)
            return torch.zeros_like(logits).scatter_(1, index, 1.0)
        elif soft:
            return F.softmax(logits, dim=1)
        elif self.use_gumbel and training and not use_target:
            return F.gumbel_softmax(logits, tau=self.current_temp, hard=False)
        else:
            if self.use_gumbel:
                return F.softmax(logits / self.current_temp, dim=1)
            else:
                return F.softmax(logits, dim=1)
    
    def predict_next_state(self, s_x, c):
        """
        Predict the next state probabilities based on current state and control.
        
        Args:
            s_x: Current state probabilities
            c: Control input
        
        Returns:
            s_y_pred: Predicted next state probabilities
        """

        return self.predictor(s_x, c)

    
    def compute_value(self, s_y):
        """
        Extract value from state probabilities.
        
        Args:
            s_y: State probabilities (predicted)
        
        Returns:
            val: Predicted value (e.g., market share)
        """
        return self.value_net(s_y)
    
    def compute_values_for_all_states(self):
        """
        Compute values for all possible one-hot encoded states.
        
        Returns:
            v_pred_for_all_states: Tensor of shape (num_states, value_dim)
        """
        one_hot_states = torch.eye(self.num_states, device=next(self.parameters()).device)
        return self.compute_value(one_hot_states)
    
    # State sorting
    def sort_states_by_value(self, system_type=None, value_method=None, sorted_indices=None, descending=True):
        """
        Sort model states based on their values to make visualizations more consistent.
        
        Args:
            system_type: Type of system (e.g., 'saddle_system'). Not needed if sorted_indices provided.
            value_method: Value method used (e.g., 'angle'). Not needed if sorted_indices provided.
            sorted_indices: Pre-computed sorting indices. If None, will compute from values.
            
        Returns:
            tuple: (model, sorted_indices) where model is modified in-place
        """
        if sorted_indices is None:
            if system_type is None or value_method is None:
                raise ValueError("Must provide either sorted_indices or both system_type and value_method")
            
            
            # Compute values for each one-hot encoded state
            one_hot_states = torch.eye(self.num_states, device=next(self.parameters()).device)
            with torch.no_grad():
                values = self.compute_value(one_hot_states)  # Shape: (num_states, value_dim)
            
            # Get sorting function and compute sort keys
            sorting_func = get_value_sorting_function(SystemType[system_type.upper()], value_method)
            sort_keys = sorting_func(values)
            
            # Create sorted indices
            sorted_indices = torch.argsort(sort_keys, descending=True).cpu()
            
            print(f"Computed state sorting based on {value_method} values:")
            for i, orig_idx in enumerate(sorted_indices):
                print(f"  New state {i} <- Original state {orig_idx.item()} (value: {sort_keys[orig_idx].item():.2f})")
        
        else:
            sorted_indices = torch.tensor(sorted_indices) if not isinstance(sorted_indices, torch.Tensor) else sorted_indices
            print(f"Using provided sorted indices: {sorted_indices.tolist()}")
        
        # Apply the reordering to model parameters
        self._reorder_state_parameters(sorted_indices)
        
        return self, sorted_indices

    def _reorder_state_parameters(self, sorted_indices):
        """
        Reorder model parameters according to sorted_indices.
        
        Args:
            sorted_indices: Tensor of shape (num_states,) with new->old state mapping
        """
        device = next(self.parameters()).device
        sorted_indices = sorted_indices.to(device)
        
        with torch.no_grad():
            # 1. Reorder encoder output layer (last layer)
            encoder_output_layer = self.encoder[-1]
            encoder_output_layer.weight.data = encoder_output_layer.weight.data[sorted_indices]
            if encoder_output_layer.bias is not None:
                encoder_output_layer.bias.data = encoder_output_layer.bias.data[sorted_indices]
            
            # 2. Reorder target encoder if it exists
            if hasattr(self, 'target_encoder') and self.target_encoder is not None:
                target_output_layer = self.target_encoder[-1]
                target_output_layer.weight.data = target_output_layer.weight.data[sorted_indices]
                if target_output_layer.bias is not None:
                    target_output_layer.bias.data = target_output_layer.bias.data[sorted_indices]
            
            # 3. Reorder predictor (type-dependent)
            if hasattr(self.predictor, '__class__'):
                predictor_class_name = self.predictor.__class__.__name__
                
                if predictor_class_name == 'StandardPredictor':
                    self._reorder_standard_predictor(sorted_indices)
                elif predictor_class_name == 'BilinearPredictor':
                    self._reorder_bilinear_predictor(sorted_indices)
                else:
                    print(f"Warning: Reordering not implemented for {predictor_class_name}")
            
            # 4. Reorder value network input layer
            value_input_layer = self.value_net[0]  # First layer takes state probabilities
            # Reorder columns (input dimensions) corresponding to state probabilities
            value_input_layer.weight.data = value_input_layer.weight.data[:, sorted_indices]

    def _reorder_standard_predictor(self, sorted_indices):
        """Reorder StandardPredictor parameters"""
        # Input layer: concatenates [state_probs, encoded_control]
        # Only reorder the first num_states columns (state part)
        predictor_input_layer = self.predictor.predictor[0]  # First layer of sequential predictor
        old_weight = predictor_input_layer.weight.data.clone()
        
        # Split weight matrix: [state_part, control_part]
        state_weight = old_weight[:, :self.num_states]  # First num_states columns
        control_weight = old_weight[:, self.num_states:]  # Remaining columns
        
        # Reorder state part, keep control part unchanged
        reordered_state_weight = state_weight[:, sorted_indices]
        predictor_input_layer.weight.data = torch.cat([reordered_state_weight, control_weight], dim=1)
        
        # Output layer: produces next state probabilities - reorder rows
        predictor_output_layer = self.predictor.predictor[-1]  # Last layer
        predictor_output_layer.weight.data = predictor_output_layer.weight.data[sorted_indices]
        if predictor_output_layer.bias is not None:
            predictor_output_layer.bias.data = predictor_output_layer.bias.data[sorted_indices]

    def _reorder_bilinear_predictor(self, sorted_indices):
        """Reorder BilinearPredictor parameters"""
        # Bilinear layer: interaction(state_probs, control_features)
        # Need to reorder the state input dimension of the bilinear layer
        bilinear_layer = self.predictor.interaction
        
        # Bilinear layer weight shape: (output_dim, input1_dim, input2_dim)
        # input1_dim = num_states (state probabilities)
        # input2_dim = hidden_dim (control features)
        old_weight = bilinear_layer.weight.data.clone()
        
        # Reorder along the state input dimension (dim=1)
        bilinear_layer.weight.data = old_weight[:, sorted_indices, :]
        
        # Output layer: produces next state probabilities - reorder rows
        output_layer = self.predictor.output
        output_layer.weight.data = output_layer.weight.data[sorted_indices]
        if output_layer.bias is not None:
            output_layer.bias.data = output_layer.bias.data[sorted_indices]
    
class BasePredictor(nn.Module):
    """Base class for predictor implementations"""
    def __init__(self, num_states, control_dim, hidden_dim, control_format="continuous"):
        super().__init__()
        self.num_states = num_states
        self.control_dim = control_dim
        self.hidden_dim = hidden_dim
        self.control_format = control_format
    
    def forward(self, s_x, c):
        """
        Predict next state representation
        Args:
            s_x: Current state probabilities (batch_size, num_states)
            c: Control input (batch_size, control_dim)
        Returns:
            s_y_pred: Predicted next state probabilities (batch_size, num_states)
        """
        raise NotImplementedError("Subclasses must implement forward")
    
    def process_control(self, c):
        """
        Process control input based on format
        Args:
            c: Control input (batch_size, control_dim)
        Returns:
            Processed control input
        """
        # For categorical controls, the dataset already provides one-hot encoding
        # So we just return the control as-is
        return c
    
class StandardPredictor(BasePredictor):
    """Standard predictor using concatenation of state and control"""
    def __init__(self, num_states, control_dim, hidden_dim, control_format="continuous"):
        super().__init__(num_states, control_dim, hidden_dim, control_format)
        
        # If control is categorical, control_dim is number of categories (saddle points)
        # self.control_encoder = nn.Linear(control_dim, num_states) #old
        self.control_encoder = nn.Sequential(
            nn.Linear(control_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_states)
            )

        self.predictor = nn.Sequential(
            nn.Linear(2*num_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_states)
        )

    def forward(self, s_x, c):
        
        # Encode control
        c_enc = self.control_encoder(c)
        
        # s_x is already encoded, no further encoding needed
        predictor_input = torch.cat([s_x, c_enc], dim=1)
        logits = self.predictor(predictor_input)
        s_y_pred = F.softmax(logits, dim=1)
        return s_y_pred


class BilinearPredictor(BasePredictor):
    """Predictor that directly uses encoded state with control interaction
    
    Note: This predictor works with both continuous and categorical controls
    since they are encoded before the bilinear operation.
    """
    def __init__(self, num_states, control_dim, hidden_dim, control_format="continuous"):
        super().__init__(num_states, control_dim, hidden_dim, control_format)
        
        # Only encode control - use state representation directly
        # self.control_encoder = nn.Linear(control_dim, num_states)
        self.control_encoder = nn.Sequential(
            nn.Linear(control_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_states)
            )
        
        # Interaction layer - captures how control affects each state dimension
        self.interaction = nn.Bilinear(num_states, num_states, hidden_dim)
        
        # Output processing
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_states)
    
    def forward(self, s_x, c):
        
        # Process control
        control_features = F.relu(self.control_encoder(c))
        
        # Directly use s_x with control features to model interaction
        interaction = self.interaction(s_x, control_features)
        
        # Further processing
        hidden = F.relu(self.hidden(interaction))
        logits = self.output(hidden)
        s_y_pred = F.softmax(logits, dim=1)
        return s_y_pred

class ControlGatePredictor(BasePredictor):
    """Predictor using control gate to modulate features"""
    def __init__(self, num_states, control_dim, hidden_dim, control_format="continuous"):
        super().__init__(num_states, control_dim, hidden_dim, control_format)
        
        # Apply control gate directly on the state embedding
        self.control_gate = ControlGate(num_states, control_dim)
        self.predictor_hidden = nn.Linear(num_states, hidden_dim)
        self.predictor_output = nn.Linear(hidden_dim, num_states)
    
    def forward(self, s_x, c):
        # Process control input
        c_processed = self.process_control(c)
        
        # Apply control gating
        gated_features = self.control_gate(s_x, c_processed)
        hidden = F.relu(self.predictor_hidden(gated_features))
        logits = self.predictor_output(hidden)
        s_y_pred = F.softmax(logits, dim=1)
        return s_y_pred

class ControlGate(nn.Module):
    # FiLM inspired
    # NOTE: might want to replace this with a region aware method
    def __init__(self, feature_dim, control_dim):
        super().__init__()
        # Scale network
        self.scale_net = nn.Linear(control_dim, feature_dim)
        # Shift network
        self.shift_net = nn.Linear(control_dim, feature_dim)
        
    def forward(self, features, control):
        # Scale range: [0, 3] to allow more amplification
        scale = 3 * torch.sigmoid(self.scale_net(control))
        # Shift range: [-1, 1] to allow bidirectional shifts
        shift = torch.tanh(self.shift_net(control))
        
        # Apply both scale and shift
        modulated_features = scale * features + shift
        
        return modulated_features


# NOTE: This function might need some tweaking for future use

def sort_existing_model(model_path, output_path=None, system_type=None, value_method=None, sorted_indices=None):
    """
    Sort states in an already trained model and save the result.
    
    Args:
        model_path: Path to saved model checkpoint
        output_path: Path to save sorted model. If None, adds '_sorted' to original name
        system_type: Type of system. Not needed if sorted_indices provided
        value_method: Value method. Not needed if sorted_indices provided  
        sorted_indices: Pre-computed sorting indices. If None, will compute from values
        
    Returns:
        tuple: (sorted_model, sorted_indices, output_path)
    """
    # Load the model
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    # Recreate model
    model = DiscreteRepresentationsModel(
        obs_dim=config['obs_dim'],
        control_dim=config['control_dim'],
        value_dim=config['value_dim'],
        num_states=config['num_states'],
        hidden_dim=config['hidden_dim'],
        # Add other necessary config parameters as needed
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Sort the states
    model, sorted_indices = model.sort_states_by_value(
        system_type=system_type,
        value_method=value_method, 
        sorted_indices=sorted_indices
    )
    
    # Determine output path
    if output_path is None:
        path_parts = model_path.split('.')
        output_path = '.'.join(path_parts[:-1]) + '_sorted.' + path_parts[-1]
    
    # Save sorted model
    sorted_checkpoint = checkpoint.copy()
    sorted_checkpoint['model_state_dict'] = model.state_dict()
    sorted_checkpoint['sorted_indices'] = sorted_indices.tolist()
    if system_type:
        sorted_checkpoint['system_type'] = system_type
    if value_method:
        sorted_checkpoint['value_method'] = value_method
    
    torch.save(sorted_checkpoint, output_path)
    print(f"Saved sorted model to: {output_path}")
    
    return model, sorted_indices, output_path