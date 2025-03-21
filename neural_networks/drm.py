import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sqlalchemy import create_engine, select, Table, MetaData
from sqlalchemy.orm import Session
    
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
    
class BasePredictor(nn.Module):
    """Base class for predictor implementations"""
    def __init__(self, num_states, control_dim, hidden_dim):
        super().__init__()
        self.num_states = num_states
        self.control_dim = control_dim
        self.hidden_dim = hidden_dim
    
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
    
class StandardPredictor(BasePredictor):
    """Standard predictor using concatenation of state and control"""
    def __init__(self, num_states, control_dim, hidden_dim):
        super().__init__(num_states, control_dim, hidden_dim)

        # NOTE: Using a very simple encoding of controls with same amount of nodes as num_states
        self.control_encoder = nn.Linear(control_dim, num_states)

        self.predictor = nn.Sequential(
            nn.Linear(2*num_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_states)
        )

        def forward(self, s_x, c):
            c_enc = self.control_encoder(c)
            # s_x is already encoded, no further encoding needed
            predictor_input = torch.cat([s_x, c_enc], dim=1)
            logits = self.predictor(predictor_input)
            s_y_pred = F.softmax(logits, dim=1)
            return s_y_pred
        
class BilinearPredictor(BasePredictor):
    """Predictor that directly uses encoded state with control interaction"""
    def __init__(self, num_states, control_dim, hidden_dim):
        super().__init__(num_states, control_dim, hidden_dim)
        # Only encode control - use state representation directly
        self.control_encoder = nn.Linear(control_dim, hidden_dim)
        
        # Interaction layer - captures how control affects each state dimension
        self.interaction = nn.Bilinear(num_states, hidden_dim, hidden_dim)
        
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
    def __init__(self, num_states, control_dim, hidden_dim):
        super().__init__(num_states, control_dim, hidden_dim)
        # apply control gate directly on the state embedding
        self.control_gate = ControlGate(num_states, control_dim)
        self.predictor_hidden = nn.Linear(num_states, hidden_dim)
        self.predictor_output = nn.Linear(hidden_dim, num_states)
    
    def forward(self, s_x, c):
        gated_features = self.control_gate(s_x, c)
        hidden = F.relu(self.predictor_hidden(gated_features))
        logits = self.predictor_output(hidden)
        s_y_pred = F.softmax(logits, dim=1)
        return s_y_pred

class DiscreteRepresentationsModel(nn.Module):
    def __init__(self, obs_dim=2, control_dim=1, value_dim=1, num_states=4, hidden_dim=64, 
                 predictor_type='bilinear', use_gumbel=False, initial_temp=1.0, min_temp=0.1):
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
        """
        super(DiscreteRepresentationsModel, self).__init__()
        
        # Store Gumbel softmax parameters
        self.use_gumbel = use_gumbel
        self.current_temp = initial_temp
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        
        # Rest of the initialization code remains the same
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_states)  # Logits for state probabilities
        )
        
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
        
        self.num_states = num_states
    
    def get_state_probs(self, x, training=True, hard=False):
        """
        Get the state probabilities of an observation.
        
        Args:
            x: Observation tensor
            training: Whether the model is in training mode
            hard: Whether to use hard (one-hot) assignments
        
        Returns:
            prob_x: State probabilities
        """
        logits = self.encoder(x)
        
        if self.use_gumbel and training:
            # During training with Gumbel
            gumbel_dist = torch.distributions.RelaxedOneHotCategorical(
                self.current_temp, logits=logits)
            prob_x = gumbel_dist.rsample()
        elif self.use_gumbel and hard:
            # During inference with hard assignments
            index = torch.argmax(logits, dim=1).unsqueeze(1)
            prob_x = torch.zeros_like(logits).scatter_(1, index, 1.0)
        else:
            # Regular softmax (for non-Gumbel mode or Gumbel inference without hard assignment)
            prob_x = F.softmax(logits, dim=1)
        
        return prob_x
    

    def update_temperature(self, epoch, total_epochs, annealing_proportion=0.95, delay_epochs=10):
        """Much slower temperature annealing"""
        if not self.use_gumbel:
            return
            
        # Higher starting temperature
        if epoch <= delay_epochs:
            self.current_temp = self.initial_temp
            
        # Calculate annealing factor (from 0 to 1)
        annealing_end = int(total_epochs * annealing_proportion)
        if epoch >= annealing_end:
            self.current_temp = self.min_temp
        else:
            # Use exponential decay instead of linear
            progress = (epoch - delay_epochs) / (annealing_end - delay_epochs)
            self.current_temp = self.initial_temp * (self.min_temp / self.initial_temp) ** progress
            
        return self.current_temp
    
    def forward(self, x, c, y, v_true, training=True):
        """
        Forward pass through the complete architecture.
        
        Args:
            x: Current observation
            c: Control input
            y: Next observation
            v_true: True value computed from the environment
            training: Whether the model is in training mode
        
        Returns:
            s_x: Current state probabilities
            s_y: Next state probabilities
            s_y_pred: Predicted next state probabilities
            v_pred: Predicted value from predicted next state
        """
        # Encode current observation
        s_x = self.get_state_probs(x, training=training)
        
        # Encode next observation
        s_y = self.get_state_probs(y, training=training)
        
        # Predict next state probabilities - always use soft predictions
        s_y_pred = self.predict_next_state(s_x, c)
        
        # Compute value from predicted state
        v_pred = self.compute_value(s_y_pred)
        
        return s_x, s_y, s_y_pred, v_pred
    
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
    
