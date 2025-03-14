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
        self.predictor = nn.Sequential(
            nn.Linear(num_states + control_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_states)
        )

        def forward(self, s_x, c):
            predictor_input = torch.cat([s_x, c], dim=1)
            logits = self.predictor(predictor_input)
            s_y_pred = F.softmax(logits, dim=1)
            return s_y_pred
        
class ControlGatePredictor(BasePredictor):
    """Predictor using control gate to modulate features"""
    def __init__(self, num_states, control_dim, hidden_dim):
        super().__init__(num_states, control_dim, hidden_dim)
        self.predictor_input = nn.Linear(num_states, hidden_dim)
        self.control_gate = ControlGate(hidden_dim, control_dim)
        self.predictor_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.predictor_output = nn.Linear(hidden_dim, num_states)
    
    def forward(self, s_x, c):
        features = F.relu(self.predictor_input(s_x))
        gated_features = self.control_gate(features, c)
        hidden = F.relu(self.predictor_hidden(gated_features))
        logits = self.predictor_output(hidden)
        s_y_pred = F.softmax(logits, dim=1)
        return s_y_pred

class DiscreteRepresentationsModel(nn.Module):
    def __init__(self, obs_dim=2, control_dim=1, num_states=4, hidden_dim=64, predictor_type='control_gate'):
        """
        Initialize the Discrete Representations architecture with three components:
        1. Encoder: Maps observations to state probabilities via softmax
        2. Predictor: Predicts next state probabilities based on current state and control
        3. Value Network: Maps predicted next state probabilities to a value
        
        Args:
            obs_dim: Dimension of the observations x and y
            control_dim: Dimension of the control input c
            num_states: Number of discrete states to model (number of logits)
            hidden_dim: Hidden layer size
            predictor_type: Type of predictor ('standard' or 'control_gate')
        """
        super(DiscreteRepresentationsModel, self).__init__()
        
        # Encoder network (shared for x and y)
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
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
        
        # Value network: extracts information from the predicted state probabilities
        # For tech substitution, we might want to predict market share of technology 2
        self.value_net = nn.Sequential(
            nn.Linear(num_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Predict a single value (e.g., market share)
        )
        
        self.num_states = num_states
    
    def get_state_probs(self, x):
        """
        Get the state probabilities of an observation.
        
        Args:
            x: Observation tensor
        
        Returns:
            prob_x: State probabilities (softmax of logits)
        """
        logits = self.encoder(x)
        prob_x = F.softmax(logits, dim=1)
        return prob_x
    
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
    
    def forward(self, x, c, y, v_true):
        """
        Forward pass through the complete architecture.
        
        Args:
            x: Current observation
            c: Control input
            y: Next observation
            v_true: True value computed from the environment
        
        Returns:
            s_x: Current state probabilities
            s_y: Next state probabilities
            s_y_pred: Predicted next state probabilities
            v_pred: Predicted value from predicted next state
        """
        # Encode current observation
        s_x = self.get_state_probs(x)
        
        # Encode next observation
        s_y = self.get_state_probs(y)
        
        # Predict next state probabilities
        s_y_pred = self.predict_next_state(s_x, c)
        
        # Compute value from predicted state
        v_pred = self.compute_value(s_y_pred)
        
        return s_x, s_y, s_y_pred, v_pred
