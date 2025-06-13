import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
from sqlalchemy import create_engine, select, text, MetaData
from sqlalchemy.orm import Session

import json

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.simulations.grid import tangent_transformation, logistic_transformation
from neural_networks.system_registry import SystemType, get_system_config

class BaseDataset(Dataset):
    """Base dataset class with common database functionality"""
    
    def __init__(self, db_path):
        """Initialize base dataset with database connection"""
        self.db_path = db_path
        
        # Create SQLAlchemy engine
        self.engine = create_engine(f"sqlite:///{db_path}")
        
        # Reflect database structure
        metadata = MetaData()
        metadata.reflect(bind=self.engine)
        
        # Find the data table (skip configs table)
        data_tables = [table for name, table in metadata.tables.items() 
                      if name.lower() != 'configs']
        
        if not data_tables:
            raise ValueError("No data tables found in the database (only found configs?)")
        
        # Take the first non-configs table
        self.table = data_tables[0]
        
        print(f"Using table: {self.table.name}")
        print(f"Columns: {[column.name for column in self.table.columns]}")
        
        # Count total rows
        with Session(self.engine) as session:
            self.length = session.query(self.table).count()
            print(f"Found {self.length} rows in the dataset")
    
    def __len__(self):
        return self.length
    
    def _validate_samples(self, num_samples=5):
        """Check first few samples for any issues"""
        print(f"Validating first {num_samples} samples:")
        try:
            for i in range(min(num_samples, len(self))):
                data = self[i]
                print(f"Sample {i}: {[d.numpy() if torch.is_tensor(d) else d for d in data]}")
                # Check for NaN values
                for d in data:
                    if torch.is_tensor(d) and torch.isnan(d).any():
                        print(f"WARNING: NaN values detected in sample {i}")
        except Exception as e:
            print(f"Error validating samples: {e}")


class TechSubstitutionDataset(BaseDataset):
    """Dataset for technology substitution system"""
    
    def __init__(self, db_path, value_method='market_share'):
        self.value_method = value_method
        super().__init__(db_path)
        self._validate_samples()
    
    def __getitem__(self, idx):
        required_columns = ['x0', 'x1', 'c0', 'y0', 'y1']
        for col in required_columns:
            if not hasattr(self.table.c, col):
                raise ValueError(f"Required column '{col}' not found in table")
        
        # SQLAlchemy query for this index
        with Session(self.engine) as session:
            query = select(
                self.table.c.x0, 
                self.table.c.x1, 
                self.table.c.c0, 
                self.table.c.y0, 
                self.table.c.y1
            ).offset(idx).limit(1)
            
            result = session.execute(query).fetchone()
            
        if result is None:
            raise IndexError(f"Index {idx} out of bounds")
        
        # Extract data
        x0, x1, c0, y0, y1 = result
        
        # Convert to tensors
        x = torch.tensor([x0, x1], dtype=torch.float32)
        c = torch.tensor([c0], dtype=torch.float32)
        y = torch.tensor([y0, y1], dtype=torch.float32)
        
        # Calculate v_true using the value method
        v_true = self.f_v(np.array([y0, y1]))
        
        # Check for NaN values and replace with zeros
        x = torch.nan_to_num(x, nan=0.0)
        c = torch.nan_to_num(c, nan=0.0)
        y = torch.nan_to_num(y, nan=0.0)
        v_true = torch.nan_to_num(v_true, nan=0.0)
        
        return x, c, y, v_true
    
    def f_v(self, y):
        """Calculate value function for tech substitution"""
        if isinstance(y, (tuple, list)):
            y = np.array(y)

        if y.ndim == 1:
            y1, y2 = y
        elif y.ndim == 2 and y.shape[1] == 2:
            y1 = y[:, 0]
            y2 = y[:, 1]
        else:
            raise ValueError("Input must be a tuple of two values or an array of shape (n, 2)")
        
        if self.value_method == 'market_share':
            v_true = y2 / (y1 + y2 + 1e-10)
            return torch.tensor([v_true], dtype=torch.float32)
        elif self.value_method == 'identity':
            # Get the tangent transformation function
            x0 = 3.0  # Center parameter
            alpha = 0.5  # Alpha parameter
            
            # Unpack only the forward transformation
            transform_func, _, _ = tangent_transformation(x0, alpha)
            
            # Transform each dimension
            if y.ndim == 1:
                transformed_y = np.array([transform_func(y1), transform_func(y2)])
                return torch.tensor(transformed_y, dtype=torch.float32)
            else:
                transformed_y1 = np.array([transform_func(val) for val in y1])
                transformed_y2 = np.array([transform_func(val) for val in y2])
                transformed_y = np.column_stack((transformed_y1, transformed_y2))
                return torch.tensor(transformed_y, dtype=torch.float32)
        elif self.value_method == '90% market share':
            v_true = (y2 / (y1 + y2 + 1e-10)) >= 0.9
            return torch.tensor([float(v_true)], dtype=torch.float32)
        else:
            raise NotImplementedError(f"Method '{self.value_method}' is not implemented.")


class SaddleSystemDataset(BaseDataset):
    """Dataset for saddle system"""
    
    def __init__(self, db_path, value_method='angle', num_saddles=None):
        self.value_method = value_method
        self.num_saddles = num_saddles
        super().__init__(db_path)
        
        # Infer number of saddles from data if not provided
        if self.num_saddles is None:
            self._infer_num_saddles()
        
        self._validate_samples()
    
    def _infer_num_saddles(self):
        """Infer number of saddle points from control values in the data"""
        with Session(self.engine) as session:
            query = select(self.table.c.c0).distinct()
            results = session.execute(query).fetchall()
            unique_controls = [r[0] for r in results]
            self.num_saddles = len(unique_controls)
            print(f"Inferred {self.num_saddles} saddle points from control values: {unique_controls}")
    
    def __getitem__(self, idx):
        required_columns = ['x0', 'x1', 'c0', 'y0', 'y1']
        for col in required_columns:
            if not hasattr(self.table.c, col):
                    
                raise ValueError(f"Required column '{col}' not found in table")
        # SQLAlchemy query for this index
        with Session(self.engine) as session:
            query = select(
                self.table.c.x0, 
                self.table.c.x1, 
                self.table.c.c0, 
                self.table.c.y0, 
                self.table.c.y1
            ).offset(idx).limit(1)
            
            result = session.execute(query).fetchone()
            
        if result is None:
            raise IndexError(f"Index {idx} out of bounds")
        
        # Extract data
        x0, x1, c0, y0, y1 = result

            # Validate control value
        if int(c0) >= self.num_saddles or int(c0) < 0:
            raise ValueError(f"Invalid control value {c0}. Must be in range [0, {self.num_saddles})")
        
        # Convert to tensors
        x = torch.tensor([x0, x1], dtype=torch.float32)
        
        # Convert control to one-hot encoding
        c_onehot = torch.zeros(self.num_saddles, dtype=torch.float32)
        c_onehot[int(c0)] = 1.0
        
        y = torch.tensor([y0, y1], dtype=torch.float32)
        
        # Calculate v_true using the value method
        v_true = self.f_v(np.array([y0, y1]))
        
        # Check for NaN values and replace with zeros
        x = torch.nan_to_num(x, nan=0.0)
        c_onehot = torch.nan_to_num(c_onehot, nan=0.0)
        y = torch.nan_to_num(y, nan=0.0)
        v_true = torch.nan_to_num(v_true, nan=0.0)
        
        return x, c_onehot, y, v_true
    
    def f_v(self, y):
        """Calculate value function for saddle system"""
        if isinstance(y, (tuple, list)):
            y = np.array(y)

        if y.ndim == 1:
            y1, y2 = y
        else:
            raise ValueError("Input must be a tuple or array of two values")
        
        if self.value_method == 'angle':
            # Calculate angle and convert to sine-cosine representation
            angle_rad = np.arctan2(y2, y1)
            sin_theta = np.sin(angle_rad)
            cos_theta = np.cos(angle_rad)
            return torch.tensor([sin_theta, cos_theta], dtype=torch.float32)
        elif self.value_method == 'identity':
            # Use logistic transformation
            logistic_params = {'k': 0.5, 'x_0': 0.0}
            transform_func, _, _ = logistic_transformation(logistic_params)
            
            # Transform each dimension
            transformed_y1 = transform_func(y1)
            transformed_y2 = transform_func(y2)
            return torch.tensor([transformed_y1, transformed_y2], dtype=torch.float32)
        else:
            raise NotImplementedError(f"Method '{self.value_method}' is not implemented for saddle system.")
    
    def probing_values(self, indices=None, probing_method="halfspace", **kwargs):
        """
        Calculate values for probing analysis after training.
        
        Args:
            indices: Optional list of indices to compute values for
            probing_method: Method to use for probing ('halfspace', etc.)
            **kwargs: Additional arguments specific to the probing method
            
        Returns:
            torch.Tensor: Probing values
        """
        if probing_method == "halfspace":
            return self._halfspace_probing(indices, **kwargs)
        else:
            raise NotImplementedError(f"Probing method '{probing_method}' is not implemented.")
    
    def _halfspace_probing(self, indices=None, saddle_points=None, manifold_directions=None):
        """
        Calculate halfspace values for value probing.
        
        Args:
            indices: Optional list of indices to compute values for
            saddle_points: List of saddle point coordinates
            manifold_directions: List of stable manifold directions (optional)
            
        Returns:
            torch.Tensor: Halfspace values (n_samples, n_saddles)
        """
        if saddle_points is None:
            raise ValueError("Saddle points must be provided for halfspace calculation")
        
        if indices is None:
            indices = range(len(self))
        
        halfspace_values = []
        
        for idx in indices:
            x, _, y, _ = self[idx]  # Get the data point
            
            # Use y (next state) for halfspace calculation
            point = y.numpy()
            
            # Calculate halfspace values for each saddle
            values = []
            for i, saddle_point in enumerate(saddle_points):
                # Vector from saddle_point to y
                v = point - np.array(saddle_point)
                
                if manifold_directions is not None:
                    # Use the actual stable manifold direction
                    normal = np.array([-manifold_directions[i][1], manifold_directions[i][0]])
                    # Signed distance from point to manifold
                    signed_dist = np.dot(v, normal)
                    # Assign 0 for negative halfspace, 1 for positive halfspace
                    value = 0.0 if signed_dist < 0 else 1.0
                else:
                    # Simple quadrant-based classification as fallback
                    quadrant = (v[0] > 0) * 1 + (v[1] > 0) * 2
                    value = float(quadrant)
                
                values.append(value)
            
            halfspace_values.append(values)
        
        return torch.tensor(halfspace_values, dtype=torch.float32)

def create_data_loaders(system_type, db_path, batch_size=64, val_size=1000, 
                       test_size=1000, seed=42, value_method=None):
    """
    Create training, validation, and test data loaders for specified system
    
    Args:
        system_type: Type of system ('tech_substitution' or 'saddle_system')
        db_path: Path to the SQLite database
        batch_size: Batch size for training
        val_size: Number of samples to use for validation
        test_size: Number of samples to use for final testing
        seed: Random seed for reproducibility
        value_method: Optional value method (if None, uses system default)
    
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    from neural_networks.system_registry import SystemType, get_system_config
    
    # Get system configuration
    system_config = get_system_config(SystemType[system_type.upper()])
    
    # Validate/set value_method
    if value_method is None:
        value_method = system_config['default_value_type']
        print(f"Using default value method for {system_type}: {value_method}")
    elif value_method not in system_config['value_types']:
        raise ValueError(f"Invalid value method '{value_method}' for {system_type}. "
                        f"Available: {system_config['value_types']}")
    
    
    # Create the appropriate dataset
    if system_type == 'tech_substitution':
        dataset = TechSubstitutionDataset(db_path, value_method=value_method)
    elif system_type == 'saddle_system':
        dataset = SaddleSystemDataset(db_path, value_method=value_method)
    else:
        raise ValueError(f"Unknown system type: {system_type}")
    
    # Prepare indices for training, validation, and testing
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    
    # Ensure sizes aren't too large for the dataset
    total_reserved = min(val_size + test_size, len(dataset) // 2)
    val_size = min(val_size, total_reserved // 2)
    test_size = min(test_size, total_reserved - val_size)
    
    # Split indices
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]
    
    print(f"Training set: {len(train_indices)} samples")
    print(f"Validation set: {len(val_indices)} samples")
    print(f"Test set: {len(test_indices)} samples")
    
    # Create loaders with proper seeding
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices, generator=generator)
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices, generator=generator)
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_indices, generator=generator)
    )
    
    return train_loader, val_loader, test_loader


def get_saddle_configuration(db_path):
    """
    Extract saddle points and angles from the database configuration.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        dict: Dictionary containing saddle_points and angles_degrees
              Returns None if not a saddle system or no config found
    """
    
    engine = create_engine(f"sqlite:///{db_path}")
    
    try:
        with engine.connect() as conn:
            # Query the configs table
            result = conn.execute(text("SELECT configurations FROM configs LIMIT 1"))
            config_row = result.fetchone()
            
            if config_row is None:
                print("No configuration found in configs table")
                return None
                
            # Parse the JSON configuration
            config = json.loads(config_row[0])
            
            # Extract model configuration
            solver_config = config.get('solver', {})
            model_config = solver_config.get('model', {})
            
            # Check if this is a saddle system
            if model_config.get('model') != 'MultiSaddleSystem':
                print(f"Not a saddle system: {model_config.get('model')}")
                return None
                
            # Extract saddle points and angles
            saddle_points = model_config.get('saddle_points', [])
            angles_degrees = model_config.get('angles_degrees', [])
            
            print(f"Found {len(saddle_points)} saddle points: {saddle_points}")
            print(f"Found {len(angles_degrees)} angles: {angles_degrees}")
            
            return {
                'saddle_points': saddle_points,
                'angles_degrees': angles_degrees
            }
            
    except Exception as e:
        print(f"Error extracting saddle configuration: {e}")
        return None