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
    
    def __init__(self, db_path, cache_data=True):
        """Initialize base dataset with database connection"""
        self.db_path = db_path
        self.cache_data = cache_data
        self.cached_data = None
        
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
        
        # Cache all data at startup if requested
        if self.cache_data:
            self._cache_all_data()
    
    def _cache_all_data(self):
        """Load all data into memory at startup"""
        print("Loading all data into memory...")
        
        with Session(self.engine) as session:
            # Get all data in one query
            query = select(self.table)
            result = session.execute(query).fetchall()
            
        self.cached_data = result
        print(f"Cached {len(self.cached_data)} samples in memory")
        
        # Close database connection since we don't need it anymore
        self.engine.dispose()

    def _verify_cache_indexing(self):
        """Verify that cached data matches original database indexing"""
        # Test first 3 and last 3 indices to verify order is preserved
        test_indices = [0, 1, 2, len(self.cached_data)-3, len(self.cached_data)-2, len(self.cached_data)-1]
        
        print("Testing cache indexing consistency:")
        for idx in test_indices:
            if idx < 0 or idx >= len(self.cached_data):
                continue
                
            # Get from cache
            cached_row = self.cached_data[idx]
            
            # Get from database using original method
            with Session(self.engine) as session:
                query = select(self.table).offset(idx).limit(1)
                db_row = session.execute(query).fetchone()
            
            # Compare first few columns
            cache_vals = [getattr(cached_row, col.name) for col in list(self.table.columns)[:3]]
            db_vals = list(db_row)[:3]
            
            if cache_vals != db_vals:
                raise ValueError(f"Cache indexing mismatch at index {idx}! "
                               f"Cache: {cache_vals}, DB: {db_vals}")
        
        print("✓ Cache indexing verified - matches original database order")
    
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
    
    def __init__(self, db_path, value_method='market_share', cache_data=True):
        self.value_method = value_method
        super().__init__(db_path, cache_data=cache_data)
        self._validate_samples()
    
    def __getitem__(self, idx):
        if self.cached_data:
            # Use cached data - no database query!
            result = self.cached_data[idx]
            
            # Extract data directly from cached result
            x0, x1, c0, y0, y1 = result.x0, result.x1, result.c0, result.y0, result.y1
            
        else:
            # Fallback to original database query method
            required_columns = ['x0', 'x1', 'c0', 'y0', 'y1']
            for col in required_columns:
                if not hasattr(self.table.c, col):
                    raise ValueError(f"Required column '{col}' not found in table")
            
            with Session(self.engine) as session:
                query = select(
                    self.table.c.x0, self.table.c.x1, self.table.c.c0, 
                    self.table.c.y0, self.table.c.y1
                ).offset(idx).limit(1)
                
                result = session.execute(query).fetchone()
                
            if result is None:
                raise IndexError(f"Index {idx} out of bounds")
            
            x0, x1, c0, y0, y1 = result
        
        # Convert to tensors (keep existing logic)
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
        """Calculate value function for tech substitution system"""
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
    
    def __init__(self, db_path, value_method='angle', num_saddles=None, cache_data=True, cache_verification=True):
        self.value_method = value_method
        self.num_saddles = num_saddles
        super().__init__(db_path, cache_data=cache_data, cache_verification=cache_verification)
        
        # Infer number of saddles from data if not provided
        if self.num_saddles is None:
            self._infer_num_saddles()

        # After inferring num_saddles, validate against system config
        self.saddle_config = get_saddle_configuration(db_path, verbose=False)
        if self.saddle_config:
            expected_saddles = len(self.saddle_config['saddle_points'])
            if self.num_saddles != expected_saddles:
                print(f"WARNING: Inferred {self.num_saddles} saddles from data, "
                    f"but system config shows {expected_saddles} saddle points!")
        
        self._validate_samples()
    
    def _infer_num_saddles(self):
        """Infer number of saddle points from control values in the data"""
        if self.cached_data:
            # Use cached data to infer saddles
            unique_controls = set(row.c0 for row in self.cached_data)
            self.num_saddles = len(unique_controls)
            print(f"Inferred {self.num_saddles} saddle points from cached control values: {sorted(unique_controls)}")
        else:
            # Original database query method
            with Session(self.engine) as session:
                query = select(self.table.c.c0).distinct()
                results = session.execute(query).fetchall()
                unique_controls = [r[0] for r in results]
                self.num_saddles = len(unique_controls)
                print(f"Inferred {self.num_saddles} saddle points from control values: {unique_controls}")
    
    def __getitem__(self, idx):
        if self.cached_data:
            # Use cached data - no database query!
            result = self.cached_data[idx]
            
            # Extract data directly from cached result
            x0, x1, c0, y0, y1 = result.x0, result.x1, result.c0, result.y0, result.y1
            
        else:
            # Fallback to original database query method
            required_columns = ['x0', 'x1', 'c0', 'y0', 'y1']
            for col in required_columns:
                if not hasattr(self.table.c, col):
                    raise ValueError(f"Required column '{col}' not found in table")
            
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
            
            x0, x1, c0, y0, y1 = result

        # Validate control value
        if int(c0) >= self.num_saddles or int(c0) < 0:
            raise ValueError(f"Invalid control value {c0}. Must be in range [0, {self.num_saddles})")
        
        # Convert to tensors (keep existing logic)
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
        
    def _halfspace_for_single_point(self, x):
        """Calculate halfspace values for a single point tensor"""
        point = x.numpy()
        
        # Use stored config instead of reloading from database
        saddle_points = self.saddle_config['saddle_points']
        angles_degrees = self.saddle_config['angles_degrees']
        
        values = []
        for saddle_point, angle_deg in zip(saddle_points, angles_degrees):
            v = point - np.array(saddle_point)
            angle_rad = np.radians(angle_deg)
            manifold_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            normal = np.array([-manifold_dir[1], manifold_dir[0]])
            signed_dist = np.dot(v, normal)
            values.append(0.0 if signed_dist < 0 else 1.0)
        
        return torch.tensor(values, dtype=torch.float32)
    
    def _halfspace_for_batch(self, x_batch):
        """Calculate halfspace values for a batch of points"""
        points = x_batch.numpy()  # (batch_size, 2)
        
        # Use stored config
        saddle_points = self.saddle_config['saddle_points']
        angles_degrees = self.saddle_config['angles_degrees']
        
        batch_size = points.shape[0]
        num_saddles = len(saddle_points)
        values = np.zeros((batch_size, num_saddles))
        
        for saddle_idx, (saddle_point, angle_deg) in enumerate(zip(saddle_points, angles_degrees)):
            # Vectorized calculation
            v = points - np.array(saddle_point)  # (batch_size, 2)
            angle_rad = np.radians(angle_deg)
            manifold_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            normal = np.array([-manifold_dir[1], manifold_dir[0]])
            signed_dist = np.dot(v, normal)  # (batch_size,)
            values[:, saddle_idx] = (signed_dist >= 0).astype(float)
        
        return torch.tensor(values, dtype=torch.float32)

class SocialTippingDataset(BaseDataset):
    """Dataset for social tipping system"""
    
    def __init__(self, db_path, value_method='abs_distance', cache_data=True):
        self.value_method = value_method
        super().__init__(db_path, cache_data=cache_data)
        self._validate_samples()
    
    def __getitem__(self, idx):
        if self.cached_data:
            # Use cached data - no database query!
            result = self.cached_data[idx]
            
            # Extract data directly from cached result
            x0, x1, c0, c1, c2, c3, y0, y1 = (result.x0, result.x1, result.c0, 
                                              result.c1, result.c2, result.c3, 
                                              result.y0, result.y1)
            
        else:
            # Fallback to original database query method
            required_columns = ['x0', 'x1', 'c0', 'c1', 'c2', 'c3', 'y0', 'y1']
            for col in required_columns:
                if not hasattr(self.table.c, col):
                    raise ValueError(f"Required column '{col}' not found in table")
            
            with Session(self.engine) as session:
                query = select(
                    self.table.c.x0, self.table.c.x1, 
                    self.table.c.c0, self.table.c.c1, self.table.c.c2, self.table.c.c3,
                    self.table.c.y0, self.table.c.y1
                ).offset(idx).limit(1)
                
                result = session.execute(query).fetchone()
                
            if result is None:
                raise IndexError(f"Index {idx} out of bounds")
            
            x0, x1, c0, c1, c2, c3, y0, y1 = result
        
        # Convert to tensors (keep existing logic)
        x = torch.tensor([x0, x1], dtype=torch.float32)
        c = torch.tensor([c0, c1, c2, c3], dtype=torch.float32)
        y = torch.tensor([y0, y1], dtype=torch.float32)
        
        # Calculate v_true using next state y (to be consistent with naming)
        v_true = self.f_v(np.array([y0, y1]))
        
        # Check for NaN values and replace with zeros
        x = torch.nan_to_num(x, nan=0.0)
        c = torch.nan_to_num(c, nan=0.0)
        y = torch.nan_to_num(y, nan=0.0)
        v_true = torch.nan_to_num(v_true, nan=0.0)
        
        return x, c, y, v_true
    
    def f_v(self, state):
        """Calculate value function for social tipping system"""
        if isinstance(state, (tuple, list)):
            state = np.array(state)

        if state.ndim == 1:
            x0, x1 = state
        else:
            raise ValueError("Input must be a tuple or array of two values")
        
        if self.value_method == 'abs_distance':
            # Absolute distance between x0 and x1 (first and second dimensions)
            distance = abs(x0 - x1)
            return torch.tensor([distance], dtype=torch.float32)
        elif self.value_method == 'identity':
            # Return the 2D state as-is (no transformation)
            return torch.tensor([x0, x1], dtype=torch.float32)
        else:
            raise NotImplementedError(f"Method '{self.value_method}' is not implemented for social tipping system.")



def create_data_loaders(system_type, db_path, batch_size=64, val_size=1000, 
                       test_size=1000, probing_size=None, seed=42, value_method=None,
                       num_workers=1):
    """
    Create training, validation, test, and optionally probing data loaders
    
    Args:
        system_type: Type of system ('tech_substitution', 'saddle_system', or 'social_tipping')
        db_path: Path to the SQLite database
        batch_size: Batch size for training
        val_size: Number of samples to use for validation
        test_size: Number of samples to use for final testing
        probing_size: Number of samples to reserve for probing (if None, no probing loader)
        seed: Random seed for reproducibility
        value_method: Optional value method (if None, uses system default)
    
    Returns:
        If probing_size is None: train_loader, val_loader, test_loader
        If probing_size is not None: train_loader, val_loader, test_loader, probing_loader
    """
    from neural_networks.system_registry import SystemType, get_system_config
    
    # Get system configuration
    system_config = get_system_config(SystemType[system_type.upper()])
    
    # Validate/set value_method
    if value_method is None:
        value_method = system_config['default_value_method']
        print(f"Using default value method for {system_type}: {value_method}")
    elif value_method not in system_config['value_methods']:
        raise ValueError(f"Invalid value method '{value_method}' for {system_type}. "
                        f"Available: {system_config['value_methods']}")
    
    # Create the appropriate dataset
    if system_type == 'tech_substitution':
        dataset = TechSubstitutionDataset(db_path, value_method=value_method)
    elif system_type == 'saddle_system':
        dataset = SaddleSystemDataset(db_path, value_method=value_method)
    elif system_type == 'social_tipping':
        dataset = SocialTippingDataset(db_path, value_method=value_method)
    else:
        raise ValueError(f"Unknown system type: {system_type}")
    
    # Prepare indices for splitting
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    
    if probing_size is not None:
        # Reserve probing data first (never seen during training)
        probing_indices = indices[:probing_size]
        remaining_indices = indices[probing_size:]
        
        # Ensure sizes aren't too large for remaining data
        remaining_size = len(remaining_indices)
        total_reserved = min(val_size + test_size, remaining_size // 2)
        val_size = min(val_size, total_reserved // 2)
        test_size = min(test_size, total_reserved - val_size)
        
        # Split remaining indices
        test_indices = remaining_indices[:test_size]
        val_indices = remaining_indices[test_size:test_size + val_size]
        train_indices = remaining_indices[test_size + val_size:]
        
        print(f"Probing set: {len(probing_indices)} samples")
    else:
        # Original logic without probing
        total_reserved = min(val_size + test_size, len(dataset) // 2)
        val_size = min(val_size, total_reserved // 2)
        test_size = min(test_size, total_reserved - val_size)
        
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
        sampler=SubsetRandomSampler(train_indices, generator=generator),
        num_workers=num_workers,  
        pin_memory=True,  # For GPU transfer efficiency
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices, generator=generator),
        num_workers=num_workers,  
        pin_memory=True,  # For GPU transfer efficiency
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_indices, generator=generator),
        num_workers=num_workers,  
        pin_memory=True,  # For GPU transfer efficiency
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive
    )
    
    if probing_size is not None:
        probing_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(probing_indices, generator=generator),
            num_workers=num_workers,  
            pin_memory=True,  # For GPU transfer efficiency
            persistent_workers=True if num_workers > 0 else False  # Keep workers alive
        )
        return train_loader, val_loader, test_loader, probing_loader
    else:
        return train_loader, val_loader, test_loader


def get_saddle_configuration(db_path, verbose=True):
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
            
            if verbose:
                print(f"Found {len(saddle_points)} saddle points: {saddle_points}")
                print(f"Found {len(angles_degrees)} angles: {angles_degrees}")
            
            return {
                'saddle_points': saddle_points,
                'angles_degrees': angles_degrees
            }
            
    except Exception as e:
        print(f"Error extracting saddle configuration: {e}")
        return None
    
