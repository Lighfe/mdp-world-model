import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from sqlalchemy import create_engine, select, MetaData
from sqlalchemy.orm import Session

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"Added {PROJECT_ROOT} to Python path")

from data_generation.simulations.grid import tangent_transformation

class TechSubstitutionDataset(Dataset):
    def __init__(self, db_path, tech_sub_solver, value_method):
        """
        Dataset for loading tech substitution data from SQLite database using SQLAlchemy
        
        Args:
            db_path: Path to the SQLite database
            tech_sub_solver: NumericalSolver instance for calculating v_true
        """
        self.db_path = db_path
        self.tech_sub_solver = tech_sub_solver
        self.value_method = value_method
        
        # Create SQLAlchemy engine
        self.engine = create_engine(f"sqlite:///{db_path}")
        
        # Reflect database structure
        metadata = MetaData()
        metadata.reflect(bind=self.engine)
        
        # Find the tech table - assuming it has 'tech' in the name
        tech_tables = [table for name, table in metadata.tables.items() if 'tech' in name.lower()]
        if not tech_tables:
            # If no table with 'tech' in name, just use the first table
            tech_tables = list(metadata.tables.values())
            if not tech_tables:
                raise ValueError("No tables found in the database")
        
        self.table = tech_tables[0]
        print(f"Using table: {self.table.name}")
        print(f"Columns: {[column.name for column in self.table.columns]}")
        
        # Count total rows
        with Session(self.engine) as session:
            self.length = session.query(self.table).count()
            print(f"Found {self.length} rows in the dataset")
        
        # Validate first few entries
        self._validate_samples()
    
    def _validate_samples(self, num_samples=5):
        """Check first few samples for any issues"""
        print(f"Validating first {num_samples} samples:")
        try:
            for i in range(min(num_samples, len(self))):
                x, c, y, v_true = self[i]
                print(f"Sample {i}: x={x.numpy()}, c={c.numpy()}, y={y.numpy()}, v_true={v_true.numpy()}")
                # Check for NaN values
                if (torch.isnan(x).any() or torch.isnan(c).any() or 
                    torch.isnan(y).any() or torch.isnan(v_true).any()):
                    print(f"WARNING: NaN values detected in sample {i}")
        except Exception as e:
            print(f"Error validating samples: {e}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
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
        
        # Calculate v_true using the solver's f_v function
        v_true = self.f_v(np.array([y0, y1]), value_method=self.value_method)
        
        # Check for NaN values
        if (torch.isnan(x).any() or torch.isnan(c).any() or 
            torch.isnan(y).any() or torch.isnan(v_true).any()):
            print(f"WARNING: NaN values found in sample {idx}: x={x}, c={c}, y={y}, v_true={v_true}")
            # Replace NaNs with zeros
            x = torch.nan_to_num(x, nan=0.0)
            c = torch.nan_to_num(c, nan=0.0)
            y = torch.nan_to_num(y, nan=0.0)
            v_true = torch.nan_to_num(v_true, nan=0.0)
        
        return x, c, y, v_true
    
    def f_v(self, y, value_method):
        if isinstance(y, (tuple, list)):
            y = np.array(y)

        # Now y is a NumPy array.
        if y.ndim == 1:
            # Expecting a single pair: (y1, y2)
            y1, y2 = y
        elif y.ndim == 2 and y.shape[1] == 2:
            # Expecting an array of shape (n, 2)
            y1 = y[:, 0]
            y2 = y[:, 1]
        else:
            raise ValueError("Input must be a tuple of two values or an array of shape (n, 2)")
        
        if value_method is None or value_method == 'market_share': # this is default
            v_true = y2 / (y1+y2 +1e-10)
            return torch.tensor([v_true], dtype=torch.float32)
        elif value_method == 'identity':
            # Get the tangent transformation function
            x0 = 3.0  # Center parameter
            alpha = 0.5  # Alpha parameter
            
            # Unpack only the forward transformation from the returned tuple
            transform_func, _, _ = tangent_transformation(x0, alpha)
            
            # Transform each dimension of y
            if y.ndim == 1:
                transformed_y = np.array([transform_func(y1), transform_func(y2)])
                return torch.tensor(transformed_y, dtype=torch.float32)
            else:
                # Apply transformation to each element
                transformed_y1 = np.array([transform_func(val) for val in y1])
                transformed_y2 = np.array([transform_func(val) for val in y2])
                transformed_y = np.column_stack((transformed_y1, transformed_y2))
                return torch.tensor(transformed_y, dtype=torch.float32)  # Shape (batch_size, 2)
        elif value_method == '90% market share':
            v_true = (y2 / (y1+y2 +1e-10)) >= 0.9
            return torch.tensor([v_true], dtype=torch.bool)
        else:
            raise NotImplementedError(f"Method '{value_method}' is not implemented.")


def create_data_loaders(db_path, tech_sub_solver, value_method= None, batch_size=64, val_size=1000, test_size=1000, seed=42):
    """
    Create training, validation, and test data loaders from the database
    
    Args:
        db_path: Path to the SQLite database
        tech_sub_solver: NumericalSolver instance
        batch_size: Batch size for training
        val_size: Number of samples to use for validation
        test_size: Number of samples to use for final testing
        seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects for training, validation and testing
    """
    
    # Create the dataset
    dataset = TechSubstitutionDataset(db_path, tech_sub_solver, value_method)
    
    # Prepare indices for training, validation, and testing
    indices = list(range(len(dataset)))
    np.random.seed(seed)
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
    
    # Create loaders
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices)
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices)
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_indices)
    )
    
    return train_loader, val_loader, test_loader