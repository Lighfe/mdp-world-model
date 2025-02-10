# database.py
from sqlalchemy import Column, String, Float, JSON, ForeignKey, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.sqlite import FLOAT as SQLITE_FLOAT
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# Create shared metadata instance
metadata = MetaData()

# Create configs table
configs = Table(
    'configs', 
    metadata,
    Column('run_id', String, primary_key=True),
    Column('configurations', JSON),
    Column('simulation_params', JSON),
    Column('performance', JSON),
    Column('system', JSON)
)

def create_results_table(name, x_dim, control_dim):
    """Dynamically create results table for specific model using shared metadata"""
    columns = [
        Column('run_id', String, ForeignKey('configs.run_id')),
        Column('trajectory_id', String),
        Column('t0', Float),
        Column('t1', Float)
    ]
    
    # Add state variables (highest precision)
    for i in range(x_dim):
        columns.append(Column(f'x{i}', SQLITE_FLOAT))
        
    # Add control variables
    for i in range(control_dim):
        columns.append(Column(f'c{i}', SQLITE_FLOAT))
        
    # Add output variables
    for i in range(x_dim):
        columns.append(Column(f'y{i}', SQLITE_FLOAT))
        
    return Table(name, metadata, *columns, extend_existing=True)

def get_engine(filename):
    """Create SQLAlchemy engine"""
    return create_engine(f'sqlite:///{filename}')

def init_db(engine):
    """Initialize database with all tables"""
    metadata.create_all(engine)