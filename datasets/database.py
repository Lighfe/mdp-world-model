# database.py
from sqlalchemy import Column, String, Float, JSON, ForeignKey, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.sqlite import FLOAT as SQLITE_FLOAT
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text

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

def clear_data_by_run_id(filename, tablename, run_ids):
    
    engine = create_engine(f'sqlite:///{filename}')
    init_db(engine)

    # Convert run_ids into a format for SQL query
    if isinstance(run_ids, (list, tuple, set)):
        run_ids_tuple = tuple(run_ids)
    else:
        raise ValueError("run_ids must be a list, tuple, or set of IDs")
    
    try:
        with engine.connect() as connection:
            # Use a parameterized query to prevent SQL injection
            query = text(f"DELETE FROM {tablename} WHERE run_id IN ({', '.join([':id' + str(i) for i in range(len(run_ids_tuple))])})")
            params = {f"id{i}": run_id for i, run_id in enumerate(run_ids_tuple)}
            connection.execute(query, params)
            connection.commit()  # Commit the changes

            # Delete also in configs table
            query = text(f"DELETE FROM configs WHERE run_id IN ({', '.join([':id' + str(i) for i in range(len(run_ids_tuple))])})")
            params = {f"id{i}": run_id for i, run_id in enumerate(run_ids_tuple)}
            connection.execute(query, params)
            connection.commit()  # Commit the changes
            print(f"Successfully deleted rows with run_id in {run_ids_tuple}.")
    
    except Exception as e:
        print(f"Error deleting rows: {e}")
