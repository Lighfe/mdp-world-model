import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

import sqlite3
import pandas as pd
import json
import numpy as np

from data_generation.simulations.grid import Grid, logistic_transformation, fractional_transformation
from data_generation.models.tech_substitution import TechnologySubstitution, NumericalSolver
#TODO: Add the other models and solvers and transformations here, but the solvers must have different names!!!



def get_data_from_database(db_path, table_name, run_ids):
   
    # Connect to the database
    conn = sqlite3.connect(db_path)

    # Create a query to select data from the table and filter by run_ids
    query = f"SELECT * FROM {table_name} WHERE run_id IN ({','.join('?' for _ in run_ids)})"
    query_conf = f"SELECT * FROM configs WHERE run_id IN ({','.join('?' for _ in run_ids)})"

    # Execute the query and load the data into pandas dataframes
    df = pd.read_sql_query(query, conn, params=run_ids)
    configs_df = pd.read_sql_query(query_conf, conn, params=run_ids)
    
    # Convert JSON strings to Python dictionaries
    configs_df["configurations"] = configs_df["configurations"].apply(json.loads)
    configs_df["simulation_params"] = configs_df["simulation_params"].apply(json.loads)

    # Merge both dictionaries into a single dictionary per row and create final dictionary indexed by run_id
    configs_df["configurations"] = configs_df.apply(
        lambda row: {**row["configurations"], "simulation_params": row["simulation_params"]}, axis=1)
    configs_dict = configs_df.set_index("run_id")["configurations"].to_dict()
    
    # Close the database connection
    conn.close()

    return (df, configs_dict)


def reconstruct_xyc(df):
    # TODO should we use another dataformat for the c column (has to be hashable!)?
    # Reconstruct data points and cells
    x_columns = [col for col in df.columns if col.startswith('x')]
    y_columns = [col for col in df.columns if col.startswith('y')]
    c_columns = [col for col in df.columns if col.startswith('c')]
    df['x'] = df.apply(lambda row: np.array([row[col] for col in x_columns]), axis=1)
    df['y'] = df.apply(lambda row: np.array([row[col] for col in y_columns]), axis=1)
    df['c'] = df.apply(lambda row: tuple(row[col] for col in c_columns), axis=1)
    df = df.drop(columns=x_columns+y_columns+c_columns)
    
    return df


def get_and_reconstruct_data(db_path, table_name, run_ids):
    df, configs_dict = get_data_from_database(db_path, table_name, run_ids)
    df = reconstruct_xyc(df)
    return (df, configs_dict)


def reconstruct_solver_and_grid(run_id, configs_dict):
    #Reconstruct the grid and solver from the configs_dict for a specific run_id

    co_grid = configs_dict[run_id]['grid']
    co_solver = configs_dict[run_id]['solver']

    bounds = co_grid['bounds']
    resolution = co_grid['resolution']
    transformations = [globals()[name](params['param']) for name, params in zip(co_grid['transformations'], co_grid['transformation_params'])]
    grid = Grid(bounds, resolution, transformations)

    # Extract class names
    model_class_name = co_solver['model']['model']
    solver_class_name = co_solver['solver']

    # Dynamically get the class from globals() (assuming they are defined in the current script)
    ModelClass = globals()[model_class_name]
    SolverClass = globals()[solver_class_name]

    # Extract model parameters (excluding non-model parameters like 'model', 'x_dim', etc.)
    model_params = {k: v for k, v in co_solver['model'].items() if k not in ['model', 'x_dim', 'control_dim', 'control_params']}

    # Instantiate the model and solver
    model = ModelClass(**model_params)
    solver = SolverClass(model)

    return grid, solver