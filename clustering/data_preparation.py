import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)
from pathlib import Path

import sqlite3
import pandas as pd
import json
import numpy as np


from data_generation.simulations.grid import Grid, logistic_transformation, fractional_transformation
from data_generation.simulations.grid import fractional_transformation as frac_transformation


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


def reconstruct_data(df, configs_dict, run_ids):
    # TODO find better dataformat for the c column
    
    # Check if all configurations are the same
    all_equal = all(configs_dict[run_id] == configs_dict[run_ids[0]] for run_id in run_ids)
    print("All configurations are the same:", all_equal)

    # Reconstruct the grid
    grid_dict = configs_dict[run_ids[0]]['grid']
    bounds = grid_dict['bounds']
    resolution = grid_dict['resolution']
    transformations = [globals()[name](params['param']) for name, params in zip(grid_dict['transformations'], grid_dict['transformation_params'])]
    grid = Grid(bounds, resolution, transformations)

    # Reconstruct data points and cells
    x_columns = [col for col in df.columns if col.startswith('x')]
    y_columns = [col for col in df.columns if col.startswith('y')]
    c_columns = [col for col in df.columns if col.startswith('c')]
    df['x'] = df.apply(lambda row: np.array([row[col] for col in x_columns]), axis=1)
    df['y'] = df.apply(lambda row: np.array([row[col] for col in y_columns]), axis=1)
    df['c'] = df.apply(lambda row: tuple(row[col] for col in c_columns), axis=1)
    df = df.drop(columns=x_columns+y_columns+c_columns)
    df['x_cell'] = df['x'].apply(grid.get_cell_index)
    df['y_cell'] = df['y'].apply(grid.get_cell_index)

    return df



def prepare_data(db_path, table_name, run_ids):
    df, configs_dict = get_data_from_database(db_path, table_name, run_ids)
    df = reconstruct_data(df, configs_dict, run_ids)
    return df, configs_dict