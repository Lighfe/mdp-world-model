import numpy as np
import random
import copy
from itertools import product
import sys


class Grid:

    """ 
    Grid Class implementation for finite and infinite grids, with the possibility of transformations.

    If a transformation is defined, the grid is constructed in the transformed space with the given resolution.
    Then, the grid lines are transformed back to the original space.
    """
    
    def __init__(self, bounds, resolution, grid_transformations=None):
        """
        Initializes the grid based on space intervals and resolution.

        Args:
            bounds (list of tuples): [(x_min, x_max), (y_min, y_max), ...] with np.inf for unboundedness.
            resolution (list of ints): Number of cells (divisions) in each dimension.
            grid_transformations (list of 3tuples functions): (transformation function, 
                                                    inverse transformation function,
                                                    transformation function derivative)
        """
        self.bounds = copy.deepcopy(bounds)
        self.dimension = len(bounds)
        self.resolution = resolution
        self.indices = list((product(*[range(res) for res in self.resolution])))

        # initialize empty
        # NOTE: Which of these are needed to be initialized empty in python?
        self.transformations = None
        self.inverse_transformations = None
        self.transformation_derivatives = None
        self.transformed_bool = False
        self.transformation_params = dict()
        self.tf_bounds = copy.deepcopy(bounds)
        self.grid_lines = None
        self.tf_grid_lines = None
        
        if grid_transformations is not None:
            self._init_transformations(grid_transformations)
        else:
            self.grid_lines = [np.linspace(start, end, num+1) for (start, end), num in zip(self.bounds, self.resolution)]
            self.tf_grid_lines = copy.deepcopy(self.grid_lines)


    def _init_transformations(self, grid_transformations):
        # check for correct format
        if isinstance(grid_transformations, list):
            if len(grid_transformations) != self.dimension:
                raise ValueError("Number of transformation tuples must match grid dimension.")
        # same functions used in every dimension
        # TODO: Is this needed? Does this work this way?
        elif callable(grid_transformations):
            grid_transformations = [grid_transformations] * self.dimension


        # Unzip functions
        transformations, inverse_transformations, transformation_derivatives = zip(*grid_transformations)
        self.transformations = transformations
        self.inverse_transformations = inverse_transformations
        self.transformation_derivatives = transformation_derivatives

        # save parameters
        self.transformation_params = [tf[0].parameters for tf in grid_transformations]
        
        # calculate transformed bounds
        for i, bound in enumerate(self.bounds):
            self.tf_bounds[i] = (self.transformations[i](bound[0]), self.transformations[i](bound[1]))

        # creates gridlines in Z-space based on bounds and resolution
        self.tf_grid_lines = [np.linspace(start, end, num+1) for (start, end), num in zip(self.tf_bounds, self.resolution)]
        # gridlines in X-space
        self.grid_lines = [np.vectorize(self.inverse_transformations[dim])(self.tf_grid_lines[dim]) for dim in range(self.dimension)]

        self.transformed_bool = True

    def get_config(self):
        config = {
            'bounds': self.bounds,
            'dimension': self.dimension,
            'resolution': self.resolution,
            'transformations': [tf.__name__ for tf in self.transformations],
            'transformation_params': self.transformation_params # list of dicts
        }
        return config

    
    def transform(self, coords):
        """
        Transforms a given coordinate or a set of such using the transformation functions.

        Args: coords (tuple or array): Real-world coordinate (e.g., (x, y)).
                                        If an array is given, the transformation is applied to each row (interpreted as a coordinate)
        """
        
        if not self.transformed_bool:
            return coords       #OR raise ValueError("No transformation functions defined.")
        coords = np.array(coords)
        if coords.ndim == 1:
            transformed_coords = np.array([self.transformation[dim](coords[dim]) for dim in range(self.dimension)])
        else:
            transformed_coords = np.empty_like(coords)
            for dim in range(self.dimension):
                transformed_coords[:, dim] = np.vectorize(self.transformation[dim])(coords[:, dim])
        return transformed_coords
        
        

    def inverse_transform(self, coords):
        """
        Transforms a given coordinate or a set of such from the transformed space using the inverse transformation functions.

        Args: coords (tuple or array): Transformed coordinate (e.g., (x, y)).
                                        If an array is given, the transformation is applied to each row (interpreted as a coordinate)
        """
        if not self.transformed_bool:
            return coords        #OR raise ValueError("No transformation functions defined.")
        
        coords = np.array(coords)
        if coords.ndim == 1:
            original_coords = np.array([self.inverse_transformation[dim](coords[dim]) for dim in range(self.dimension)])
        else:
            original_coords = np.empty_like(coords)
            for dim in range(self.dimension):
                original_coords[:, dim] = np.vectorize(self.inverse_transformation[dim])(coords[:, dim])
        return original_coords
        
       

    def get_cell_index(self, coord, in_transformed_space = False):
        """
        Finds the grid cell index for a given coordinate.
            For the boundary: point is always contained in the 'larger' index

        Args:   coord (tuple): Real-world coordinate (e.g., (x, y)).
                in_transformed_space (bool): If True, the coordinate is given in the transformed space, otherwise in the original space.

        Returns: idx (tuple): The grid cell index (e.g., (i, j)).
        """
        # TODO: Does this need to be vectorized?

        if len(coord) != self.dimension:
            raise ValueError("Coordinate dimensionality does not match space intervals.")
        
        if not in_transformed_space:
            coord = self.transform(coord)
        
        bounds = self.tf_bounds #search always in transformed space to avoid errors with np.inf
                    
        for dim in range(self.dimension):
            if (coord[dim] < bounds[dim][0]) or (coord[dim] > bounds[dim][1]):
                raise ValueError("Coordinate is out of bounds.")
                    
        idx = tuple(int((coord[dim] - bounds[dim][0]) /(bounds[dim][1] - bounds[dim][0]) * self.resolution[dim])
                        for dim in range(self.dimension))

        return idx


    
    def get_cell_coordinates(self, idx, transformed_space = False):
        """
        Returns the coordinates of a grid cell.

        Args: idx (tuple): The grid cell index (e.g., (i, j)).

        Returns: coords (list of tuples): The coordinates of the grid cell.
        """
        #TODO: Does this need to be vectorized?

        if not transformed_space:
            return [self.grid_lines[dim][i:i+2] for dim, i in enumerate(idx)]
        else:
            return [self.tf_grid_lines[dim][i:i+2] for dim, i in enumerate(idx)]

   
    def get_initial_conditions(self, num_points_per_cell=1):
        """
        Efficiently generates initial conditions for the whole grid.
        Returns a set of num_points_per_cell times num_grid_cells initial conditions for the grid.

        Args: num_points_per_cell (int): Number of initial conditions per grid cell.

        Returns: x (np.array): The initial conditions, each row is a point.
        """

        
        
        num_grid_cells = len(self.indices)
        
        X = np.random.random_sample((num_points_per_cell*num_grid_cells, self.dimension))
        
        delta_array = np.array([(bounds[1] - bounds[0]) / self.resolution[i] for i, bounds in enumerate(self.tf_bounds)]) # Array of cell size for each dimension
        X = np.multiply(X, delta_array)     # Scale the random numbers to the cell size

        repeated_lower_bounds_all_dim = np.zeros((num_points_per_cell*num_grid_cells, self.dimension))
        w = num_points_per_cell*num_grid_cells
        w0 = copy.deepcopy(w)
        for i, grid_lines in enumerate(self.tf_grid_lines):
            
            w = w/self.resolution[i]
            lower_bounds = grid_lines[:-1]
            repeated_lower_bounds = np.repeat(lower_bounds, w)
            multiple = w0 // len(repeated_lower_bounds)
            
            repeated_lower_bounds_all_dim[:, i] = np.tile(repeated_lower_bounds, multiple)

        X = np.add(X, repeated_lower_bounds_all_dim) 

        X_original = self.inverse_transform(X)

        #Create indices
        X_ids = []
        for idx in self.indices:
            for j in range(num_points_per_cell):
                X_ids.append("-".join(map(str, idx))+ f"_{j}")

        return (X_original, np.array(X_ids))
        

    def choose_random_point_from_cell(self, idx):
        """
        Chooses a single random point from a grid cell corresponding to the transformed space (because of unboundedness).
        Not as efficient as get_initial_conditions, but useful for testing.
        
        Args: idx (tuple): The grid cell index (e.g., (i, j)).

        Returns: point (tuple): The random point in original coordinates.
        """

        if self.transformed_bool:
            tf_coords = tuple(random.uniform(self.tf_grid_lines[dim][i], self.tf_grid_lines[dim][i+1]) for dim, i in enumerate(idx))
            rnd_point = self.inverse_transform(tf_coords)
        else:
            rnd_point = tuple(random.uniform(self.grid_lines[dim][i], self.grid_lines[dim][i+1]) for dim, i in enumerate(idx))
        
        return rnd_point

def fractional_transformation(param):
    """
    Generates the fractional transformation z=x/(x+param), its inverse and its derivate for a given parameter.
    Useful for space compression in the case [0, np.inf], cannot be used for spaces which include x = -param.
    
    Args:
        param (float, > 0): The parameter to be used in the fractional transformations
    Returns:
        tuple: A tuple containing three functions:
            - frac_transformation (function): Transforms an input x using the formula x / (x + param).
            - inverse_frac_transformation (function): Computes the inverse of the fractional transformation.
            - frac_transformation_derivative (function): Computes the derivative of the fractional transformation.
    
    """

    def frac_transformation(x):
        if np.isinf(x):
            return 1
        else:
            return x / (x+param)
        
    frac_transformation.parameters = {'param': param}

    def inverse_frac_transformation(z):
        if z == 1:
            return np.inf
        else:
            return -param*z / (z-1)
        
    def frac_transformation_derivative(x):
        if np.isinf(x):
            return 0
        else:
            return param / (x+param)**2
        
    
    return (frac_transformation, inverse_frac_transformation, frac_transformation_derivative)

