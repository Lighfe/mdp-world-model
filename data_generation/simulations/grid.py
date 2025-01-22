import numpy as np
import random
import copy
from itertools import product
import matplotlib.pyplot as plt




class Grid:

    """ 
    Grid Class implementation for finite and infinite grids, with the possibility of transformations.

    If a transformation is defined, the grid is constructed in the transformed space with the given resolution.
    Then, the grid lines are transformed back to the original space.
    """
    
    def __init__(self, bounds, resolution, transformation=None, inverse_transformation=None):
        """
        Initializes the grid based on space intervals and resolution.

        Args:
            bounds (list of tuples): [(x_min, x_max), (y_min, y_max), ...] with np.inf for unboundedness.
            resolution (list of ints): Number of cells (divisions) in each dimension.
            transformation (function or list of functions): Transformation functions for the grid coordinates.
                                                            These functions should be able to handle np.inf.
        """
        self.bounds = copy.deepcopy(bounds)
        self.dimension = len(bounds)
        self.resolution = resolution
        self.indices = list((product(*[range(res) for res in self.resolution])))

        self.transformation = transformation
        self.inverse_transformation = inverse_transformation
        self.transformed_bool = False
        self.tf_bounds = copy.deepcopy(bounds) #will be changed in case of transformation

        #Transformation
        if self.transformation is not None:
            
            if isinstance(self.transformation, list):
                if len(self.transformation) != self.dimension:
                    raise ValueError("The number of transformation functions must match the grid dimension.")
            elif callable(self.transformation):
                self.transformation = [self.transformation] * self.dimension
                if callable(self.inverse_transformation):
                    self.inverse_transformation = [self.inverse_transformation] * self.dimension
            else:
                raise ValueError("Transformation must be a function or a list of functions.")
            
            
            for i, bound in enumerate(self.bounds):
                self.tf_bounds[i] = (self.transformation[i](bound[0]), self.transformation[i](bound[1]))

            self.tf_grid_lines = [np.linspace(start, end, num+1) for (start, end), num in zip(self.tf_bounds, self.resolution)]
            self.grid_lines = [np.vectorize(self.inverse_transformation[dim])(self.tf_grid_lines[dim]) for dim in range(self.dimension)]

            self.transformed_bool = True

        else:
            self.grid_lines = [np.linspace(start, end, num+1) for (start, end), num in zip(self.bounds, self.resolution)]
            self.tf_grid_lines = copy.deepcopy(self.grid_lines)

    
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

        return X_original
        

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
    
    
    def plot_2_dimensions_of_grid_with_vectorfield(self, dim1 = 0, dim2 = 1, vectorfield=None, streamplot=False):
        """
        Plots the grid in 2 dimensions.

        Args:
            dim1 (int): Dimension to plot on the x-axis.
            dim2 (int): Dimension to plot on the y-axis.
            vectorfield (function): Vectorfield to plot on the grid.
        
        Problem: How to plot the vectorfield in the transformed space?
        """
        """
        X = self.grid_lines[dim1]
        Y = self.grid_lines[dim2]
        X, Y = np.meshgrid(X, Y)

        if vectorfield is None:
            U, V = np.zeros_like(X), np.zeros_like(Y)
        else:
            U, V = vectorfield(X, Y)

        plt.figure(figsize=(8, 8))
        plt.quiver(X, Y, U, V, color='b', alpha=0.3)
        plt.grid(True)

        if streamplot:
            plt.streamplot(X, Y, U, V, color='b', linewidth=0.7)

        plt.show()
       
        """

        #TODO: Implement this method

        raise NotImplementedError("This method is not implemented yet.")


if __name__ == "__main__":

# Example for the usage of the Grid class
    myinfbounds = [(0, np.inf), (0, np.inf), (0, np.inf)]
    myresolution = [2, 2, 2]
    myeasybounds = [(0,4), (0, 4), (0, 4), (0, 4)]
    myeasyresolution = [2, 2, 2, 2]

    def mytransformation(x):
        if np.isinf(x):
            return 1
        else:
            return x / (x+10)

    def myinverse_transformation(y):
        if y == 1:
            return np.inf
        else:
            return -10*y / (y-1)

    infgrid = Grid(myinfbounds, myresolution, mytransformation, myinverse_transformation)
    easygrid = Grid(myeasybounds, myeasyresolution)

    x = infgrid.get_initial_conditions(2)
    
    
    print(infgrid.inverse_transform(x))
    for r in x:
        print(infgrid.get_cell_index(r, in_transformed_space=False))
    
    
