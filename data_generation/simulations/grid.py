import numpy as np
import random
import copy
from itertools import product
from scipy.spatial import Voronoi, Delaunay, KDTree, ConvexHull
from scipy.stats import qmc
import sys
from matplotlib import path

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
            self._init_transformed_grid_lines()
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

        self.transformed_bool = True


    def _init_transformed_grid_lines(self):
        """
        Initializes the grid lines in the transformed space based on the bounds and resolution.
        This is called after the transformations are initialized.
        """
        # creates gridlines in Z-space based on bounds and resolution
        self.tf_grid_lines = [np.linspace(start, end, num+1) for (start, end), num in zip(self.tf_bounds, self.resolution)]
        # gridlines in X-space
        self.grid_lines = [np.vectorize(self.inverse_transformations[dim])(self.tf_grid_lines[dim]) for dim in range(self.dimension)]



    def get_config(self):
        config = {
            'bounds': self.bounds,
            'dimension': self.dimension,
            'resolution': self.resolution,
            'transformations': None if self.transformations == None else [tf.__name__ for tf in self.transformations] ,
            'transformation_params': self.transformation_params if self.transformation_params != None else None # list of dicts
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
            transformed_coords = np.array([self.transformations[dim](coords[dim]) for dim in range(self.dimension)])
        else:
            transformed_coords = np.empty_like(coords)
            for dim in range(self.dimension):
                transformed_coords[:, dim] = np.vectorize(self.transformations[dim])(coords[:, dim])
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
            original_coords = np.array([self.inverse_transformations[dim](coords[dim]) for dim in range(self.dimension)])
        else:
            original_coords = np.empty_like(coords)
            for dim in range(self.dimension):
                original_coords[:, dim] = np.vectorize(self.inverse_transformations[dim])(coords[:, dim])
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

    
    def get_cell_centers(self, transformed_space=False):
        """
        Computes the set of all grid cell centers in the transformed or the original space.

        Args:
            transformed_space (bool): If True, computes centers in the transformed space; otherwise, in the original space.

        Returns:
            centers (np.array of dim (n_cells, n_dim)): A list of the center coordinates of each grid cell.
        """

        centers_1d = [(lines[:-1] + lines[1:]) / 2 for lines in self.tf_grid_lines]  # Compute midpoints
        
        # Generate all possible center coordinates using Cartesian product
        centers = np.array(list(product(*centers_1d)))

        if transformed_space == False:
            centers = self.inverse_transform(centers)

        return centers


    def get_neighbors(self, neighborhood_type='moore'):
        """
        Creates a dictionary mapping each grid cell to its neighbors for arbitrary dimensions.

        Parameters:  
        neighborhood_type: 'moore' (includes diagonals) or 'neumann' (axis-aligned only).

        Returns:
        - Dictionary where keys are coordinate tuples, and values are lists of neighboring coordinate tuples.
        """
        neighbors_dict = {}
        
        # Generate all possible direction shifts
        direction_shifts = list(product([-1, 0, 1], repeat=self.dimension))  # All possible shifts
        direction_shifts.remove((0,) * self.dimension)  # Remove the (0,0,..,0) case (itself)

        if neighborhood_type == 'neumann':
            direction_shifts = [shift for shift in direction_shifts if sum(abs(x) for x in shift) == 1]

        for index in product(*(range(s) for s in self.resolution)):  # Iterate over all grid cells
            neighbors = []
            for shift in direction_shifts:
                neighbor = tuple(idx + s for idx, s in zip(index, shift))
                if all(0 <= neighbor[d] < self.resolution[d] for d in range(self.dimension)):  # Boundary check
                    neighbors.append(neighbor)
            neighbors_dict[index] = neighbors

        return neighbors_dict



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



def dirichlet_random_point(vertices):
    """
    Sample uniformly from the simplex defined by `vertices`.
    `vertices` is an m x d array of simplex vertices.
    """
    m, d = vertices.shape
    # sample exponential(1) variates
    y = -np.log(np.random.rand(m))
    bary = y / np.sum(y)
    return bary @ vertices

class VoronoiGrid(Grid):
    """
    VoronoiGrid Class implementation for finite and infinite spaces, with the possibility of transformations.
    This class inherits from the Grid class.
    It is used to generate a Voronoi grid based on the initial conditions.
    ATTENTION: Takes a uniform side length of the bounding box as granted for Poisson Disk sampling.
    """

    def __init__(self, 
                 bounds: list[list[float]], 
                 numbercells: int = 10, 
                 seed: int = None, 
                 grid_transformations=None, 
                 cell_centers: np.ndarray = None, 
                 padding_margin = 'auto',
                 center_sampling_type: str = 'Poisson_Disk'):
        """
        Initializes the Voronoi tesselation grid based on the bounds of the space and the number of cells.
        The Voronoi tesselation is generated within th transformed space, if transformations are defined.

        Args:
            bounds (list of tuples): [(x_min, x_max), (y_min, y_max), ...] with np.inf for unboundedness. 
                ATTENTION: Takes a uniform side length of the bounding box as granted for Poisson Disk sampling.
            numbercells (int): number of Voronoi cells in the bounded space
            grid_transformations (list of 3tuples functions): (transformation function, 
                                                    inverse transformation function,
                                                    transformation function derivative)
            cell_centers (np.ndarray of dim(numbercenters, self.dimension)): 
                    Optional, if given, the Voronoi cells are generated based on these centers.
            center_sampling_type: str — type of sampling, either 'uniform' or 'Poisson_Disk'
        """

        self.bounds = np.array(copy.deepcopy(bounds))
        self.dimension = self.bounds.shape[0]

        self.transformations = None
        self.inverse_transformations = None
        self.transformation_derivatives = None
        self.transformed_bool = False
        self.transformation_params = dict()
        self.tf_bounds = copy.deepcopy(self.bounds)
        
        if grid_transformations is not None:
            self._init_transformations(grid_transformations)

        if cell_centers is not None:
            if cell_centers.shape[1] != self.dimension:
                raise ValueError("Cell centers dimensionality does not match space intervals.")
            elif not np.all((cell_centers >= self.tf_bounds[:, 0]) & (cell_centers <= self.tf_bounds[:, 1])):
                raise ValueError("Some cell centers are out of the transformed bounds.")
            else:
                self.original_sites = np.array(cell_centers)
                self.numbercells = self.original_sites.shape[0]

        else:
            self._generate_cell_centers(numbercells, type = center_sampling_type, seed=seed)

        self.kdtree = KDTree(self.original_sites)
        self._generate_voronoi_tesselation(padding_margin)    
        self._generate_padding_sites(margin=padding_margin) #Add padding sites to avoid unbounded Voronoi cells
        
        self.indices = list(range(self.numbercells)) #correspond to the cell centers in the order of the cell centers
        self.delaunay = Delaunay(self.padded_sites)
        
        self._cell_data = {} #storage for per-cell ridge data 

    def _generate_cell_centers(self, numbercells, type: str = 'Poisson_Disk', seed=None):
        """
        Generates the cell centers for the Voronoi grid.
        This is done by sampling points from the original sites.
        ATTENTION: Takes a uniform side length of the bounding box as granted for Poisson Disk sampling.

        Parameters:
        - number_cells: int — number of cell centers to generate approximately 
        - type: str — type of sampling, either 'uniform' or 'Poisson_Disk'
        - seed: int or None — random seed for reproducibility, if None, a random seed is generated
        """
        
        if seed is None:
            seed = np.random.SeedSequence().generate_state(1)[0]
        self.seed = seed
        rng = np.random.default_rng(self.seed)

        if type == 'uniform':
        # Sample points uniformly in the (transformed) bounded space
            self.original_sites = rng.uniform(
                low=self.tf_bounds[:, 0],
                high=self.tf_bounds[:, 1],
                size=(numbercells, self.dimension))
            
        elif type == 'Poisson_Disk':
            def estimate_poisson_radius(n_points, efficiency=0.75, side_length = 1, dim=2):
                return efficiency * side_length / (np.round(n_points ** (1/dim)) - 1)
            
            sidelengths = [self.tf_bounds[dim][1] - self.tf_bounds[dim][0] for dim in range(self.dimension)]
            if not all(np.isclose(sidelengths[0], s) for s in sidelengths):
                raise ValueError("All entries in sidelengths must be equal for the implemented version of Poisson Disk sampling.")
            sidelength = sidelengths[0]

            r = estimate_poisson_radius(numbercells, side_length = self.tf_bounds[0][1] - self.tf_bounds[0][0], dim=self.dimension)

            engine = qmc.PoissonDisk(self.dimension, radius=r, l_bounds=[0 for _ in range(self.dimension)], u_bounds=[sidelength for _ in range(self.dimension)], rng=rng)
            samples = engine.fill_space()

            # Move samples to the transformed space
            self.original_sites = samples + np.array(self.tf_bounds[:, 0])
        
        self.numbercells = self.original_sites.shape[0]


    def _generate_voronoi_tesselation(self, padding_margin = 'auto'):
        """
        Generates the Voronoi tesselation based on the cell centers.
        Using padding_margin, it reflects the sites near the boundaries of the bounding box to avoid unbounded Voronoi cells.

        Parameters:
        - padding_margin: float or 'auto' — margin threshold for reflecting sites near boundaries

        Used attributes:
        - original_sites: (n, d) array of original site coordinates
        - tf_bounds: list of (min, max) tuples for each dimension
        """
        used_margin = self._generate_padding_sites(margin=padding_margin)
        
        # Generate Voronoi tesselation in the transformed space
        self.voronoi = Voronoi(self.padded_sites)

        # Check if the Voronoi tesselation is bounded for the original sites
        orig_regions_bounded = False
        orig_regions_in_bounds = False
        while not orig_regions_bounded or not orig_regions_in_bounds:
        # Check if the Voronoi regions are bounded
        # If not, re-generate padding sites with a larger margin
        # This is done to ensure that the Voronoi cells are bounded 
            orig_regions_indices = self.voronoi.point_region[:self.numbercells]
            orig_regions = [self.voronoi.regions[i] for i in orig_regions_indices]
            orig_regions_bounded = not any(-1 in region for region in orig_regions)
            # Test also if any of the original sites are unbounded in the original space
            if orig_regions_bounded:
                vertices_of_orig_regions = np.vstack([self.voronoi.vertices[region] for region in orig_regions if len(region) > 0])
                print(vertices_of_orig_regions.shape)
                orig_regions_in_bounds = all(np.all((vertex >= self.tf_bounds[:, 0]) & (vertex <= self.tf_bounds[:, 1])) for vertex in vertices_of_orig_regions)
            
            if not orig_regions_bounded or not orig_regions_in_bounds:
                
                print(f"Warning: Original Voronoi cells are unbounded in the original space. Re-generating padding sites with larger margin: {2*used_margin}")
                used_margin = self._generate_padding_sites(margin=2*used_margin)  # Re-generate padding sites if original sites are unbounded
                self.voronoi = Voronoi(self.padded_sites)  # Recompute Voronoi tesselation with new padding sites
            if used_margin > self.tf_bounds[0][1] - self.tf_bounds[0][0]:
                break
        
        self.padding_margin = used_margin
        self.voronoi.sorted_ridge_vertices = [sorted(ridge) for ridge in self.voronoi.ridge_vertices]


    def _generate_padding_sites(self, margin='auto'):
        """
        Generate padding sites (cell centers) by reflecting those sites near the boundaries of the bounding box.
        For margin = 'auto', it estimates the margin based on the nearest-neighbor distances of the original sites.

        Parameters:
        - margin: float or 'auto' — margin threshold for reflecting sites near boundaries

        Used attributes:
        - original_sites: (n, d) array of original site coordinates
        - tf_bounds: list of (min, max) tuples for each dimension

        Returns:
        - padded_sites: (m, d) array including original and mirrored padding sites
        - is_original: (m,) boolean array indicating which are original sites
        """
        sites = np.asarray(self.original_sites)

        if margin == 'auto':
            # Estimate nearest-neighbor distances using KDTree
            distances, _ = self.kdtree.query(sites, k=2)  # first is the point itself
            margin = np.max(distances[:, 1])*1.5  # maximum nearest-neighbor distance with some security factor of 1.5

        padded_sites = [sites]
        is_original = [np.ones(self.numbercells, dtype=bool)]

        for axis in range(self.dimension):
            min_b, max_b = self.tf_bounds[axis]

            # Reflect near min boundary
            near_min_mask = sites[:, axis] - min_b <= margin
            near_min_sites = sites[near_min_mask].copy()
            if near_min_sites.size > 0:
                near_min_sites[:, axis] = 2 * min_b - near_min_sites[:, axis]
                padded_sites.append(near_min_sites)
                is_original.append(np.zeros(len(near_min_sites), dtype=bool))

            # Reflect near max boundary
            near_max_mask = max_b - sites[:, axis] <= margin
            near_max_sites = sites[near_max_mask].copy()
            if near_max_sites.size > 0:
                near_max_sites[:, axis] = 2 * max_b - near_max_sites[:, axis]
                padded_sites.append(near_max_sites)
                is_original.append(np.zeros(len(near_max_sites), dtype=bool))

        self.padded_sites = np.vstack(padded_sites)
        self.site_is_original = np.concatenate(is_original)
        return margin


    def get_config(self):
        config = {
            'grid_type': 'Voronoi',
            'bounds': self.bounds.tolist(),
            'dimension': self.dimension,
            'transformations': None if self.transformations is None else [tf.__name__ for tf in self.transformations],
            'transformation_params': self.transformation_params if self.transformation_params is not None else None,  # list of dicts
            'grid_seed': int(self.seed) if hasattr(self, 'seed') else None,
            'cell_centers': self.original_sites.tolist(),
            'padding_margin': self.padding_margin,
        }
        return config
    


    def get_cell_index(self, coord:np.ndarray, in_transformed_space = False):
        """
        Finds the Voronoi grid cell for a given coordinate.
        Returns the index of the closest Voronoi center (cell index).

        Args:   coord (tuple): Real-world coordinate (e.g., (x, y)).
                in_transformed_space (bool): If True, the coordinate is given in the transformed space, otherwise in the original space.

        Returns: idx (int): The Voronoi grid cell index.
        """
        if len(coord) != self.dimension:
            raise ValueError("Coordinate dimensionality does not match space intervals.")
        
        orig_coord = copy.deepcopy(coord)
        if not in_transformed_space:
            coord = self.transform(coord)
        
        bounds = self.tf_bounds #search always in transformed space to avoid errors with np.inf
                    
        for dim in range(self.dimension):
            if (coord[dim] < bounds[dim][0]) or (coord[dim] > bounds[dim][1]):
                raise ValueError(f"Coordinate {coord} is out of bounds, orginal coord before trafo was {orig_coord}.")

        return self.kdtree.query(coord)[1]
    

    def get_cell_coordinates(self, idx: int, transformed_space = False):
        """
        Returns the vertices of the corresponding Voronoi cell.

        Args: idx (int): The grid cell index .

        Returns: coords (list of tuples): The coordinates of the vertices.
        """
 
        region_index = self.voronoi.point_region[idx]
        region = self.voronoi.regions[region_index]
        if -1 in region or len(region) == 0:
            raise ValueError("Voronoi cell is unbounded or degenerate.")
        simplex = self.voronoi.vertices[region]
        if transformed_space:
            return simplex
        else:
            # Transform the vertices back to the original space
            return self.inverse_transform(simplex)


    def get_cell_centers(self, transformed_space=False):
        """
        Returns the set of all Voronoi cell centers in the transformed or the original space.

        Args:
            transformed_space (bool): If True, computes centers in the transformed space; otherwise, in the original space.

        Returns:
            centers (np.array of dim (n_cells, n_dim)): A list of the center coordinates of each Voronoi cell.
        """
        if transformed_space:
            return self.original_sites
        else:
            return self.inverse_transform(self.original_sites)
        

    def get_neighbors(self) -> dict[int, list[int]]:
        """
        Returns a dictionary mapping each Voronoi cell to its neighbors in the bounded region.
        The neighbors are determined by the Delaunay triangulation of the Voronoi cells.
        The keys are the cell indices, and the values are lists of neighboring cell indices.
        """

        neighbors = {i: set() for i in range(self.numbercells)}

        for simplex in self.delaunay.simplices:
            orig_sites_in_simplex = simplex[simplex < self.numbercells]
            if len(orig_sites_in_simplex) > 1:
                for i, site in enumerate(orig_sites_in_simplex):
                    for neighb in range(i + 1, len(orig_sites_in_simplex)):
                        neighbors[site].add(int(orig_sites_in_simplex[neighb]))
                        neighbors[orig_sites_in_simplex[neighb]].add(int(site))

        return {k: list(v) for k, v in neighbors.items()}
    
    def get_complete_cell_data(self):
        """
        Computes the facets of all Voronoi cells and stores them in the _cell_data attribute.
        This is used to avoid recomputing facets for each cell when sampling points.
        """
        for i in range(self.numbercells):
            if i not in self._cell_data:
                self._compute_cell_facets(i)
        return self._cell_data

    def _compute_cell_facets(self, i):
        """
        Computes the facets of a Voronoi cell based on its index.
        Args: i (int): The index of the Voronoi cell.
        Returns: facets (list of dicts): A list of dictionaries containing the vertices, height, cone volume, and simplices of the facets.
        """

        region_index = self.voronoi.point_region[i]
        region = self.voronoi.regions[region_index]
        if -1 in region or len(region) < self.dimension:
            raise ValueError(f"Voronoi cell {i} is unbounded or degenerate.")
        region_vertices = np.array([self.voronoi.vertices[v] for v in region])
        facets = []
        if self.dimension == 2:
            # In 2D, the region is a polygon; each facet is a line segment
            n = len(region_vertices)
            for k in range(n):
                v1 = region_vertices[k]
                v2 = region_vertices[(k + 1) % n]
                edge = np.array([v1, v2])
                edge_vec = v2 - v1
                site_vec = self.original_sites[i] - v1
                # now compute the component of site_vec which is orthogonal to the edge = height of the cone
                edge_unit = edge_vec / np.linalg.norm(edge_vec)
                proj = np.dot(site_vec, edge_unit) * edge_unit
                height_vec = site_vec - proj

                height = np.linalg.norm(height_vec)
                area = np.linalg.norm(edge_vec)  # edge length
                cone_volume = area * height / 2
                facets.append({
                    'vertices': edge,
                    'height': height,
                    'cone_volume': cone_volume,
                    'simplices': [edge]
                })
        else:
            # Higher dimensions: use convex hull facets
            # TODO still has some bugs!!  Cannot compute ConvexHull for d-1 simplices ...
            raise NotImplementedError("Facet computation for dimensions > 2 is not implemented yet.")
            hull = ConvexHull(region_vertices)
            for simplex_indices in hull.simplices:
                facet_vertices = region_vertices[simplex_indices]
                print(f"Facet vertices of cell {i}: {facet_vertices}")
                try:
                    facet_hull = ConvexHull(facet_vertices)
                    area = facet_hull.volume
                except:
                    print(f"Degenerate facet for cell {i} with vertices {facet_vertices}. Skipping.")
                    continue # skip degenerate facets
                
                facet_vertices_indices_vor = [self.voronoi.vertices.index(v) for v in facet_vertices]
                ridge_index = self.voronoi.sorted_ridge_vertices.index(tuple(sorted(facet_vertices_indices_vor)), None)
                neighbor_site_index = self.voronoi.ridge_points[ridge_index][0] if self.voronoi.ridge_points[ridge_index][0] != i else self.voronoi.ridge_points[ridge_index][1]
                neighbor_site = self.original_sites[neighbor_site_index]
                midpoint = (neighbor_site - self.original_sites[i])/2
                height = np.linalg.norm(self.original_sites[i] - midpoint)  # height is the distance to the midpoint of the ridge
                
                ''' 
                OR 

                # Compute height from site to affine span of facet
                x0 = facet_vertices[0]
                basis = facet_vertices[1:] - x0  # (d-1) x d matrix
                u = self.original_sites[i] - x0  # vector from facet base to site

                # Use QR decomposition to project u onto orthogonal complement of basis
                Q, _ = np.linalg.qr(basis.T)  # orthonormal basis of facet span
                proj = Q @ (Q.T @ u)          # projection of u onto span
                orthogonal_component = u - proj
                height = np.linalg.norm(orthogonal_component)
                '''
                
                cone_volume = area * height / self.dimension
                facets.append({
                    'vertices': facet_vertices,
                    'height': height,
                    'cone_volume': cone_volume,
                    'simplices': [facet_vertices]
                })
        if len(facets) == 0:
            raise ValueError(f"Could not compute valid facets for cell {i}.")
        self._cell_data[i] = facets
        return facets


    def choose_random_point_from_cell(self, i: int) -> np.ndarray:
        """
        Sample a single random point from a Voronoi cell corresponding to the transformed space (because of unboundedness).
       
        Args: i (int): The grid cell index.

        Returns: point (tuple): The random point in original coordinates.
        """
        if i not in self._cell_data:
            facets = self._compute_cell_facets(i)
        else:
            facets = self._cell_data[i]
        vols = np.array([f['cone_volume'] for f in facets])
        total = vols.sum()
        if total <= 0:
            raise ValueError(f"Degenerate cell {i} with zero total cone volume.")
        idx = np.random.choice(len(facets), p=vols / total) #TODO use my own random generator??
        f = facets[idx]
        simplex = f['simplices'][0]
        x_facet = dirichlet_random_point(simplex)
        u = (x_facet - self.original_sites[i])
        r = np.random.rand() ** (1.0 / self.dimension)
        rnd_point = self.original_sites[i] + r * u

        if self.transformed_bool:
            return self.inverse_transform(rnd_point)
        else:
            return rnd_point
    

    def get_initial_conditions(self, num_points_per_cell: int) -> np.ndarray:
        """
        Efficiently generates initial conditions for the Voronoi grid.
        Returns a set of num_points_per_cell times numbercells initial conditions for the grid.
        Args: num_points_per_cell (int): Number of initial conditions per Voronoi cell.
        Returns: points (np.array of dim(numbercells *num_points_per_cell, dimension)): The initial conditions, each row is a point.
        """
        points = []
        X_ids = []
        for idx in range(self.numbercells):
            for j in range(num_points_per_cell):
                pt = self.choose_random_point_from_cell(idx)
                points.append(pt)
                X_ids.append(f"{idx}_{j}")  # Unique ID for each point in the format "cell_index_point_index"

        return (np.array(points), np.array(X_ids))




# TODO the transformation functions should be named exactly as the generating function
def fractional_transformation(x0):
    """
    Generates the fractional transformation z=x/(x+x0), its inverse and its derivate for a given center x0.
    Useful for space compression in the case [0, np.inf], cannot be used for spaces which include x = -x0.
    Also known as Scale Odds Transformation
    
    Args:
        param (float, > 0): The parameter to be used in the fractional transformations
    Returns:
        tuple: A tuple containing three functions:
            - frac_transformation (function): Transforms an input x using the formula x / (x + x0).
            - inverse_frac_transformation (function): Computes the inverse of the fractional transformation.
            - frac_transformation_derivative (function): Computes the derivative of the fractional transformation.
    
    """
    def fractional_transformation(x):
        if np.isinf(x):
            return 1
        else:
            return x / (x+x0)
        
    fractional_transformation.parameters = {'param': x0}

    def inverse_fractional_transformation(z):
        if z == 1:
            return np.inf
        else:
            return -x0*z / (z-1)
        
    def fractional_transformation_derivative(x):
        if np.isinf(x):
            return 0
        else:
            return x0 / (x+x0)**2
        
    
    return (fractional_transformation, inverse_fractional_transformation, fractional_transformation_derivative)


def logistic_transformation(param_dict= {'k': 1, 'x_0': 0} ):
    """
    Generates the logistics transformation z=1/(1+exp(-k*(x-x_0))), its inverse and its derivate for given parameters k and x_0.
        k is the logistic growth rate, the steepness of the curve; and
        x_0 is the x value of the function's midpoint.
        For k = 1 and x_0 = 0 this returns the expit function.

    Useful for space compression in the case [np.inf, np.inf]
    See https://en.wikipedia.org/wiki/Logistic_function

    Args:
        param_dict (dict): containing the parameters 'k'(scalar) and 'x_0'(np.array) e.g. {'k': 1, 'x_0': 0}
    Returns:
        tuple: A tuple containing three functions:
            - logistic_trafo (function): Transforms an input x using the formula above.
            - inverse_logistic_trafo (function): Computes the inverse of the fractional transformation.
            - logistic_trafo_derivative (function): Computes the derivative of the fractional transformation.
    
    """
    k = param_dict.get('k')
    x_0 = param_dict.get('x_0')

    

    def logistic_transformation(x):
        return 1 / (1 + np.exp(-k*(x-x_0)))
    
    logistic_transformation.parameters = param_dict
    
    def inverse_logistic_transformation(x):
        if x == 1 or x == 0:
            return np.inf
        else:
            return (x_0 * k - np.log((1/x - 1))) / k
    
    def logistic_transformation_derivative(x):
        return k * np.exp(k*(x-x_0)) / (1 + np.exp(k*(x-x_0)))**2
    

    return (logistic_transformation, inverse_logistic_transformation, logistic_transformation_derivative)




def tangent_transformation(x0, alpha=0.5):
    """
    Generates the alphaed tangent transformation from x-space [0, ∞) to z-space [0, 1),
    its inverse, and its derivative. The transformation is defined so that the center
    x0 corresponds to z = 0.5.
    
    The mapping is:
        Forward Transformation (x -> z):
            z(x) = (2/π) * arctan[(x/x0)^alpha]
        Inverse Transformation (z -> x):
            x(z) = x0 * [ tan((π/2)*z) ]^(1/alpha)
        Derivative (dz/dx):
            dz/dx = (2/(π*x0)) * alpha * (x/x0)^(alpha-1) / [1 + (x/x0)^(2*alpha)]
    
    Args:
        x0 (float > 0): The center parameter such that the inverse transformation of 0.5 is x0.
        alpha (float > 0): alpha parameter to adjust the spread. When alpha=1, this
                             reduces to the usual tangent transformation.
        
    Returns:
        tuple: A tuple containing three functions:
            - transformation(x): Maps x in [0, ∞) to z in [0,1).
            - inverse_transformation(z): Maps z in [0,1) to x in [0, ∞).
            - derivative(x): The derivative dz/dx.
    """

    #NOTE: there is also a symmetric verison of this around 0 (which could be shifted by adding +x0)

    
    def tangent_transformation(x):
        # Maps x (in [0, ∞)) to z (in [0, 1)).
        if np.isinf(x):
            return 1
        else:
            return (2/np.pi) * np.arctan((x / x0)**alpha)
    
    tangent_transformation.parameters = {'x0': x0, 'alpha': alpha}
    
    def inverse_tangent_transformation(z):
        # Maps z (in [0, 1)) to x (in [0, ∞)).
        if z == 1:
            return np.inf
        else:
            return x0 * (np.tan((np.pi/2) * z))**(1/alpha)
    
    def tangent_transformation_derivative(x):
        # Computes dz/dx.
        if np.isinf(x):
            return 0
        else:
            ratio = x / x0 + 1e-10
            return (2 * alpha / (np.pi * x0)) * (ratio**(alpha - 1)) / (1 + ratio**(2 * alpha))
    
    return (tangent_transformation, inverse_tangent_transformation, tangent_transformation_derivative)

def negative_log_transformation(x0, alpha=0.5):
    """
    Generates the alphaed negative logarithm transformation from x-space [0, ∞) to z-space [0, 1),
    its inverse, and its derivative. The transformation is defined so that the center
    x0 corresponds to z = 0.5.
    
    The mapping is:
        Forward Transformation (x -> z):
            z(x) = 1 - exp[ - ln2 * (x/x0)^alpha ]
        Inverse Transformation (z -> x):
            x(z) = x0 * [ - (1/ln2) * ln(1 - z) ]^(1/alpha)
        Derivative (dz/dx):
            dz/dx = (ln2 * alpha / x0) * (x/x0)^(alpha - 1) * exp[ - ln2 * (x/x0)^alpha ]
    
    Args:
        x0 (float > 0): The center parameter such that the inverse transformation of 0.5 is x0.
        alpha (float > 0): alpha parameter to adjust the spread. When alpha=1, this
                             reduces to the usual negative logarithm transformation.
        
    Returns:
        tuple: A tuple containing three functions:
            - transformation(x): Maps x in [0, ∞) to z in [0,1).
            - inverse_transformation(z): Maps z in [0,1) to x in [0, ∞).
            - derivative(x): The derivative dz/dx.
    """
    
    def negative_log_transformation(x):
        # Maps x (in [0, ∞)) to z (in [0, 1)).
        if np.isinf(x):
            return 1
        else:
            return 1 - np.exp(- np.log(2) * (x / x0)**alpha)
    
    negative_log_transformation.parameters = {'x0': x0, 'alpha': alpha}
    
    def inverse_negative_log_transformation(z):
        # Maps z (in [0, 1)) to x (in [0, ∞)).
        if z == 1:
            return np.inf
        else:
            return x0 * ((-1/np.log(2)) * np.log(1 - z))**(1/alpha)
    
    def negative_log_transformation_derivative(x):
        # Computes dz/dx.
        if np.isinf(x):
            return 0
        else:
            ratio = x / x0
            return (np.log(2) * alpha / x0) * (ratio**(alpha - 1)) * np.exp(- np.log(2) * (ratio)**alpha)
    
    return (negative_log_transformation, inverse_negative_log_transformation, negative_log_transformation_derivative)