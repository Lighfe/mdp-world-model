import numpy as np
from scipy.spatial import ConvexHull
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

class CoverageAnalyzer:
    def __init__(self, n_bins=20, space_type='Z', max_value=1e6):
        """
        Initialize coverage analyzer
        
        Args:
            n_bins: Number of bins per dimension
            space_type: 'Z' for transformed space [0,1], 'X' for original space
            max_value: Upper bound for X space binning (only used if space_type='X')
        """
        self.n_bins = n_bins
        self.space_type = space_type
        self.max_value = max_value
        self.bin_edges = None
        
    def compute_coverage_metrics(self, points):
        """
        Compute various coverage metrics for points in Z-space
        
        Args:
            points: Array of shape (n_points, n_dims) containing z-space coordinates
            
        Returns:
            dict containing coverage metrics
        """
        n_points, n_dims = points.shape
        
        # Create bin edges based on space type
        if self.bin_edges is None:
            if self.space_type == 'Z':
                # For Z-space, always use [0,1] range
                self.bin_edges = [np.linspace(0, 1, self.n_bins + 1) 
                                for _ in range(n_dims)]
            else:
                # For X-space, use logarithmic binning from min to max_value
                self.bin_edges = []
                for i in range(n_dims):
                    min_val = max(np.min(points[:, i]), 1e-10)  # Avoid log(0)
                    self.bin_edges.append(
                        np.exp(np.linspace(np.log(min_val),
                                         np.log(self.max_value),
                                         self.n_bins + 1))
                    )
        
        # Compute histogram
        hist, _ = np.histogramdd(points, bins=self.bin_edges)
        
        # Basic coverage metrics
        metrics = {
            'occupied_bins': np.sum(hist > 0),
            'total_bins': np.prod([len(edges)-1 for edges in self.bin_edges]),
            'coverage_ratio': np.mean(hist > 0),
            'max_density': np.max(hist),
            'min_density': np.min(hist[hist > 0]) if np.any(hist > 0) else 0
        }
        
        # Add convex hull volume if dimensionality allows
        try:
            hull = ConvexHull(points)
            metrics['hull_volume'] = hull.volume
        except:
            metrics['hull_volume'] = None
            
        # Estimate effective volume using kernel density
        try:
            kde = KernelDensity(bandwidth=0.1).fit(points)
            log_dens = kde.score_samples(points)
            metrics['effective_volume'] = np.exp(-np.mean(log_dens))
        except:
            metrics['effective_volume'] = None
            
        return metrics
    
    def plot_2d_coverage(self, points, dims=(0,1)):
        """Create 2D visualization of coverage"""
        
        # TODO find a way to visualize X space
        if self.space_type == 'X':
            raise NotImplementedError("X-space plotting not yet implemented")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Scatter plot with increased visibility
        ax1.scatter(points[:, dims[0]], points[:, dims[1]], alpha=0.5, s=5)
        ax1.set_title('Raw Data Points')
        
        # Use pre-defined bin edges for histogram
        hist, _, _ = np.histogram2d(
            points[:, dims[0]], 
            points[:, dims[1]],
            bins=[self.bin_edges[dims[0]], self.bin_edges[dims[1]]]
        )
        
        # Create meshgrid from bin edges
        X, Y = np.meshgrid(self.bin_edges[dims[0]][:-1], self.bin_edges[dims[1]][:-1])
        im = ax2.pcolormesh(X, Y, hist.T, shading='auto', vmax=5)
        plt.colorbar(im, ax=ax2)
        ax2.set_title('Coverage Density')
        
        for ax in [ax1, ax2]:
            ax.set_xlabel(f'Dimension {dims[0]}')
            ax.set_ylabel(f'Dimension {dims[1]}')
            
        plt.tight_layout()
        return fig, (ax1, ax2)