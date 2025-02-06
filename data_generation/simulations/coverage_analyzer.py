import numpy as np
from scipy.spatial import ConvexHull
from sklearn.neighbors import KernelDensity

class CoverageAnalyzer:
    def __init__(self, bin_edges=None, n_bins=20):
        """
        Initialize coverage analyzer
        
        Args:
            bin_edges: List of arrays defining bin edges for each dimension
            n_bins: Number of bins to use if bin_edges not provided
        """
        self.bin_edges = bin_edges
        self.n_bins = n_bins
        
    def compute_coverage_metrics(self, points):
        """
        Compute various coverage metrics for points in Z-space
        
        Args:
            points: Array of shape (n_points, n_dims) containing z-space coordinates
            
        Returns:
            dict containing coverage metrics
        """
        n_points, n_dims = points.shape
        
        # Create bin edges if not provided
        if self.bin_edges is None:
            self.bin_edges = [np.linspace(np.min(points[:, i]), 
                                        np.max(points[:, i]), 
                                        self.n_bins + 1) 
                            for i in range(n_dims)]
        
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
        """
        Create 2D visualization of coverage for specified dimensions
        
        Args:
            points: Array of shape (n_points, n_dims)
            dims: Tuple of two dimensions to plot
        
        Returns:
            fig, ax: matplotlib figure and axis objects
        """
        import matplotlib.pyplot as plt
        
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(
            points[:, dims[0]], 
            points[:, dims[1]],
            bins=[self.bin_edges[dims[0]], self.bin_edges[dims[1]]]
        )
        
        # Plot heatmap
        fig, ax = plt.subplots()
        im = ax.imshow(hist.T, origin='lower', aspect='auto',
                      extent=[xedges[0], xedges[-1], 
                             yedges[0], yedges[-1]])
        plt.colorbar(im, ax=ax)
        
        ax.set_xlabel(f'Dimension {dims[0]}')
        ax.set_ylabel(f'Dimension {dims[1]}')
        ax.set_title('Coverage Density')
        
        return fig, ax