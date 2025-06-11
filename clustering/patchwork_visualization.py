import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
import seaborn as sns
import holoviews as hv
import panel as pn
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import colorcet as cc  # Better categorical colormaps
from data_generation.visualization.create_plots import create_2D_vectorfield, plot_2D_vector_field_over_grid
from clustering.patchwork import create_patchwork

hv.extension('bokeh')

def plot_entropy_overlay(ax, patchwork, cmap='viridis', alpha=0.5):
    """
    Overlays entropy values as a heatmap on the existing vector field plot.
    
    Parameters:
        ax (matplotlib axis): Axis on which to overlay the entropy heatmap.
        entropy_dict (dict): Dictionary mapping (i, j) grid indices to entropy values.
        X1, X2 (2D arrays): Meshgrid coordinates matching the entropy dictionary.
        cmap (str): Colormap for the entropy visualization.
        alpha (float): Transparency level of the overlay.
    """

    entropy_dict = {patchwork.patchindex_to_cell[k]: v['avg'] for k, v in patchwork.entropy_dict.items()}
    bounds, resolution = patchwork.grid.tf_bounds, patchwork.grid.resolution[0]
    cellsize = (bounds[0][1] - bounds[0][0]) / resolution
    X1 = np.linspace(bounds[0][0] + cellsize/2, bounds[0][1] - cellsize/2, resolution)
    X2 = np.linspace(bounds[1][0]+ cellsize/2, bounds[1][1] - cellsize/2, resolution)
    X1, X2 = np.meshgrid(X1, X2)

    # Convert entropy dictionary to a 2D array
    entropy_array = np.zeros_like(X1)
    
    for (i, j), entropy_value in entropy_dict.items():
        entropy_array[i, j] = entropy_value  # Fill the corresponding grid cell

    # Plot the entropy heatmap using pcolormesh (matching grid dimensions)
    heatmap = ax.pcolormesh(X1, X2, entropy_array.T, cmap=cmap, shading='auto', alpha=alpha)

    # Add a colorbar for entropy values
    cbar = plt.colorbar(heatmap, ax=ax, label="Entropy")
    cbar.ax.tick_params(labelsize=10)



def generate_random_grayscale_cmap(num_categories, seed=42):
    """
    Generates a randomized categorical grayscale colormap with `num_categories` distinct shades.

    Args:
        num_categories (int): Number of categories.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        List of `num_categories` grayscale colors in hex format.
    """
    if seed is not None:
        np.random.seed(seed)  # Ensure reproducibility

    # Generate `num_categories` shades between black (0) and white (1)
    grayscale_shades = np.linspace(0.3, 0.9, num_categories)  # Avoid pure black & white
    np.random.shuffle(grayscale_shades)  # Randomize order

    # Convert grayscale values to hex colors
    grayscale_cmap = [plt.cm.gray(val)[:3] for val in grayscale_shades]  # RGB tuples
    grayscale_cmap = ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) 
                      for r, g, b in grayscale_cmap]  # Convert to hex

    return grayscale_cmap




def plot_interactive_patchwork(patchwork, controls, solver, title = "", vf_resolution = 30, save_to_path = None, save_steps = 10, show_interactive = True):

    grid_class_name = type(patchwork.grid).__name__
    num_cells = len(patchwork.grid.indices)
    if hasattr(patchwork.grid, 'resolution'):
        grid_resolution = patchwork.grid.resolution
        grid_size = grid_resolution[0]
        cell_size = 1/grid_resolution[0]
    
    number_of_steps = patchwork.current_patches[-1] - (num_cells -1)
    bounds = [patchwork.grid.tf_bounds[0], patchwork.grid.tf_bounds[1]]
    X1lin = np.linspace(bounds[0][0], bounds[0][1], vf_resolution)
    X2lin = np.linspace(bounds[1][0], bounds[1][1], vf_resolution)
    
    
    # Prepare meshgrid for the streamplot calculations
    if patchwork.grid.transformed_bool:
       
        x_axis_tick_labels = [(tick, str(np.round(patchwork.grid.inverse_transformations[0](tick),2))) for tick in np.linspace(0,1,11)]
        y_axis_tick_labels = [(tick, str(np.round(patchwork.grid.inverse_transformations[1](tick),2))) for tick in np.linspace(0,1,11)]
        #Remove boundaries for infinity case (transformation doesn't work herd)
        if np.isinf(patchwork.grid.bounds[0][1]):
            X1lin = X1lin[:-1]
        if np.isinf(patchwork.grid.bounds[0][0]):
            X1lin = X1lin[1:]
        if np.isinf(patchwork.grid.bounds[1][1]):
            X2lin = X2lin[:-1]
        if np.isinf(patchwork.grid.bounds[1][0]):
            X2lin = X2lin[1:]
        
        X1, X2 = np.meshgrid(X1lin, X2lin)
        X1_org = np.vectorize(patchwork.grid.inverse_transformations[0])(X1)
        X2_org = np.vectorize(patchwork.grid.inverse_transformations[1])(X2)
    
    else:
        tick_positions = np.linspace(0,1,11)
        x_axis_tick_labels = [(tick_positions[i], str(np.round(tick,2))) for i, tick in enumerate(np.linspace(bounds[0][0],bounds[0][1],11))]
        y_axis_tick_labels = [(tick_positions[i], str(np.round(tick,2))) for i, tick in enumerate(np.linspace(bounds[1][0],bounds[1][1],11))]
        X1, X2 = np.meshgrid(X1lin, X2lin)


    def create_2D_vectorfield_here(control):
        if patchwork.grid.transformed_bool:
       
            U_org, V_org = create_2D_vectorfield(X1_org, X2_org, control=control, solver=solver)
            
            #Transform the vectorfield, multiply with Jacobian of transformation
            U = np.multiply(np.vectorize(patchwork.grid.transformation_derivatives[0])(X1_org), U_org)
            V = np.multiply(np.vectorize(patchwork.grid.transformation_derivatives[1])(X2_org), V_org)
        else:
            U, V = create_2D_vectorfield(X1, X2, control=control, solver=solver)
        return U, V
    
    #Function to generate streamplots for the different controls
    def generate_streamplot(count, U, V):
        """Generate a streamplot for a given control."""

        fig, ax = plt.subplots(figsize = (20,20), dpi=150)
        #TODO improve this color choice
        color = ['#3a3d3d', # dark gray
                 '#fe6100', #'#e35605',  # (dark) orange
                 '#0248fa', #'#0293fa', #'#02b8fa', #'#1d16f0',  # blue
                '#f52f98'  # pink – good light contrast, use sparingly
                ][count % 4] #[ '#1D58F3', '#DC267F', '#FFB000']
        ax.streamplot(X1, X2, U, V, color=color, linewidth=3, density = 0.7, arrowsize=4, broken_streamlines=False)

        # Convert the Matplotlib plot to a NumPy array
        ax.axis('off')  # Turn off the axis
        fig.tight_layout(pad=0)  # Remove padding
        ax.margins(0)  # Remove margins
        fig.canvas.draw()  # Draw the figure on the canvas
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # Reshape to height x width x 3 (RGB)
        # Make white and almost white pixels transparent
        threshold = 190  # Define a threshold for "almost white"
        alpha = np.where(np.all(image >= threshold, axis=-1), 0, 255).astype(np.uint8)
        image = np.dstack((image, alpha))
        plt.close(fig)  # Close the figure to prevent it from displaying immediately
        image[..., 3] = (image[..., 3].astype(float) * 0.6).astype(np.uint8)
        im = hv.RGB(image, bounds=(0, 0, 1, 1)) 
        return im
    
    def generate_nullclines(count, U, V):
        """Generate the 2 nullclines for a given control."""
               
        fig, ax = plt.subplots(figsize = (20,20))
        #TODO improve this color choice
        nullcline_colors = ['#161717',  # almost black
                '#ad0202',  # dark red
                 '#13239c',  # dark blue
                 '#750641'  # dark pink
                ][count % 4] #[ '#1D58F3', '#DC267F', '#FFB000']
        ax.contour(X1, X2, U, levels=[0], colors=[nullcline_colors], linewidths=14)
        ax.contour(X1, X2, V, levels=[0], colors=[nullcline_colors], linewidths=14)

        # Convert the Matplotlib plot to a NumPy array
        ax.axis('off')  # Turn off the axis
        fig.tight_layout(pad=0)  # Remove padding
        ax.margins(0)  # Remove margins
        fig.canvas.draw()  # Draw the figure on the canvas
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # Reshape to height x width x 3 (RGB)
        # Make white and almost white pixels transparent
        threshold = 190  # Define a threshold for "almost white"
        alpha = np.where(np.all(image >= threshold, axis=-1), 0, 255).astype(np.uint8)
        image = np.dstack((image, alpha))
        plt.close(fig)  # Close the figure to prevent it from displaying immediately
        image[..., 3] = (image[..., 3].astype(float) * 0.45).astype(np.uint8)
        im = hv.RGB(image, bounds=(0, 0, 1, 1))
        return im

    if show_interactive == True:
        # Precompute streamplots based on controls
        streamplots = {}
        nullclines = {}
        # Precompute streamplots and nullclines for all controls
        for count, control in enumerate(controls):
            U, V = create_2D_vectorfield_here(list(control))
            streamplots[control]  = generate_streamplot(count, U, V)
            nullclines[control] = generate_nullclines(count, U, V)
  
    #Create a grayscale + color map for patch color mapping
    grayscale_cmap = generate_random_grayscale_cmap(num_cells)
    color_cmap = (cc.glasbey_light * (number_of_steps // len(cc.glasbey_light) + 1))[:number_of_steps]
    combined_cmap = grayscale_cmap + color_cmap
    patch_color_mapping = {patch: combined_cmap[i] for i, patch in enumerate(range(patchwork.current_patches[-1]+1))}

    #Entropy colormap
    entropy_cmap = plt.cm.get_cmap('viridis')
    max_entropy = max(entropy for lst in patchwork.patch_to_history_of_avg_entropy.values() for _, entropy in lst)
    entropy_norm = mplcol.Normalize(vmin=0, vmax=max_entropy)

    def create_colorbar(colormap, norm, label="Patch Transition Entropy"):
        """Creates a matplotlib colorbar using the given colormap and normalization."""
        fig, ax = plt.subplots(figsize=(3.5,0.3), dpi = 400)  # Adjust figure size for the colorbar
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])  # Needed for the colorbar to work
        cbar = plt.colorbar(sm, cax=ax, orientation="horizontal")
        cbar.outline.set_visible(False)  # Remove the outer frame
        cbar.ax.tick_params(labelsize=8)  # Change the fontsize of the colorbar numbers
        cbar.set_label(label, fontsize=8)
        cbar.solids.set_alpha(0.6)  # Set alpha value for the colorbar
        return fig
   
    def create_loss_function_plot_hv(step=0):
        """
        Create a HoloViews plot of the loss function values over time with a vertical line.
        """
        curves = []
        color_cycle = plt.get_cmap('tab10').colors #hv.Cycle('Category10')  # Colorblind-friendly

        for idx, (loss_type, values) in enumerate(patchwork.loss_function.history_of_loss_function_values.items()):
            curve = hv.Curve((list(range(len(values))), values), 'Time Step', 'Loss Function Value', label =loss_type)\
                .opts(
                    color=color_cycle[idx],
                    line_width=2,
                    hover_tooltips=["$label", 'Time Step', ('Value','@{Loss Function Value}')]
                )
            curves.append(curve)

        # Create vertical line
        vline = hv.VLine(step).opts(color='red', line_dash='dashed', line_width=2)

        # Overlay curves and vertical line
        overlay = hv.Overlay(curves + [vline]).opts(
            title="Loss Function Value Over Time",
            width=400,
            height = 370,
            legend_position='bottom',
            legend_offset=(-50, 10),
            fontsize={'title': 12, 'labels': 10, 'ticks': 8, 'legend': 8},
            show_grid=True
        )

        return overlay
    

    #Preparation for VoronoiGrid
    if grid_class_name == 'VoronoiGrid':
        space_size = patchwork.grid.tf_bounds[0][1] - patchwork.grid.tf_bounds[0][0]
        normalize_vor = lambda v: (v - patchwork.grid.tf_bounds[0][0]) / space_size 
        regions_vertices = [patchwork.grid.voronoi.vertices[patchwork.grid.voronoi.regions[patchwork.grid.voronoi.point_region[v]]] for v in range(patchwork.grid.numbercells)]
        normalized_regions_vertices = [[normalize_vor(x) for x in region_vs] for region_vs in regions_vertices]
        cell_id_to_polygon = {i: {'x': [v[0] for v in region_vs], 'y': [v[1] for v in region_vs]} for i, region_vs in enumerate(normalized_regions_vertices)}

        patch_polygons = []

        for cell_id in patchwork.grid.indices:
            poly = cell_id_to_polygon[cell_id]
            patch_polygons.append({
                **poly,
                'patch': str(cell_id),
                'entropy': patchwork.get_patches_to_current_avg_entropy(0)[cell_id]
            })
        # Create HoloViews Polygons
        entropy_values = {patch: patchwork.get_patches_to_current_avg_entropy(0)[patch] for patch in patchwork.grid.indices}
        cmap_dict = {str(patch): mplcol.to_hex(entropy_cmap(entropy_norm(entropy))) for patch, entropy in entropy_values.items()}
        hv_polys = hv.Polygons(patch_polygons, vdims=['patch', 'entropy']).opts(
                                cmap=cmap_dict,  # your predefined patch color map
                                xlim=(0, 1), ylim=(0, 1),
                                width=600, height=600, tools=['hover'],
                                xticks=x_axis_tick_labels,  # Custom x-axis tick labels
                                yticks=y_axis_tick_labels,   # Custom y-axis tick labels
                                toolbar = 'above',
                                line_color= "#4C4C4C",  # keeps cell borders unless overridden
                                line_width = 0.5,
                                alpha=0.6
                            ) 

    def get_current_patchwork(step, selected_controls, selected_nullcline_controls, selected_color):
        
        cells_to_current_patches = patchwork.get_cells_to_current_patches(step)
        patch_to_current_avg_entropy = patchwork.get_patches_to_current_avg_entropy(step)
        unique_patches = set(cells_to_current_patches.values())
        # Assign colors to patches
        if selected_color == 'Patches':
            cmap_dict = {str(patch): patch_color_mapping[patch] for patch in unique_patches}
        elif selected_color == 'Entropy':
            entropy_values = {patch: patch_to_current_avg_entropy[patch] for patch in unique_patches}
            cmap_dict = {str(patch): mplcol.to_hex(entropy_cmap(entropy_norm(entropy))) for patch, entropy in entropy_values.items()}
        elif selected_color == 'White':
            cmap_dict = {str(patch): '#FFFFFF' for patch in unique_patches}


        if grid_class_name == 'Grid':
            # Normalize coordinates (scale x and y from [0, grid_size-1] to [0,1]) #maybe this should be changed if we don't map to (0,1)
            normalize = lambda v: ((v / grid_size) + 0.5*cell_size)

            cell_infos = [(normalize(x), normalize(y), str(patch), patch_to_current_avg_entropy[patch]) for (x,y), patch in cells_to_current_patches.items()]

            heatmap =  hv.HeatMap(cell_infos, kdims=['x', 'y'], vdims=['patch', 'entropy']).opts(alpha = 0.5,
                cmap=cmap_dict, 
                toolbar = 'above',
                xlim=(0, 1), ylim=(0, 1),
                width=600, height=600, tools=['hover'],
                xticks=x_axis_tick_labels,  # Custom x-axis tick labels
                yticks=y_axis_tick_labels   # Custom y-axis tick labels
                )
            # Identify patch borders
            patch_borders = []
            for (x, y), patch in cells_to_current_patches.items(): #find neighboring patch borders
                # Normalize coordinates
                cx, cy = normalize(x), normalize(y)
                dx = cell_size * 0.5

                if (x-1, y) not in cells_to_current_patches or cells_to_current_patches[(x-1, y)] != patch:
                    patch_borders.append((cx - dx, cy - dx, cx - dx, cy + dx))  # Left border

                if (x+1, y) not in cells_to_current_patches or cells_to_current_patches[(x+1, y)] != patch:
                    patch_borders.append((cx + dx, cy - dx, cx + dx, cy + dx))  # Right border

                if (x, y-1) not in cells_to_current_patches or cells_to_current_patches[(x, y-1)] != patch:
                    patch_borders.append((cx - dx, cy - dx, cx + dx, cy - dx))  # Bottom border

                if (x, y+1) not in cells_to_current_patches or cells_to_current_patches[(x, y+1)] != patch:
                    patch_borders.append((cx - dx, cy + dx, cx + dx, cy + dx))  # Top border

            # Create border segments
            linewidth = 4 / (1 + 0.1 * grid_resolution[0]**1.1) #smaller lines for higher resolutions
            borders = hv.Segments(patch_borders).opts(color='white', line_width=linewidth) 

            #Overlay 
            layers = [heatmap *borders] + [streamplots[control] for control in selected_controls if control in streamplots] + [nullclines[control] for control in selected_nullcline_controls if control in nullclines]

            

        elif grid_class_name == 'VoronoiGrid':

            patch_polygons = []

            for cell_id, patch_id in cells_to_current_patches.items():
                poly = cell_id_to_polygon[cell_id]
                patch_polygons.append({
                    **poly,
                    'patch': str(patch_id),
                    'entropy': patch_to_current_avg_entropy[patch_id]
                })
            # Update HoloViews Polygons
            hv_polys.data = patch_polygons  # Update the data in the existing hv_polys object
            hv_polys.opts(cmap=cmap_dict)  # your predefined patch color map

            # Identify patch borders
            patch_borders = []
            for patch_id in unique_patches:
                boundaries = patchwork.patch_to_vertices[patch_id]
                for vertices in boundaries:
                    for i in range(len(vertices)):
                        start = normalize_vor(vertices[i])
                        end = normalize_vor(vertices[(i + 1) % len(vertices)])
                        patch_borders.append((start[0], start[1], end[0], end[1]))  # (x1, y1, x2, y2)
            p_borders = hv.Segments(patch_borders).opts(color='white', line_width=1) 

            layers = [hv_polys * p_borders] + [streamplots[control] for control in selected_controls if control in streamplots] + [nullclines[control] for control in selected_nullcline_controls if control in nullclines]

        return hv.Overlay(layers)


    
    #Save to .gif
    if save_to_path != None:
        patchworks = {step: get_current_patchwork(step, [], [], selected_color='Entropy') for step in range(0,number_of_steps,save_steps)}
        holomap = hv.HoloMap(patchworks, kdims=['Step'])
        hv.save(holomap, save_to_path +".gif", fmt='gif')

        #Save colorbar externally
        fig = create_colorbar(entropy_cmap, entropy_norm)
        fig.savefig(save_to_path + "_colormap.png", bbox_inches='tight')

    if show_interactive == True:
        # Interactive slider and selection widget
        slider = pn.widgets.IntSlider(name="Clustering Step", start=0, end=number_of_steps, step=1)
        
        vector_selector_streamplots = pn.widgets.MultiChoice(name="Select Controls for Streamplots", options=list(controls), value=[])
        vector_selector_nullclines = pn.widgets.MultiChoice(name="Select Controls for Nullclines", options=list(controls), value=[]) # Initially, value=[] means no vector fields are selected.
        color_selector = pn.widgets.Select(name= "Select Color Mapping", options = ['Entropy', 'Patches', 'White'])
        
        # Bind the functions to the slider
        interactive_plot = pn.bind(get_current_patchwork, 
                                   step=slider, 
                                   selected_controls = vector_selector_streamplots, 
                                   selected_nullcline_controls = vector_selector_nullclines, 
                                   selected_color = color_selector)
        
        interactive_loss_function_plot = pn.bind(create_loss_function_plot_hv,
                                                 step = slider)
                                                 

        # Title for the widgetbox
        title = pn.pane.Markdown(f"<h1> Patchwork Visualization </h1> <br />  {title}")

        # Colorbar ##########################################
        colorbar_pane = pn.pane.Matplotlib(create_colorbar(entropy_cmap, entropy_norm), 
                                           tight=True,     
                                           #width=400,   # Adjust width as needed
                                           height=110   # Adjust height as needed
                                            )

        #### Layout ###########################################
        # Title Column
        loss_fct_explanation = pn.pane.Markdown("    ### " + patchwork.loss_function.loss_function_strg)
        title_column = pn.Column(title,
                                 pn.Spacer(height=5), 
                        loss_fct_explanation,
                        pn.Spacer(height=30),
                        pn.WidgetBox('## Patchwork Display Tools', slider, vector_selector_streamplots, vector_selector_nullclines, color_selector),
                        width = 400)
        
        # Wrap the interactive plot in a fixed-height column to help with alignment
        interactive_column = pn.Column(interactive_plot, sizing_mode='fixed', height=500)

        # Right-side column with bottom-aligned plots
        
        right_column = pn.Column(
            pn.Spacer(height=50), 
            colorbar_pane,
            pn.Spacer(height=50), 
            interactive_loss_function_plot,
            sizing_mode='fixed',
            height=500  # Match height with interactive_column
        )

        # Full dashboard row with bottom alignment
        dashboard = pn.Row(title_column, interactive_column, right_column)
        dashboard.servable() 
        dashboard.show()

    return



def create_plot_and_save_patchwork(db_name, 
                                   table_name,
                                   run_ids,  
                                   path_to_save = None, 
                                   gif_steps = 10, 
                                   title_interactive = "", 
                                   show_interactive = False, 
                                   entropy_strategy_strg= 'ShannonEntropyOnlyMerged',
                                   entropy_measure = 'shannon_entropy',
                                   loss_function_strg = 'TransitionEntropyLoss',
                                   loss_function_coeff = None):
    """
    Create a patchwork, plot it together with its first entropy and save the results to the corresponding path.
    Parameters:
        db_name (str): Path of the database to load data from.
        table_name (str): Name of the table in the database.
        run_ids (list): List of run IDs to include in the patchwork.
        path_to_save (str, optional): Path to save the generated patchwork visualization. Defaults to None, then nothing is saved.
        gif_steps (int, optional): Number of steps between frames in the saved GIF. Defaults to 10.
        title_interactive (str, optional): Title for the interactive visualization. Defaults to an empty string.
            Possibly something like techsub_title + f"resolution: {res}x{res} cells  <br /> timestep: {delta_t} <br /> samples per cell: {samples_per_cell} </h3>  <br />"
        show_interactive (bool, optional): Whether to display the interactive visualization. Defaults to False.
    """
    
    
    #Create subfolder and name
    if path_to_save != None:
        path_id = os.path.join(path_to_save, "-".join(run_ids)) 
        if not os.path.exists(path_id):
            os.makedirs(path_id)
        path_to_save = path_id + f"/Patchwork_{str(gif_steps)}stepsPerTime"

    patchwork, controls, solver = create_patchwork(db_name, 
                                                   table_name, 
                                                   run_ids,  
                                                   entropy_strategy_strg, 
                                                   entropy_measure,
                                                   loss_function_strg,
                                                   loss_function_coeff)
    if path_to_save != None:
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_2D_vector_field_over_grid(patchwork.grid, solver, control=controls[0], ax=ax, display_vectorfield=True, resolution = 21)
        plot_entropy_overlay(ax, patchwork, cmap='viridis', alpha=0.5)
        fig.savefig(path_id + "/StartEntropy.png")
    
    patchwork.run()
    plot_interactive_patchwork(patchwork, controls, solver, title_interactive, save_to_path=path_to_save, save_steps = gif_steps, show_interactive=show_interactive)


