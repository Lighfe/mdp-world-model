import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
import holoviews as hv
import panel as pn
import pandas as pd
import colorcet as cc  # Better categorical colormaps
from data_generation.visualization.create_plots import create_2D_vectorfield, plot_2D_vector_field_over_grid
from patchwork import create_patchwork

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

    grid_resolution = patchwork.grid.resolution
    grid_size = grid_resolution[0]
    num_cells = len(patchwork.grid.indices)
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

        fig, ax = plt.subplots(figsize = (20,20))
        #TODO improve this color choice
        color = ['#ff7f00',  # orange
                 '#0248fa', #'#0293fa', #'#02b8fa', #'#1d16f0',  # blue
                 '#4e5252',  # gray
                '#f52f98'  # pink – good light contrast, use sparingly
                ][count % 4] #[ '#1D58F3', '#DC267F', '#FFB000']
        ax.streamplot(X1, X2, U, V, color=color, linewidth=4, density = 0.7, arrowsize=4, broken_streamlines=False)

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
        image[..., 3] = (image[..., 3].astype(float) * 0.7).astype(np.uint8)
        im = hv.RGB(image, bounds=(0, 0, 1, 1)) 
        return im
    
    def generate_nullclines(count, U, V):
        """Generate the 2 nullclines for a given control."""
               
        fig, ax = plt.subplots(figsize = (20,20))
        #TODO improve this color choice
        nullcline_colors = ['#ad0202',  # dark red
                 '#13239c',  # dark blue
                 '#161717',  # almost black
                 '#750641'  # dark pink
                ][count % 4] #[ '#1D58F3', '#DC267F', '#FFB000']
        ax.contour(X1, X2, U, levels=[0], colors=[nullcline_colors], linewidths=11)
        ax.contour(X1, X2, V, levels=[0], colors=[nullcline_colors], linewidths=11)

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
        image[..., 3] = (image[..., 3].astype(float) * 0.5).astype(np.uint8)
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

    def create_colorbar(colormap, norm, label="Entropy"):
        """Creates a matplotlib colorbar using the given colormap and normalization."""
        fig, ax = plt.subplots(figsize=(0.3, 3.5), dpi = 200)  # Adjust figure size for the colorbar
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])  # Needed for the colorbar to work
        cbar = plt.colorbar(sm, cax=ax, orientation="vertical")
        cbar.ax.tick_params(labelsize=8)  # Change the fontsize of the colorbar numbers
        cbar.set_label(label, fontsize=8)
        cbar.solids.set_alpha(0.6)  # Set alpha value for the colorbar
        return fig


    def get_current_patchwork(step, selected_controls, selected_nullcline_controls, selected_color):
        
        cells_to_current_patches = patchwork.get_cells_to_current_patches(step)
        patch_to_current_avg_entropy = patchwork.get_patches_to_current_avg_entropy(step)

        # Normalize coordinates (scale x and y from [0, grid_size-1] to [0,1]) #maybe this should be changed if we don't map to (0,1)
        normalize = lambda v: ((v / grid_size) + 0.5*cell_size)

        cell_infos = [(normalize(x), normalize(y), str(patch), patch_to_current_avg_entropy[patch]) for (x,y), patch in cells_to_current_patches.items()]

        # Assign colors to patches
        unique_patches = set(int(patch) for (_, _, patch, _) in cell_infos)
        if selected_color == 'Patches':
            cmap_dict = {str(patch): patch_color_mapping[patch] for patch in unique_patches}
        elif selected_color == 'Entropy':
            entropy_values = {patch: patch_to_current_avg_entropy[patch] for patch in unique_patches}
            cmap_dict = {str(patch): mplcol.to_hex(entropy_cmap(entropy_norm(entropy))) for patch, entropy in entropy_values.items()}
        elif selected_color == 'White':
            cmap_dict = {str(patch): '#FFFFFF' for patch in unique_patches}
        
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
        if selected_color == 'White':
            linewidth = 3 / (1 + 0.3 * grid_resolution[0]**1.1)
            borders = hv.Segments(patch_borders).opts(color='black', line_width=linewidth) 
        else:
            linewidth = 3 / (1 + 0.1 * grid_resolution[0]**1.1) #smaller lines for higher resolutions
            borders = hv.Segments(patch_borders).opts(color='white', line_width=linewidth) 

        #Overlay 
        layers = [heatmap *borders] + [streamplots[control] for control in selected_controls if control in streamplots] + [nullclines[control] for control in selected_nullcline_controls if control in nullclines]

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
        
        # Bind the function to the slider
        interactive_plot = pn.bind(get_current_patchwork, 
                                   step=slider, 
                                   selected_controls = vector_selector_streamplots, 
                                   selected_nullcline_controls = vector_selector_nullclines, 
                                   selected_color = color_selector)

        # Title for the widgetbox
        title = pn.pane.Markdown(f"<h1> Patchwork Visualization </h1> <br />  {title}")

        # Colorbar
        colorbar_pane = pn.pane.Matplotlib(create_colorbar(entropy_cmap, entropy_norm), tight=True)
        
        # Layout
        dashboard = pn.Row(
            pn.Column(title, pn.WidgetBox('## Patchwork Display Tools', slider, vector_selector_streamplots, vector_selector_nullclines, color_selector)),
            interactive_plot,
            pn.Column(pn.Spacer(height=90), colorbar_pane)  # Add spacer to adjust the position of the colorbar
        )
        dashboard.servable() 
        dashboard.show()

    return



def create_plot_and_save_patchwork(db_name, table_name, run_ids,  path_to_save = None, gif_steps = 10, title_interactive = "", show_interactive = False, entropy_strategy_strg= 'ShannonEntropyOnlyMerged',):
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
        path_to_save = path_id + f"/firstPatchwork_{str(gif_steps)}stepsPerTime"

    patchwork, controls, solver = create_patchwork(db_name, table_name, run_ids,  entropy_strategy_strg)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_2D_vector_field_over_grid(patchwork.grid, solver, control=controls[0], ax=ax, display_vectorfield=True, resolution = 21)
    plot_entropy_overlay(ax, patchwork, cmap='viridis', alpha=0.5)
    if path_to_save != None:
        fig.savefig(path_id + "/StartEntropy.png")
    
    patchwork.run()
    plot_interactive_patchwork(patchwork, controls, solver, title_interactive, save_to_path=path_to_save, save_steps = gif_steps, show_interactive=show_interactive)


