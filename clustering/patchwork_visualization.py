import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
import holoviews as hv
import panel as pn
import pandas as pd
import colorcet as cc  # Better categorical colormaps
from data_generation.visualization.create_plots import create_2D_vectorfield

hv.extension('bokeh')

def test_hv():
    # Generate flow field (example: simple vortex)
    x, y = np.linspace(-2, 2, 20), np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    U, V = -Y, X  # Rotational field

    def plot_streamplot(step):
        fig, ax = plt.subplots()
        ax.streamplot(X, Y, U, V, color='b', linewidth=1)
        ax.set_title(f"Streamplot at step {step}")
        return fig

    # Create interactive widget
    slider = pn.widgets.IntSlider(name="Step", start=0, end=10, step=1)
    interactive_plot = pn.bind(plot_streamplot, step=slider)

    dashboard = pn.Column(slider, interactive_plot)
    dashboard.servable()
    dashboard.show()
    
    return

def create_interactive_streamplot():
    """Creates an interactive streamplot with a button to change vector fields."""
    
   
    # Define vector fields
    x, y = np.linspace(-2, 2, 20), np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)

    vector_fields = {
        "None": None,  # No streamplot
        "Rotational": (-Y, X),  # Circular flow
        "Linear": (np.ones_like(X), np.zeros_like(Y)),  # Uniform flow
        "Random": (np.random.randn(*X.shape), np.random.randn(*Y.shape)),  # Random
    }

    def plot_streamplot(step, field_name):
        """Generates a streamplot for the selected vector field."""
        
        data = [(i/5 + 0.1, j/5 +0.1,  i*j) for i in range(5) for j in range(5) if i!=j]
        hm = hv.HeatMap(data).sort().opts(width = 500, height= 500, tools=['hover'] )
        
        if field_name == "None":
            return hm  # Return an empty figure

        U, V = vector_fields[field_name]
        fig, ax = plt.subplots(figsize = (10,10))
        ax.streamplot(X, Y, U, V, color='b', linewidth=1)

        # Convert the Matplotlib plot to a NumPy array
        fig.canvas.draw()  # Draw the figure on the canvas
        # Extract the image as a numpy array from the canvas
        ax.axis('off')  # Turn off the axis
        fig.tight_layout(pad=0)  # Remove padding
        ax.margins(0)  # Remove margins
        ax.set_axis_off()  # Turn off the axis
        fig.canvas.draw()  # Draw the figure on the canvas
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # Reshape to height x width x 3 (RGB)
        
        # Make white pixels transparent
        image = np.where(np.all(image == [255, 255, 255], axis=-1, keepdims=True), [0, 0, 0, 0], np.concatenate([image, np.full((*image.shape[:2], 1), 255)], axis=-1))
        plt.close(fig)  # Close the figure to prevent it from displaying immediately
        im = hv.RGB(image, bounds=(0, 0, 1, 1)) #fig.canvas.get_width_height()[0], fig.canvas.get_width_height()[1]))

        

        return hm *im 
    # Create widgets
    slider = pn.widgets.IntSlider(name="Step", start=0, end=10, step=1)
    dropdown = pn.widgets.Select(name="Vector Field", options=list(vector_fields.keys()))

    # Bind interactive plot
    interactive_plot = pn.bind(plot_streamplot, step=slider, field_name=dropdown)

    dashboard = pn.Column(slider, dropdown,  interactive_plot)
    dashboard.servable()
    dashboard.show()
    
    return 



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




def plot_interactive_patchwork(patchwork, controls, solver, title = "", vf_resolution = 15):

    grid_resolution = patchwork.grid.resolution
    grid_size = grid_resolution[0]
    num_cells = len(patchwork.grid.indices)
    cell_size = 1/grid_resolution[0]
    number_of_steps = patchwork.current_patches[-1] - (num_cells -1)
    bounds = [patchwork.grid.tf_bounds[0], patchwork.grid.tf_bounds[1]]
    X1lin = np.linspace(bounds[0][0], bounds[0][1], vf_resolution)
    X2lin = np.linspace(bounds[1][0], bounds[1][1], vf_resolution)
    
    axis_tick_labels = [(tick, str(np.round(patchwork.grid.inverse_transformations[0](tick),2))) for tick in np.linspace(0,1,11)]
    
    # Prepare meshgrid for the streamplot calculations
    if patchwork.grid.transformed_bool:
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
        X1, X2 = np.meshgrid(X1lin, X2lin)
        
    #Function to generate streamplots for the different controls
    def generate_streamplot(control, count):
        """Generate a streamplot for a given control."""
        control = list(control)
        if patchwork.grid.transformed_bool:
       
            U_org, V_org = create_2D_vectorfield(X1_org, X2_org, control=control, solver=solver)
            
            #Transform the vectorfield, multiply with Jacobian of transformation
            U = np.multiply(np.vectorize(patchwork.grid.transformation_derivatives[0])(X1_org), U_org)
            V = np.multiply(np.vectorize(patchwork.grid.transformation_derivatives[1])(X2_org), V_org)
        else:
            U, V = create_2D_vectorfield(X1, X2, control=control, solver=solver)
        
        fig, ax = plt.subplots(figsize = (20,20))
        #TODO improve this color choice
        color = [ '#1D58F3', '#DC267F', '#FFB000'][count]
        ax.streamplot(X1, X2, U, V, color=color, linewidth=2.7, density = 1, arrowsize=2, broken_streamlines=False)

        # Convert the Matplotlib plot to a NumPy array
        ax.axis('off')  # Turn off the axis
        fig.tight_layout(pad=0)  # Remove padding
        ax.margins(0)  # Remove margins
        fig.canvas.draw()  # Draw the figure on the canvas
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # Reshape to height x width x 3 (RGB)
        # Make white and almost white pixels transparent
        threshold = 190  # Define a threshold for "almost white"
        image = np.where(np.all(image >= threshold, axis=-1, keepdims=True), [0, 0, 0, 0], np.concatenate([image, np.full((*image.shape[:2], 1), 255)], axis=-1))
        plt.close(fig)  # Close the figure to prevent it from displaying immediately
        im = hv.RGB(image, bounds=(0, 0, 1, 1)) 
        return im

    # Precompute streamplots based on controls
    streamplots = {control: generate_streamplot(control, count) for count, control in enumerate(controls)}
  
    #Create a grayscale + color map for patch color mapping
    grayscale_cmap = generate_random_grayscale_cmap(num_cells)
    color_cmap = (cc.glasbey_light * (number_of_steps // len(cc.glasbey_light) + 1))[:number_of_steps]
    combined_cmap = grayscale_cmap + color_cmap
    patch_color_mapping = {patch: combined_cmap[i] for i, patch in enumerate(range(patchwork.current_patches[-1]+1))}

    #Entropy colormap
    entropy_cmap = plt.cm.get_cmap('viridis')
    entropy_norm = mplcol.Normalize(vmin=0, vmax=1)

    def get_current_patchwork(step, selected_controls, selected_color):
        
        cells_to_current_patches = patchwork.get_cells_to_current_patches(step)

        # Normalize coordinates (scale x and y from [0, grid_size-1] to [0,1]) #maybe this should be changed if we don't map to (0,1)
        normalize = lambda v: ((v / grid_size) + 0.5*cell_size)

        cell_infos = [(normalize(x), normalize(y), str(patch), patchwork.entropy_dict[patch]['avg']) for (x,y), patch in cells_to_current_patches.items()]

        # Assign colors to patches
        unique_patches = set(int(patch) for (_, _, patch, _) in cell_infos)
        if selected_color == 'Patches':
            cmap_dict = {str(patch): patch_color_mapping[patch] for patch in unique_patches}
        elif selected_color == 'Entropy':
            entropy_values = {patch: patchwork.entropy_dict[patch]['avg'] for patch in unique_patches}
            cmap_dict = {str(patch): mplcol.to_hex(entropy_cmap(entropy_norm(entropy))) for patch, entropy in entropy_values.items()}
        
        heatmap =  hv.HeatMap(cell_infos, kdims=['x', 'y'], vdims=['patch', 'entropy']).opts(alpha = 0.5,
            cmap=cmap_dict, 
            xlim=(0, 1), ylim=(0, 1),
            width=600, height=600, tools=['hover'],
            xticks=axis_tick_labels,  # Custom x-axis tick labels
            yticks=axis_tick_labels   # Custom y-axis tick labels
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
        borders = hv.Segments(patch_borders).opts(color='black', line_width=3 / (1 + 0.02 * grid_resolution[0]**1.1)) #smaller lines for higher resolutions

        #Overlay 
        layers = [heatmap *borders] + [streamplots[control] for control in selected_controls if control in streamplots]

        return hv.Overlay(layers)

    # Interactive slider and selection widget
    slider = pn.widgets.IntSlider(name="Clustering Step", start=0, end=number_of_steps, step=1)
    vector_selector_streamplots = pn.widgets.MultiChoice(name="Select Controls for Streamplots", options=list(controls), value=[]) # Initially, value=[] means no vector fields are selected.
    color_selector = pn.widgets.Select(name= "Select Color Mapping", options = ['Patches', 'Entropy'])
    # Bind the function to the slider
    interactive_plot = pn.bind(get_current_patchwork, step=slider, selected_controls = vector_selector_streamplots, selected_color = color_selector)

    # Title for the widgetbox
    title = pn.pane.Markdown(f"<h1> Patchwork Visualization </h1> <br />  {title}")
    # Layout
    dashboard = pn.Row(pn.Column(title, pn.WidgetBox('## Patchwork Display Tools', slider, vector_selector_streamplots, color_selector)), interactive_plot)
    dashboard.servable() 
    dashboard.show()

    return
