import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_2D_vector_field(vectorfield, bounds, resolution= 50, streamplot=True):
    """
    Plots a 2D vector field for a given ODE function.

    Args:
        ode_func (function): A function of the form ode_func(t, state).
        space (list of tuples): The space intervals for each dimension.
        streamplot (bool): If True, a streamplot is generated.

    Returns:
        None
    """
    X = np.linspace(bounds[0][0], bounds[0][1], resolution)
    Y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(X, Y)

    U, V = vectorfield(X, Y)

    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, U, V, color='b', alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Vector Field')
    plt.grid(True)

    if streamplot:
        plt.streamplot(X, Y, U, V, color='b', linewidth=0.7)

    plt.show()


def create_2D_vectorfield(X1, X2, control, solver):
    """
    Computes the vector field components (U, V) on a meshgrid (X1, X2).
    #TODO fix this also for models with more than 2 dimensions (by projection or something).
    #TODO make this more efficient by using vectorized operations, doesn't work yet.

    Parameters:
    X1, X2 (2D arrays): Meshgrid coordinates, assumed to be of the same dimension.
    control: control parameter, expected as an np.array
    solver: Solver class connected to a model

    Return:
    U, V: (2D arrays): Vector components at each grid point.

    """
    #TODO Check vectorized version

    nrows, ncols = X1.shape

    U = np.zeros_like(X1)  # Placeholder for dx/dt
    V = np.zeros_like(X2)  # Placeholder for dy/dt

    #TRY TO VECTORIZE IT... but doesn't work yet for higher resolutions
    #X = np.column_stack((X1.ravel(), X2.ravel()))
    #control_stacked = np.repeat(control, X.shape[0]) #here also check if control has 1 entry or n_samples many
    #derivatives = solver.solve_equilibrium(X, control_stacked)
    #U = derivatives[:,0].reshape(nrows, ncols)
    #V = derivatives[:,1].reshape(nrows, ncols)
    
    
    if True:
        # Loop through each point in the meshgrid
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                state = np.array([X1[i, j], X2[i, j]])  # Current state vector
                state = np.atleast_2d(state)
                derivatives = solver.get_derivative(state, control).ravel()    # Evaluate 
                U[i, j] = derivatives[0]            # dx/dt
                V[i, j] = derivatives[1]            # dy/dt

    return U, V



def plot_2D_vector_field_over_grid(grid, 
                                   solver, 
                                   control, 
                                   dim1 = 0, dim2 = 1, 
                                   resolution=51,
                                   streamplot_colour = 'darkblue',
                                   vectorfield_colour = 'mediumblue', 
                                   title='2D Vector Field', 
                                   ax = None,
                                   save_to = None, 
                                   display_vectorfield=True, 
                                   display_grid=True, 
                                   display_vectorfield_magnitude=False):
    """
    Plots a 2D vector field over a specified grid.
    
    Parameters:
        grid (object): The grid object containing the bounds and transformation information.
        solver (object): The solver object containing the model. This will be used to compute the vector field.
        control (np.array): The control vector for the vectorfield. Has to be given in the form of np.array([control]). 
                            #TODO should this be adaptable for different controls corresponding to different points?

        dim1 (int, optional): The first dimension to plot. Defaults to 0.
        dim2 (int, optional): The second dimension to plot. Defaults to 1.
        resolution (int, optional): The resolution of the meshgrid for the vectorfield. Defaults to 51 (number_cells + 1).
        title (str, optional): The title of the plot. Defaults to '2D Vector Field'.
        save_to (str, optional): The path where to save the plot.
        display_vectorfield (bool, optional): Whether to display the vector field using quiver. Defaults to True.
        display_grid (bool, optional): Whether to display the grid. Defaults to True.
        display_vectorfield_magnitude (bool, optional): Whether to display the magnitude of the vector field. Defaults to False.
    
    Returns:
        fig, ax
    """
        
    bounds = [grid.tf_bounds[dim1], grid.tf_bounds[dim2]]
    X1lin = np.linspace(bounds[0][0], bounds[0][1], resolution)
    X2lin = np.linspace(bounds[1][0], bounds[1][1], resolution)
    
    if ax == None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if grid.transformed_bool:
        
        #Remove boundaries for infinity case (transformation doesn't work herd)
        if np.isinf(grid.bounds[dim1][1]):
            X1lin = X1lin[:-1]
        if np.isinf(grid.bounds[dim1][0]):
            X1lin = X1lin[1:]
        if np.isinf(grid.bounds[dim2][1]):
            X2lin = X2lin[:-1]
        if np.isinf(grid.bounds[dim2][0]):
            X2lin = X2lin[1:]
        
        X1, X2 = np.meshgrid(X1lin, X2lin)
        X1_org = np.vectorize(grid.inverse_transformations[dim1])(X1)
        X2_org = np.vectorize(grid.inverse_transformations[dim2])(X2)
    
        U_org, V_org = create_2D_vectorfield(X1_org, X2_org, control=control, solver=solver)
        
        #Transform the vectorfield, multiply with Jacobian of transformation
        U = np.multiply(np.vectorize(grid.transformation_derivatives[dim1])(X1_org), U_org)
        V = np.multiply(np.vectorize(grid.transformation_derivatives[dim2])(X2_org), V_org)

    else:
        
        X1, X2 = np.meshgrid(X1lin, X2lin)
        U, V = create_2D_vectorfield(X1, X2, control=control, solver=solver)
    
    
    ax.streamplot(X1, X2, U, V, color=streamplot_colour, linewidth=0.7, density=1, arrowsize=0.8)
    

    if display_vectorfield:
        ax.quiver(X1, X2, U, V, color=vectorfield_colour, alpha=0.2)

    if display_vectorfield_magnitude:
        # Add a color mesh to visualize the magnitude of the vector field
        magnitude = np.sqrt(U**2 + V**2)
        contour = ax.contourf(X1, X2, magnitude, levels=50, cmap='viridis', alpha=0.6)
        fig.colorbar(contour, ax=ax, label='Vectorfield Magnitude')

    if grid.transformed_bool:
        #Set ticklabels according to original space
        xtick_pos = np.linspace(bounds[0][0], bounds[0][1], 9)
        xlabels = [round(grid.inverse_transformations[dim1](tick),1) for tick in xtick_pos]
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xlabels)

        ytick_pos = np.linspace(bounds[1][0], bounds[1][1], 9)
        ylabels = [round(grid.inverse_transformations[dim1](tick),2) for tick in ytick_pos]
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ylabels)

    ax.set_xlabel('X1-axis')
    ax.set_ylabel('X2-axis')
    ax.set_title(title, pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('data', bounds[0][0]))
    ax.spines['left'].set_position(('data', bounds[1][0]))
    ax.set_xlim(left=bounds[0][0])
    ax.set_ylim(bottom=bounds[1][0])
    if display_grid:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    if save_to != None:
        fig.savefig(save_to)

  
    return 


if __name__ == '__main__':
    pass

