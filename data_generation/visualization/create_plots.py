import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("../simulations")

# TODO make this import better!!!

from grid import Grid




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



# After further testing I will implement a further function here to create a plot of a model on a grid




if __name__ == '__main__':
    pass

