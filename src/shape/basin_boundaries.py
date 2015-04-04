__author__="huziy"
__date__ ="$20 dec. 2010 19:59:26$"


from .read_shape_file import *
def plot_basin_boundaries_from_shape(basemap, plotter = None, axes = None, linewidth = 2, edge_color = 'k'):

    if axes is None:
        ax = plotter.gca()
    else:
        ax = axes
    for poly in get_features_from_shape(basemap, linewidth = linewidth, edgecolor = edge_color):
        ax.add_patch(poly)

    pass

#plotter = matplotlib.pyplot
def plot_patches(plotter, the_patches):
    ax = plotter.gca()
    for patch in the_patches:
        ax.add_patch(patch)



if __name__ == "__main__":
    print("Hello World")
