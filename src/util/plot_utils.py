from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

__author__="huziy"
__date__ ="$Aug 17, 2011 3:49:28 PM$"


import numpy as np

def draw_colorbar(fig, img, ax = None, boundaries = None, ticks = None):
    """
    Draw a nicely aligned colorbar to the axes ax
    fig - containing figure
    img - is the result returned from contourf or pcolormesh
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = fig.colorbar(img, cax = cax, boundaries = boundaries, ticks = ticks)


def get_closest_tick_value(nticks, lower_limit):
    """
    nticks - number of ticks in the colorbar
    lower_limit - is the lower limit of the data to plot [0..1]
    """

    assert 0 <= lower_limit <= 1
    d = 1.0 / float(nticks - 1.0)
    assert d > 0

    tick_value = 0
    while tick_value <= 1:
        if tick_value <= lower_limit <= tick_value + d:
            if lower_limit - tick_value < tick_value + d - lower_limit:
                return tick_value
            else:
                return tick_value + d
        tick_value += d


def apply_plot_params(font_size = 20, width_pt = 1000, aspect_ratio = 1, height_cm = None, width_cm = None):
    """
    aspect_ratio = height / (width * golden_mean)
    """
    import pylab
    import math

    if width_pt is not None and width_cm is None:
        inches_per_pt = 1.0 / 72.27               # Convert pt to inch
        golden_mean = (math.sqrt(5.0) - 1.0) / 2.0       # Aesthetic ratio
        fig_width = width_pt * inches_per_pt          # width in inches
        fig_height = fig_width * golden_mean      # height in inches
        fig_size = [fig_width, aspect_ratio * fig_height]
    else:
        inches_per_cm = 1.0 / 2.54
        width_cm = 16.0 if width_cm is None else width_cm
        height_cm = 23.0 if height_cm is None else height_cm
        fig_size = [width_cm * inches_per_cm, height_cm * inches_per_cm ]

    params = {
        'axes.labelsize': font_size,
        'font.size': font_size,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'figure.figsize': fig_size,
        "axes.titlesize": font_size
    }

    pylab.rcParams.update(params)


def zoom_to_qc(plotter = None):
    ymin, ymax = plotter.ylim()
    plotter.ylim(ymin + 0.05 * (ymax - ymin) , ymax * 0.25)

    xmin, xmax = plotter.xlim()
    plotter.xlim(xmin + (xmax - xmin) * 0.55, 0.72*xmax)


def draw_meridians_and_parallels(the_basemap, step_degrees = 5.0, ax = None):
    meridians = np.arange(-180,180, step_degrees)
    parallels = np.arange(-90,90, step_degrees)
    the_basemap.drawparallels(parallels,labels=[0,0,0,0],fontsize=16,linewidth=0.25, ax = ax)
    the_basemap.drawmeridians(meridians,labels=[0,0,0,0],fontsize=16,linewidth=0.25, ax = ax)

def get_ranges(x_interest, y_interest):
    """
    Get region of zoom for a given map
    """
    x_min, x_max = np.min( x_interest ), np.max( x_interest )
    y_min, y_max = np.min( y_interest ), np.max( y_interest )
    dx = 0.1 * ( x_max - x_min )
    dy = 0.1 * ( y_max - y_min )
    return x_min - dx, x_max + dx, y_min - dy, y_max + dy



if __name__ == "__main__":
    print "Hello World"
