from brewer2mpl import brewer2mpl
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import numpy as np

__author__ = 'huziy'

_varname_to_units = {
    "STFL": "${\\rm m^3/s}$",
    "TT": "${\\rm ^{\circ}C}$",
    "PR": "${\\rm mm/day}$",
    "DEPTH_TO_BEDROCK": "${\\rm m}$",
    "SAND":"${\\rm \\%}$",
    "CLAY":"${\\rm \\%}$",
    "AV" : "${\\rm W/m^2}$",
    "AH" : "${\\rm W/m^2}$"
}

_varname_to_long_name = {
    "STFL": "Streamflow",
    "TT": "Temperature",
    "PR": "Total precipitation",
    "DEPTH_TO_BEDROCK" : "Depth to bedrock",
    "lake_fraction": "Lake fraction",
    "SAND" : "Soil fraction of sand",
    "CLAY" : "Soil fraction of clay",
    "AV": "Latent heat flux at the surface",
    "AH": "Sensible heat flux at the surface"
}


def get_colorbar_formatter(varname):
    if varname == "STFL":
        return None
    else:
        #format the colorbar tick labels
        sfmt = ScalarFormatter(useMathText=True)
        sfmt.set_powerlimits((-3, 3))
        return sfmt



def get_units(var_name):
    if var_name not in _varname_to_units:
        return ""
    return _varname_to_units[var_name.upper()]

def get_long_name(var_name):
    if var_name not in _varname_to_long_name:
        return ""
    return _varname_to_long_name[var_name]



def get_boundary_norm_using_all_vals(to_plot, ncolors):
    vmin = np.percentile(to_plot[~to_plot.mask], 5)
    vmax = np.percentile(to_plot[~to_plot.mask], 95)
    med = np.median(to_plot[~to_plot.mask])
    locator = MaxNLocator(ncolors)
    bounds = locator.tick_values(vmin, vmax)

    return BoundaryNorm(bounds, ncolors=ncolors), bounds, bounds[0], bounds[-1]


def get_boundary_norm(vmin, vmax, ncolors, exclude_zero = False):


    if vmin * vmax >= 0:
        locator = MaxNLocator(ncolors)
        bounds = np.asarray(locator.tick_values(vmin, vmax))
    elif exclude_zero:
        locator = MaxNLocator(nbins=ncolors - 1, symmetric = True)
        bounds = np.asarray(locator.tick_values(vmin, vmax))
        bounds = bounds[bounds != 0]
    else:
        locator = MaxNLocator(nbins=ncolors)
        bounds = np.asarray(locator.tick_values(vmin, vmax))




    return BoundaryNorm(bounds, ncolors=ncolors), bounds, bounds[0], bounds[-1]



def get_colormap_and_norm_for(var_name, to_plot = None, ncolors = 10, vmin = None, vmax = None):
    """
    If vmin or vmax is None then to_plot parameter is required
    :param var_name:
    :param ncolors: Number of discrete colors in the colorbar, try to take a good number like 10, 5, ...
    :return:
    """
    if None in [vmin, vmax]:
        vmin, vmax = to_plot.min(), to_plot.max()


    if var_name == "STFL":
        cmap = brewer2mpl.get_map("Blues", "sequential", 9).get_mpl_colormap(N = ncolors)
        pmin = np.floor(np.log10(vmin))
        pmax = np.ceil(np.log10(vmax))
        d = (pmax - pmin) / ncolors
        nice_min = 10 ** pmin
        nice_max = 10 ** (pmin + ncolors * d)
        norm = LogNorm(vmin=nice_min, vmax=nice_max)
    else:
        if var_name in ["PR"]:
            reverse = False
        else:
            reverse = True
        cmap = brewer2mpl.get_map("spectral", "diverging", 9, reverse = reverse).get_mpl_colormap(N = ncolors)
        #norm, bounds, vmin_nice, vmax_nice = get_boundary_norm_using_all_vals(to_plot, ncolors)
        locator = MaxNLocator(ncolors)
        norm = BoundaryNorm(locator.tick_values(vmin, vmax), ncolors)

    return cmap, norm



#the fraction of a grid cell taken by lake, startting from which the lake is
#treated as global
GLOBAL_LAKE_FRACTION = 0.6