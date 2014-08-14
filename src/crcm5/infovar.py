from brewer2mpl import brewer2mpl
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from mpl_toolkits.basemap import maskoceans
import numpy as np

__author__ = 'huziy'

_varname_to_units = {
    "STFL": "${\\rm m^3/s}$",
    "TT": "${\\rm ^{\circ}C}$",
    "PR": "${\\rm mm/day}$",
    "DEPTH_TO_BEDROCK": "${\\rm m}$",
    "SAND": "${\\rm \\%}$",
    "CLAY": "${\\rm \\%}$",
    "AV": "${\\rm W/m^2}$",
    "AH": "${\\rm W/m^2}$"
}


_varname_to_label = {
    "I0": r"$T_{\rm soil}$",
    "I1": r"$\theta_{\rm liq}$",
    "I2": r"$\theta_{\rm ice}$",
    "I1+I2": r"$\theta_{\rm liq} + \theta_{\rm ice}$"
}


def get_display_label_for_var(varname):
    return _varname_to_label.get(varname, varname)



_varname_to_long_name = {
    "STFL": "Streamflow",
    "TT": "Temperature",
    "PR": "Total precipitation",
    "DEPTH_TO_BEDROCK": "Depth to bedrock",
    "lake_fraction": "Lake fraction",
    "SAND": "Soil fraction of sand",
    "CLAY": "Soil fraction of clay",
    "AV": "Latent heat flux at the surface",
    "AH": "Sensible heat flux at the surface"
}

#Names of the variables in the hdf file
HDF_VERT_SOIL_HYDR_COND_NAME = "soil_hydraulic_conductivity"
HDF_FLOW_DIRECTIONS_NAME = "flow_direction"
HDF_ACCUMULATION_AREA_NAME = "accumulation_area_km2"
HDF_CELL_AREA_NAME = "cell_area_m2"
HDF_LAKE_FRACTION_NAME = "lake_fraction"
HDF_DEPTH_TO_BEDROCK_NAME = "depth_to_bedrock"
HDF_SOIL_ANISOTROPY_RATIO_NAME = "soil_anisotropy_ratio"
HDF_SLOPE_NAME = "slope"

hdf_varname_to_description = {
    HDF_FLOW_DIRECTIONS_NAME: "flow directions in the format 1,2,4,8,16,32,64,128",
    HDF_ACCUMULATION_AREA_NAME: "flow accumulation area in km**2",
    HDF_SLOPE_NAME: "Channel slope of a river, non dimensional value",
    HDF_CELL_AREA_NAME: "Area of a grid cell in m^2",
    "sand": "sand percentage in soil 3d field sand(level, x, y)",
    "clay": "",
    HDF_DEPTH_TO_BEDROCK_NAME: "",
    HDF_LAKE_FRACTION_NAME: "",
    "drainage_density_inv_meters": "",
    "soil_hydraulic_conductivity": "",
    HDF_SOIL_ANISOTROPY_RATIO_NAME: "Soil anisotropy ratio Kh/Kv",
    "interflow_c_coef": ""

}

hdf_varname_to_rpn_varname = {
    "soil_anisotropy_ratio": "SANI",
    HDF_VERT_SOIL_HYDR_COND_NAME: "HT"
}

rpn_varname_to_hdf_varname = {}

soil_layer_widths_26_to_60 = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                         1.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])



def get_to_plot(varname, data, lake_fraction=None,
                mask_oceans=True, lons=None, lats=None, difference = False):
    if mask_oceans:
        assert lons is not None and lats is not None

    #This one is used if something is to be masked or changed before plotting
    if varname in ["STFL", "STFA"]:

        if lake_fraction is None or np.sum(lake_fraction) <= 0.01:
            #data1 = np.ma.masked_where(data < 0, data) if not difference else data
            return maskoceans(lonsin=lons, latsin=lats, datain=data)
        else:
            data1 = np.ma.masked_where(lake_fraction >= GLOBAL_LAKE_FRACTION, data)
    elif varname == "PR":
        data1 = data * 24 * 60 * 60 * 1000  # convert m/s to mm/day

    elif varname == "I0":
        data1 = data - 273.15  # convert to deg C
        data1 = np.ma.masked_where((data <= 0.1) | (lake_fraction >= GLOBAL_LAKE_FRACTION), data1)
        data1 = maskoceans(lonsin=lons, latsin=lats, datain=data1)
        return data1
    elif varname in ["TRAF", "TDRA"]:
        data1 = data * 24 * 60 * 60  # convert mm/s to mm/day
        if varname == "TDRA":
            return maskoceans(lonsin=lons, latsin=lats, datain=data1, inlands=True)
    elif varname in ["I1", "IMAV", "I5"]:
        return maskoceans(lonsin=lons, latsin=lats, datain=data, inlands=True)
    else:
        data1 = data

    if mask_oceans:
        return maskoceans(lonsin=lons, latsin=lats, datain=data1, inlands=False)
    return data1


def get_colorbar_formatter(varname):
    if varname in ["STFL", "STFA"]:
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


def get_boundary_norm(vmin, vmax, ncolors, exclude_zero=False, varname = None, difference = False):



    if vmin * vmax >= 0:
        locator = MaxNLocator(ncolors)
        bounds = np.asarray(locator.tick_values(vmin, vmax))
    elif exclude_zero:
        #implies that this is the case for difference
        delta = max(abs(vmax), abs(vmin))
        assert ncolors % 2 == 1
        d = 2.0 * delta / float(ncolors)
        print d, np.log10(d)
        ndec = -int(np.floor(np.log10(d)))
        print ndec
        d = np.round(d, decimals=ndec)

        assert d > 0
        print "ncolors = {0}".format(ncolors)

        negats = [-d / 2.0 - d * i for i in range((ncolors - 1) / 2)]
        bounds = negats[::-1] + [-the_bound for the_bound in negats]
        assert 0 not in bounds
        assert bounds[0] == -bounds[-1]
    else:
        locator = MaxNLocator(nbins=ncolors, symmetric=True)
        bounds = np.asarray(locator.tick_values(vmin, vmax))


    #custom variable norms
    if difference:
        pass
    else:
        if varname == "TRAF":
            bounds = [0, 0.1, 0.5] + list(range(1, ncolors - 3)) + [vmax, ]




    return BoundaryNorm(bounds, ncolors=ncolors), bounds, bounds[0], bounds[-1]


def get_colormap_and_norm_for(var_name, to_plot=None, ncolors=10, vmin=None, vmax=None):
    """
    If vmin or vmax is None then to_plot parameter is required
    :param var_name:
    :param ncolors: Number of discrete colors in the colorbar, try to take a good number like 10, 5, ...
    :return:

    Note: when `var_name` is STFL, the parameter ncolors is ignored

    """
    if None in [vmin, vmax]:
        vmin, vmax = to_plot.min(), to_plot.max()

    if var_name in ["STFL", "STFA"]:
        upper = 1000
        bounds = [0, 100, 200, 500, 1000]
        while upper <= vmax:
            upper += 1000
            bounds.append(upper)
        ncolors = len(bounds) - 1

        cmap = brewer2mpl.get_map("Blues", "sequential", 9).get_mpl_colormap(N=ncolors)
        norm = BoundaryNorm(bounds, ncolors=ncolors)  # LogNorm(vmin=10 ** (pmax - ncolors), vmax=10 ** pmax)
    else:
        if var_name in ["PR"]:
            reverse = False
        else:
            reverse = True
        cmap = brewer2mpl.get_map("spectral", "diverging", 9, reverse=reverse).get_mpl_colormap(N=ncolors)
        #norm, bounds, vmin_nice, vmax_nice = get_boundary_norm_using_all_vals(to_plot, ncolors)
        locator = MaxNLocator(ncolors)
        norm = BoundaryNorm(locator.tick_values(vmin, vmax), ncolors)

    return cmap, norm


#the fraction of a grid cell taken by lake, startting from which the lake is
#treated as global
GLOBAL_LAKE_FRACTION = 0.6