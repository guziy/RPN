import os
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from rpn.rpn import RPN

__author__ = 'huziy'

import numpy as np
import application_properties
import matplotlib.pyplot as plt
from permafrost import draw_regions
import matplotlib as mpl
from matplotlib import colors
from matplotlib import cm

def _plot_depth(data, lons2d, lats2d, basemap = None,
                clevels = None, lowest_value = 0.1,
                ax = None):
    if clevels is None:
        clevels = list(range(-2100, -300, 600)) + list(range(-300, 0, 20)) + list(range(0, 300, 20))
    cmap = cm.get_cmap(name="jet_r", lut = len(clevels))
    norm = colors.BoundaryNorm(clevels, cmap.N)

    x, y = basemap(lons2d, lats2d)
    if lowest_value is not None:
        mean_level = np.ma.masked_where(np.abs(data) < lowest_value, data)
    else:
        mean_level = data
    #img = basemap.contourf(x, y, mean_level, norm = norm, levels = clevels, cmap = cmap)
    img = basemap.pcolormesh(x,y, mean_level, norm = norm, cmap = cmap, ax = ax)
    basemap.drawcoastlines(ax=ax)

    if ax is None:
        ax = plt.gca()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(img, ax = ax, cax = cax)


def plot_initial_lake_depth(path = "data/from_guillimin/vary_lake_level1/pm1985010100_00000000p",
                            var_name = "CLDP", lons2d = None, lats2d = None, basemap = None
                            ):
    """
    returns initial lake depth field
    """
    r = RPN(path)
    field = r.get_first_record_for_name(var_name)
    r.close()
    _plot_depth(field, lons2d, lats2d, basemap = basemap, clevels=range(0,310, 10))
    return field

def plot_lake_fraction(path = "data/from_guillimin/vary_lake_level1/pm1985010100_00000000p",
                       var_name = "LF1", lons2d = None, lats2d = None, basemap = None):
    r = RPN(path)
    field = r.get_first_record_for_name(var_name)
    r.close()
    _plot_depth(field, lons2d, lats2d, basemap = basemap,
        clevels=np.arange(0, 1.1, 0.1), lowest_value=0.001)

    pass


def _get_seasonal_mean_anomaly(data, the_mean, months = None):

    """
    :type data: dict
    """
    sitems = sorted( list(data.items()), key = lambda x: x[0])
    times = [x[0] for x in sitems]
    values = [x[1] for x in sitems]

    bool_vector = [x.month in months for x in times]
    bool_vector = np.array(bool_vector)
    values = np.array(values)
    return np.mean(values[bool_vector,:,:], axis = 0) - the_mean



    pass

def plot_mean(data_path = "", file_prefix = "pm"):
    """
    """
    var_name = "CLDP"
    means = []
    means_dict = {}
    for file in os.listdir(data_path):
        if not file.startswith(file_prefix): continue
        file_path = os.path.join(data_path, file)

        r = RPN(file_path)
        levels = r.get_all_time_records_for_name(varname=var_name)
        means_dict.update(levels)
        means.append(np.mean( list(levels.values()) , axis = 0))
    mean_level = np.array(means).mean(axis = 0)
    b, lons2d, lats2d = draw_regions.get_basemap_and_coords(
        file_path="data/from_guillimin/vary_lake_level1/pm1985010100_00000000p",
        lon1=-68, lat1=52, lon2=16.65, lat2=0
    )

    std_level = np.std( np.array(means), axis=0)


    #plot mean
    plt.figure()
    _plot_depth(mean_level, lons2d, lats2d, basemap=b, clevels=range(0,310, 10))
    plt.savefig("mean_lake_levels.png")


    #plot std
    plt.figure()
    _plot_depth(std_level, lons2d, lats2d, basemap=b, clevels=np.arange(0,2.2, 0.2), lowest_value=1e-3)
    plt.savefig("std_lake_levels.png")



    #plot initial lake depth
    plt.figure()
    plot_initial_lake_depth(lons2d=lons2d, lats2d=lats2d, basemap=b)
    plt.savefig("initial_lake_depth.png")

    #plot lake fraction
    plt.figure()
    plot_lake_fraction(lons2d=lons2d, lats2d=lats2d, basemap=b)
    plt.savefig("lake_fraction.png")

    fig = plt.figure()
    assert isinstance(fig, Figure)
    gs = GridSpec(2,2)

    seasons = ["(a) Winter (DJF)", "(b) Spring (MAM)", "(c) Summer (JJA)", "(d) Fall (SON)"]
    months_list = [[12,1,2], [3,4,5], [6,7,8], [9,10,11]]

    i = 0
    for season, months in zip(seasons, months_list):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        assert isinstance(ax, Axes)
        data = _get_seasonal_mean_anomaly(means_dict, mean_level, months=months)
        the_max = np.round(np.max(np.abs(data)) * 10 ) / 10.0

        data = np.ma.masked_where(mean_level < 0.1, data)

        _plot_depth(data, lons2d, lats2d, ax = ax, basemap=b,
            clevels=np.arange(-0.5, 0.6, 0.1),
            lowest_value=0.001
        )
        ax.set_title(season)
        i += 1
    fig.tight_layout()
    fig.savefig("seasonal_anomalies.png")



def main(data_path = "data/from_guillimin/vary_lake_level1"):
    plot_mean(data_path=data_path)
    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print("Hello world")
  