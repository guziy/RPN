import os
from rpn.rpn import RPN

__author__ = 'huziy'

import numpy as np
import application_properties
import matplotlib.pyplot as plt
from permafrost import draw_regions
import matplotlib as mpl
from matplotlib import colors


def _plot_depth(data, lons2d, lats2d, basemap = None, clevels = None):
    if clevels is None:
        clevels = range(-2100, -300, 600) + range(-300, 0, 20) + range(0, 300, 20)
    cmap = mpl.cm.get_cmap(name="jet_r", lut = len(clevels))
    norm = colors.BoundaryNorm(clevels, cmap.N)

    x, y = basemap(lons2d, lats2d)
    mean_level = np.ma.masked_where(np.abs(data) < 1.0e-1, data)
    img = basemap.contourf(x, y, mean_level, norm = norm, levels = clevels, cmap = cmap)
    basemap.drawcoastlines()
    plt.colorbar(img)


def plot_initial_lake_depth(path = "data/from_guillimin/vary_lake_level1/pm1985010100_00000000p",
                            var_name = "CLDP", lons2d = None, lats2d = None, basemap = None
                            ):
    r = RPN(path)
    field = r.get_first_record_for_name(var_name)
    _plot_depth(field, lons2d, lats2d, basemap = basemap, clevels=xrange(0,310, 10))



def plot_mean(data_path = "", file_prefix = "pm"):
    """
    """
    var_name = "CLDP"
    means = []
    for file in os.listdir(data_path):
        if not file.startswith(file_prefix): continue
        file_path = os.path.join(data_path, file)

        r = RPN(file_path)
        levels = r.get_all_time_records_for_name(varname=var_name)
        means.append(np.mean( levels.values() , axis = 0))
    mean_level = np.array(means).mean(axis = 0)
    b, lons2d, lats2d = draw_regions.get_basemap_and_coords(
        file_path="data/from_guillimin/vary_lake_level1/pm1985010100_00000000p",
        lon1=-68, lat1=52, lon2=16.65, lat2=0
    )

    plt.figure()
    _plot_depth(mean_level, lons2d, lats2d, basemap=b)
    plt.savefig("lake_levels.png")

    plt.figure()
    plot_initial_lake_depth(lons2d=lons2d, lats2d=lats2d, basemap=b)
    plt.savefig("initial_lake_depth.png")



def main(data_path = "data/from_guillimin/vary_lake_level1"):
    plot_mean(data_path=data_path)
    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print "Hello world"
  