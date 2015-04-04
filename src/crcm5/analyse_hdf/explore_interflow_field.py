from collections import OrderedDict
import os
import brewer2mpl
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.basemap import maskoceans
import numpy as np
from crcm5 import infovar
from . import common_plot_params as cpp

__author__ = 'huziy'


#This script is for exploring a given field


default_seasons = OrderedDict([
    ("Winter", [12, 1, 2]),
    ("Spring", list(range(3, 6))),
    ("Summer", list(range(6, 9))),
    ("Autumn", list(range(9, 12)))
])

from . import do_analysis_using_pytables as analysis

images_folder = "images_for_lake-river_paper/seasonal_intfl"


#noinspection PyNoneFunctionAssignment
def calculate_and_plot_seasonal_means(seasons=default_seasons,
                                      hdf_path="", start_year=None, end_year=None):
    lon, lat, basemap = analysis.get_basemap_from_hdf(file_path=hdf_path)
    x, y = basemap(lon, lat)

    dates, levs, data = analysis.get_daily_climatology_of_3d_field(path_to_hdf_file=hdf_path, var_name="INTF",
                                                                   start_year=start_year, end_year=end_year)

    acc_area = analysis.get_array_from_file(path = hdf_path, var_name=infovar.HDF_ACCUMULATION_AREA_NAME)

    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.1], wspace=0.25, hspace=0)
    fig = plt.figure()


    season_to_field = OrderedDict()
    nlayers = 4
    soil_layer_depths = infovar.soil_layer_widths_26_to_60
    print("max depth is {0}".format(np.sum(soil_layer_depths[:nlayers])))
    vmin, vmax = None, None
    for season, months in seasons.items():
        field = np.mean([z for z, date in zip(data, dates) if date.month in list(months)], axis = 0)

        field = np.sum(field[:nlayers, :, :], axis=0)
        field *= 24 * 60 * 60
        field = np.ma.masked_where(acc_area <= 0, field)
        season_to_field[season] = field
        if vmin is None:
            vmin, vmax = np.percentile(field, 5), np.percentile(field, 95)
        else:
            vmin = min(vmin, np.percentile(field, 5))
            vmax = max(vmax, np.percentile(field, 95))

    levels = np.linspace(vmin, vmax, 11)
    levels = np.round(levels, decimals=6)

    im = None
    i = 0
    cmap = brewer2mpl.get_map("spectral", "diverging", 9, reverse=True).get_mpl_colormap(N=len(levels) - 1)
    bn = BoundaryNorm(levels, len(levels) - 1)
    for season, field in season_to_field.items():
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.set_title(season)


        field = maskoceans(lon, lat, field)
        im = basemap.pcolormesh(x, y, field, ax = ax, cmap = cmap, norm = bn, vmin = vmin, vmax = vmax)
        #basemap.colorbar(im, ax = ax)
        basemap.drawcoastlines(linewidth=cpp.COASTLINE_WIDTH, ax=ax)
        i += 1


    fig.suptitle("mm/day, in upper {0} m of soil".format(np.sum(soil_layer_depths[:nlayers])))
    cax = fig.add_subplot(gs[:, 2])
    fmt = ScalarFormatter(useMathText=True, useOffset=False)
    fmt.set_powerlimits([-2, 3])
    plt.colorbar(im, cax=cax, format = fmt)
    assert isinstance(cax, Axes)
    cax.yaxis.get_offset_text().set_position((-1, 0))


    if not os.path.isdir(images_folder):
        os.mkdir(images_folder)

    ff_name = os.path.basename(hdf_path)
    f_path = os.path.join(images_folder, "{0}.jpeg".format(ff_name))
    fig.savefig(f_path, dpi = cpp.FIG_SAVE_DPI, bbox_inches = "tight")


if __name__ == "__main__":
    pass