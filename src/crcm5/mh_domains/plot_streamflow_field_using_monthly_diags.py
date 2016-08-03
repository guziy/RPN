import calendar
from collections import OrderedDict

from pathlib import Path

from matplotlib import cm
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.gridspec import GridSpec
from rpn.domains.rotated_lat_lon import RotatedLatLon

from application_properties import main_decorator
from crcm5.model_data import Crcm5ModelDataManager
import matplotlib.pyplot as plt
import numpy as np

from rpn.rpn_multi import MultiRPN
from crcm5.mh_domains import default_domains
from util import plot_utils

img_folder = "mh"


def plot_monthly_clim_in_a_panel(months=None, diag_folder="", vname="STFL",
                                 grid_config=None, basins_of_interest_shp=""):
    """
    Plots climatologies using diagnostics outputs, not samples
    :param months:
    :param diag_folder:
    :param vname:
    """
    if months is None:
        months = list(range(1, 13))


    diag_path = Path(diag_folder)
    month_to_field = OrderedDict()

    lons, lats, bmp = None, None, None
    data_mask = None

    for m in months:

        r = MultiRPN(str(diag_path.joinpath("*{:02d}".format(m)).joinpath("pm*_moyenne")))

        date_to_field = r.get_all_time_records_for_name_and_level()

        the_mean = np.mean([f for f in date_to_field.values()], axis=0)

        the_mean = np.ma.masked_where(the_mean < 0, the_mean)

        month_to_field[m] = the_mean

        if bmp is None:
            lons, lats = r.get_longitudes_and_latitudes_of_the_last_read_rec()

            # get the basemap object
            bmp, data_mask = grid_config.get_basemap_using_shape_with_polygons_of_interest(
                lons, lats, shp_path=basins_of_interest_shp, mask_margin=5)

        r.close()



    fig = plt.figure()


    ncols = 3
    nrows = len(months) // ncols + int(len(months) % ncols != 0)

    gs = GridSpec(nrows=nrows, ncols=ncols + 1)


    xx, yy = bmp(lons, lats)

    clevs = [0, 20, 50, 100, 200, 500, 1000, 1500, 3000, 4500, 5000, 7000, 9000]
    bn = BoundaryNorm(clevs, len(clevs) - 1)
    cmap = cm.get_cmap("jet", len(clevs) - 1)
    for m, field in month_to_field.items():
        row = (m - 1) // ncols
        col = (m - 1) % ncols

        ax = fig.add_subplot(gs[row, col])
        ax.set_title(calendar.month_name[m])

        to_plot = np.ma.masked_where(~data_mask, field)
        im = bmp.pcolormesh(xx, yy, to_plot, norm=bn, cmap=cmap, vmin=clevs[0], vmax=clevs[-1])
        bmp.colorbar(im, extend="max")

        bmp.readshapefile(basins_of_interest_shp[:-4], "basins", linewidth=2, color="m", ax=ax)
        bmp.drawcoastlines(ax=ax)


    plt.close(fig)

    # plot annual mean
    ann_mean = np.mean([field for m, field in month_to_field.items()], axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(gs[:, :])
    ax.set_title("Annual mean")

    to_plot = np.ma.masked_where(~data_mask, ann_mean)
    im = bmp.pcolormesh(xx, yy, to_plot, norm=bn, cmap=cmap, vmin=clevs[0], vmax=clevs[-1])
    bmp.colorbar(im, extend="max")

    bmp.readshapefile(basins_of_interest_shp[:-4], "basins", linewidth=2, color="m", ax=ax)
    bmp.drawcoastlines(ax=ax)

    plt.show()
    plt.close(fig)


@main_decorator
def main():

    plot_utils.apply_plot_params()
    diag_folder = "/RECH2/huziy/BC-MH/bc_mh_044deg/Diagnostics"

    gc = default_domains.bc_mh_044
    plot_monthly_clim_in_a_panel(diag_folder=diag_folder, grid_config=gc, basins_of_interest_shp=default_domains.MH_BASINS_PATH)



if __name__ == '__main__':
    main()