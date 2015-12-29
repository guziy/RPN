from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap
from rpn.domains.rotated_lat_lon import RotatedLatLon
from rpn.rpn import RPN
from application_properties import main_decorator
import numpy as np
from rpn.rpn_multi import MultiRPN

from my_colormaps import get_cmap_from_ncl_spec_file
from util import plot_utils
from mpl_toolkits.basemap import cm as bcmap

BOW_RIVER_SHP = "/RESCUE/skynet3_rech1/huziy/CNRCWP/Calgary_flood/bow_river/bow_projected.shp"


def get_bow_river_basin_mask(
        path="/RESCUE/skynet3_rech1/huziy/CNRCWP/Calgary_flood/Bow_river_basin_mask_NA_0.11deg.rpn"):
    r = RPN(path)
    msk = r.get_first_record_for_name("FMSK")
    r.close()
    return msk


@main_decorator
def main(plot_vals = False):
    varname = "RFAC"
    multiplier = 24 * 3600
    data_folder = Path("/home/huziy/skynet3_rech1/CNRCWP/Calgary_flood/atm_data_for_Arman_simulations")
    day_range = range(19, 22)
    dates_of_interest = [datetime(2013, 6, d) for d in day_range]

    img_folder = Path("calgary_flood/2D")

    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)

    lons, lats, bmp = None, None, None

    the_mask = get_bow_river_basin_mask()

    i_list, j_list = np.where(the_mask > 0.5)
    imin, imax = i_list.min() - 2, i_list.max() + 5
    jmin, jmax = j_list.min() - 2, j_list.max() + 5

    # Calculate daily means
    sim_label_to_date_to_mean = OrderedDict()
    for sim_dir in data_folder.iterdir():
        mr = MultiRPN(str(sim_dir.joinpath("pm*")))
        print(str(sim_dir))
        print(mr.get_number_of_records())

        label = sim_dir.name.split("_")[-2].replace("NoDrain", "").replace("frozen", "Frozen").replace("Bow", "")
        sim_label_to_date_to_mean[label] = OrderedDict()

        data = mr.get_4d_field(varname)
        data = {d: list(v.items())[0][1] for d, v in data.items() if d.day in day_range}

        for d in dates_of_interest:
            sim_label_to_date_to_mean[label][d] = np.array(
                [field for d1, field in data.items() if d1.day == d.day]).mean(axis=0) * multiplier

        if lons is None:
            lons, lats = mr.get_longitudes_and_latitudes_of_the_last_read_rec()
            for f in sim_dir.iterdir():
                if f.name.startswith("pm"):
                    r = RPN(str(f))
                    r.get_first_record_for_name(varname=varname)
                    rll = RotatedLatLon(**r.get_proj_parameters_for_the_last_read_rec())
                    bmp = rll.get_basemap_object_for_lons_lats(lons2d=lons[imin:imax, jmin:jmax],
                                                               lats2d=lats[imin:imax, jmin:jmax], resolution="i")
                    r.close()
                    break

        mr.close()

    # reorder simulations
    sim_label_to_date_to_mean = OrderedDict(
        [(k, sim_label_to_date_to_mean[k]) for k in
         sorted(sim_label_to_date_to_mean, key=lambda z: len(z), reverse=True)])

    key_list = [k for k in sim_label_to_date_to_mean]
    key_list[-2], key_list[-1] = key_list[-1], key_list[-2]
    sim_label_to_date_to_mean = OrderedDict(
        [(k, sim_label_to_date_to_mean[k]) for k in key_list])

    # do the plots (subplots: vertically - simulations, horizontally - days)
    plot_utils.apply_plot_params(width_cm=24, height_cm=28, font_size=10)
    fig = plt.figure()
    nrows = len(sim_label_to_date_to_mean)
    ncols = len(day_range)
    gs = GridSpec(nrows, ncols, wspace=0, hspace=0.1)

    clevs_vals = [0, 0.1, 20, 50, 100, 150, 200, 300]

    base_label = None
    clevs_diff = [1, 5, 10, 20, 50, 100]
    clevs_diff = [-c for c in reversed(clevs_diff)] + [0] + clevs_diff
    cmap_diff = cm.get_cmap("bwr", len(clevs_diff) - 1)
    cmap_vals = get_cmap_from_ncl_spec_file(path="colormap_files/precip3_16lev.rgb",
                                            ncolors=len(clevs_vals) - 1)
    xx, yy = bmp(lons, lats)
    title = "Total runoff ({}, mm/day)".format(varname)
    fig.suptitle(title)

    for row, (sim_label, date_to_field) in enumerate(sim_label_to_date_to_mean.items()):

        if row == 0 or plot_vals:
            base_sim = OrderedDict([(k, 0) for k, v in date_to_field.items()])
            base_label = sim_label
            plot_label = sim_label
            clevs = clevs_vals
            cmap = cmap_vals
            extend = "max"
        else:
            base_sim = list(sim_label_to_date_to_mean.items())[0][1]
            plot_label = "{}\n-\n{}".format(sim_label, base_label)
            clevs = clevs_diff
            cmap = cmap_diff
            extend = "both"

        bn = BoundaryNorm(boundaries=clevs, ncolors=len(clevs) - 1)

        for col, (the_date, field) in enumerate(date_to_field.items()):

            ax = fig.add_subplot(gs[row, col])
            to_plot = np.ma.masked_where(the_mask < 0.5, field - base_sim[the_date])
            # cs = bmp.contourf(xx[~to_plot.mask], yy[~to_plot.mask], to_plot[~to_plot.mask], levels=clevs, extend="max", tri=True)
            cs = bmp.pcolormesh(xx, yy, to_plot[:-1, :-1], norm=bn, vmin=clevs[0], vmax=clevs[-1], cmap=cmap)

            bmp.drawcoastlines(ax=ax, linewidth=0.3)
            assert isinstance(bmp, Basemap)
            bmp.readshapefile(BOW_RIVER_SHP[:-4], "basin", zorder=5)
            cb = bmp.colorbar(cs, ax=ax, ticks=clevs, extend=extend, pad="4%", size="10%")

            if plot_vals:
                cb.ax.set_visible(col == ncols - 1 and row == 0)
            else:
                cb.ax.set_visible(col == ncols - 1 and row in (0, 1))

            if col == 0:
                ax.set_ylabel(plot_label)

            if row == 0:
                ax.set_title(the_date.strftime("%b %d"))

    if plot_vals:
        img_file = img_folder.joinpath("{}.png".format(varname))
    else:
        img_file = img_folder.joinpath("{}_diff.png".format(varname))

    fig.savefig(str(img_file))
    plt.close(fig)


if __name__ == '__main__':
    main()
    main(plot_vals=True)
