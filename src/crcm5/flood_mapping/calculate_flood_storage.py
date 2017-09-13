import calendar
from collections import OrderedDict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import xarray
from matplotlib import cm
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.basemap import Basemap, maskoceans
from rpn.rpn import RPN

from data.robust import data_source_types
from data.robust.data_manager import DataManager
from domains.rotated_lat_lon import RotatedLatLon
from lake_effect_snow.base_utils import VerticalLevel
from util import plot_utils
from util.geo.mask_from_shp import get_mask

DEFAULT_PATH_FOR_GEO_DATA = "/snow3/huziy/NEI/GL/GL_CC_CanESM2_RCP85/coupled-GL-current_CanESM2/Samples/" \
                            "coupled-GL-current_CanESM2_198812/pm1988120100_00000000p"


def __get_maximum_storage_and_corresponding_dates(start_year:int, end_year:int, data_manager:DataManager, storage_varname=""):
    cache_file_current = "cache_{}-{}_calculate_flood_storage_{}.nc".format(start_year, end_year, storage_varname)
    cache_file_current = Path(cache_file_current)

    # if the variables were calculated already
    if cache_file_current.exists():
        ds = xarray.open_dataset(str(cache_file_current))
    else:
        data_current = data_manager.get_min_max_avg_for_period(
            start_year=start_year, end_year=end_year, varname_internal=storage_varname
        )

        ds = xarray.merge([da for da in data_current.values()])
        ds.to_netcdf(str(cache_file_current))

    return ds


def __storage_cb_format_ticklabels(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)

    if abs(float(a)) < 1e-10:
        return "0"


    if b == 0:
        return r"{}".format(a)
    elif abs(float(a) - 1) < 1e-7:
        return r"$10^{{{}}}$".format(b)
    elif abs(float(a) + 1) < 1e-7:
        return r"$-10^{{{}}}$".format(b)

    return r"${} \times 10^{{{}}}$".format(a, b)


def __timing_cb_format_ticklabels(x, pos):
    v = int(x)
    if x == 0:
        return ""
    return calendar.month_abbr[v]


def __get_new_tick_locs_middle(old_ticks, ncolors, shift_direction=1):
    color_height = 1.0 / ncolors
    shift = color_height / 2.0

    new_ticks = old_ticks + shift * shift_direction
    # new_ticks = new_ticks[(new_ticks <= 1) & (new_ticks >= 0)]

    return new_ticks




def __plot_vals(ds:xarray.Dataset, bmap:Basemap, lons2d, lats2d, label="test", bankfull_storage=None,
                storage_var_name="SWSR", region_of_interest_shp=None, plot_deviations_from_bankfull_storage=False):
    """
    Plot variables in the dataset
    :param ds:
    """


    print(ds)

    img_dir = "flood_maps"
    img_dir = Path(img_dir)
    if not img_dir.exists():
        img_dir.mkdir()


    xx, yy = bmap(lons2d, lats2d)
    lons2d_ = lons2d.copy()
    lons2d_[lons2d_ > 180] -= 360.0

    storage_lower_limit_m3 = 1



    # do the plotting
    plot_utils.apply_plot_params(font_size=10)
    fig = plt.figure()

    nrows = 2; ncols = 3
    gs = GridSpec(nrows=nrows, ncols=ncols, wspace=0.01, hspace=0.1)




    axes = []

    # the 2 functions below are used to plot panel plots (used to decrease amount of code)
    def __plot_storage(prefix, show_cb=False, row=0, col=0, plot_deviations=True):

        if plot_deviations:
            clevs = [0, 1e3, 1e4, 1e5, 1.0e6, 1e7, 1e8, 1.0e9]
            clevs = [-c for c in reversed(clevs)][:-1] + clevs
            cmap = cm.get_cmap("bwr", len(clevs) - 1)
        else:
            clevs = [0, 1e3, 1e4, 1e5, 1.0e6, 1e7, 1e8, 1.0e9]
            cmap = cm.get_cmap("YlGnBu", len(clevs) - 1)

        norm = BoundaryNorm(boundaries=clevs, ncolors=len(clevs) - 1)

        _storage = ds["{}_{}".format(prefix, storage_var_name)][:]
        ax = fig.add_subplot(gs[row, col])
        _storage = _storage.where(_storage > storage_lower_limit_m3)
        _storage = maskoceans(lons2d_, lats2d, _storage)
        _storage = np.ma.masked_where(np.isnan(_storage), _storage)


        if plot_deviations:
            cs = bmap.pcolormesh(xx, yy, _storage - bankfull_storage, norm=norm, cmap=cmap)
        else:
            cs = bmap.pcolormesh(xx, yy, _storage, norm=norm, cmap=cmap)


        ext = "both" if plot_deviations else "max"


        cb = bmap.colorbar(cs, location="bottom", format=FuncFormatter(__storage_cb_format_ticklabels), extend=ext)
        cb.ax.set_visible(show_cb)
        cb.ax.set_xlabel(r"${\rm m^3}$")
        ax.set_title(prefix)
        axes.append(ax)
        return _storage




    clevs_timings = range(13)
    norm_timings = BoundaryNorm(clevs_timings, len(clevs_timings) - 1)
    cmap_timings = cm.get_cmap("Spectral", len(clevs_timings) - 1)


    def __plot_timings(prefix, show_cb=False, row=0, col=0, the_storage=None):
        _dates = ds["{}_dates.month".format(prefix)][:]

        ax = fig.add_subplot(gs[row, col])

        if the_storage is not None:
            _dates = _dates.where(~np.isnan(the_storage))
            _dates = np.ma.masked_where(the_storage.mask, _dates)

        _dates = maskoceans(lons2d_, lats2d, _dates)

        cs = bmap.pcolormesh(xx, yy, _dates, norm=norm_timings, cmap=cmap_timings)
        cb = bmap.colorbar(cs, location="bottom", format=FuncFormatter(__timing_cb_format_ticklabels))

        if show_cb:
            cb.ax.set_xlabel("month")

            maj_locator = cb.ax.xaxis.get_major_locator()


            print("old tick locs = {}".format(maj_locator.locs))
            maj_locator.locs = __get_new_tick_locs_middle(maj_locator.locs, len(clevs_timings) - 1, shift_direction=-1)
            print("new tick locs = {}".format(maj_locator.locs))


            for tick_line in cb.ax.xaxis.get_ticklines():
                tick_line.set_visible(False)

        cb.ax.set_visible(show_cb)
        ax.set_title("{} timing".format(prefix))
        axes.append(ax)


    # minimum storage
    min_storage = __plot_storage("min", show_cb=True, row=0, col=0, plot_deviations=plot_deviations_from_bankfull_storage)

    # maximum storage
    max_storage = __plot_storage("max", row=0, col=1, plot_deviations=plot_deviations_from_bankfull_storage)

    # average storage
    avg_storage = __plot_storage("avg", row=0, col=2, plot_deviations=plot_deviations_from_bankfull_storage)


    # bankfull storage (if provided)
    if bankfull_storage is not None:
        bf_storage_varname = "bankfull_{}".format(storage_var_name)

        ds[bf_storage_varname] = xarray.DataArray(data=bankfull_storage, dims=("x", "y"))
        __plot_storage("bankfull", row=1, col=2, plot_deviations=False, show_cb=True)
        ds.drop(bf_storage_varname)

    # tmin
    dates_clevs = range(0, 13)
    __plot_timings("min", row=1, col=0, show_cb=True, the_storage=min_storage)


    # tmax
    __plot_timings("max", row=1, col=1, the_storage=max_storage)


    for i, ax in enumerate(axes):
        bmap.drawcoastlines(linewidth=0.1, ax=ax)
        if region_of_interest_shp is not None:
            bmap.readshapefile(region_of_interest_shp[:-4], "basin", ax=ax, color="k", linewidth=2)



    if plot_deviations_from_bankfull_storage:
        label = label + "_bf_storage_anomalies"

    img_file = img_dir / "{}.png".format(label)
    fig.savefig(str(img_file), dpi=400, bbox_inches="tight")
    plt.close(fig)



def __get_lons_lats_basemap_from_rpn(path=DEFAULT_PATH_FOR_GEO_DATA,
                           vname="STBM", region_of_interest_shp=None, **bmp_kwargs):

    """
    :param path:
    :param vname:
    :return: get basemap object for the variable in the given file
    """
    with RPN(str(path)) as r:
        _ = r.variables[vname][:]

        proj_params = r.get_proj_parameters_for_the_last_read_rec()
        lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()

        rll = RotatedLatLon(**proj_params)

    if region_of_interest_shp is not None:
        mask = get_mask(lons, lats, region_of_interest_shp)
        delta_points = 10
        i_arr, j_arr = np.where(mask >= 0.5)
        i_min, i_max = i_arr.min() - delta_points, i_arr.max() + delta_points
        j_min, j_max = j_arr.min() - delta_points, j_arr.max() + delta_points

        slices = (slice(i_min,i_max + 1), slice(j_min,j_max + 1))

        bmp = rll.get_basemap_object_for_lons_lats(lons2d=lons[slices], lats2d=lats[slices], **bmp_kwargs)
    else:
        bmp = rll.get_basemap_object_for_lons_lats(lons2d=lons, lats2d=lats, **bmp_kwargs)


    return lons, lats, bmp


def __read_flow_directions(path=DEFAULT_PATH_FOR_GEO_DATA):
    return __read_geo_field(path, "FLDR")


def __read_bankfull_storage(path=DEFAULT_PATH_FOR_GEO_DATA):
    return __read_geo_field(path, "STBM")


def __read_geo_field(path, vname):
    with RPN(str(path)) as r:
        field = r.variables[vname][:].squeeze()
    return field


def main():

    region_of_interest_shp = "data/shp/mtl_flood_2017_basins/02JKL_SDA_Ottawa.shp"

    current_simlabel = "GL_Current_CanESM2"
    future_simlabel = "GL_Future_CanESM2"


    river_storage_varname = "SWSR"
    lake_storage_varname = "SWSL"


    start_year_current = 1989
    end_year_current = 2010

    start_year_future = 2079
    end_year_future = 2100


    base_sim_dir = Path("/snow3/huziy/NEI/GL/GL_CC_CanESM2_RCP85")
    label_to_sim_path = OrderedDict()

    label_to_sim_path[current_simlabel] = base_sim_dir / "coupled-GL-current_CanESM2" / "Samples"
    label_to_sim_path[future_simlabel] = base_sim_dir / "coupled-GL-future_CanESM2" / "Samples"


    # some common mappings
    varname_mapping = {
        river_storage_varname: river_storage_varname,
        lake_storage_varname: lake_storage_varname
    }

    level_mapping = {
        river_storage_varname: VerticalLevel(-1)
    }

    vname_to_fname_prefix = {river_storage_varname: "pm", lake_storage_varname: "pm"}

    dm_current = DataManager(
        store_config={
            "base_folder": str(label_to_sim_path[current_simlabel]),
            "data_source_type": data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT,
            "varname_to_filename_prefix_mapping": vname_to_fname_prefix,
            "varname_mapping": varname_mapping,
            "level_mapping": level_mapping
        }
    )



    dm_future = DataManager(
        store_config={
            "base_folder": str(label_to_sim_path[future_simlabel]),
            "data_source_type": data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT,
            "varname_to_filename_prefix_mapping": vname_to_fname_prefix,
            "varname_mapping": varname_mapping,
            "level_mapping": level_mapping
        }
    )



    #
    ds_current = __get_maximum_storage_and_corresponding_dates(start_year_current, end_year_current,
                                                               data_manager=dm_current,
                                                               storage_varname=river_storage_varname)



    ds_future =  __get_maximum_storage_and_corresponding_dates(start_year_future, end_year_future,
                                                               data_manager=dm_future,
                                                               storage_varname=river_storage_varname)



    # get constant in time geophysical fields
    bf_storage = __read_bankfull_storage()

    #
    lons, lats, bmap = __get_lons_lats_basemap_from_rpn(resolution="i", region_of_interest_shp=region_of_interest_shp)




    # plot current climate values
    label = "storage_{}-{}".format(start_year_current, end_year_current)
    __plot_vals(ds_current, bmap, lons, lats, label=label, storage_var_name=river_storage_varname,
                bankfull_storage=bf_storage, region_of_interest_shp=region_of_interest_shp,
                plot_deviations_from_bankfull_storage=True)

    label = "storage_{}-{}".format(start_year_future, end_year_future)
    __plot_vals(ds_future, bmap, lons, lats, label=label, storage_var_name=river_storage_varname,
                bankfull_storage=bf_storage, region_of_interest_shp=region_of_interest_shp,
                plot_deviations_from_bankfull_storage=True)





if __name__ == '__main__':
    main()
    #import profile
    #print(profile.run("main()"))