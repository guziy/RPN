import matplotlib
from memory_profiler import profile

matplotlib.use("Agg")

from lake_effect_snow.non_local_mask_for_lake_effect import get_nonlocal_mean_snowfall
import multiprocessing
from collections import OrderedDict
from collections import defaultdict
from datetime import timedelta, datetime
from multiprocessing.pool import Pool
from pathlib import Path

import xarray
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, BoundaryNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.basemap import Basemap
from pendulum import Pendulum
from rpn import level_kinds
from scipy.spatial import cKDTree as KDTree
from xarray import DataArray


from lake_effect_snow import base_utils
from lake_effect_snow import common_params
from lake_effect_snow import data_source_types
from lake_effect_snow import default_varname_mappings
from lake_effect_snow import winds
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.data_manager import DataManager
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import maskoceans

import numpy as np

from pendulum import Period

from util import plot_utils

from matplotlib import colors

from util.geo import lat_lon
from util.geo.mask_from_shp import get_mask


def calculate_lake_effect_snowfall(label_to_config, period=None):
    """

    :param label_to_config:
    :param period:  The period of interest defined by the start and the end year of the period (inclusive)
    """

    assert hasattr(period, "months_of_interest")

    for label, the_config in label_to_config.items():
        data_manager = DataManager(store_config=the_config)

        if "out_folder" in the_config:
            out_folder = the_config["out_folder"]
        else:
            out_folder = "."

        calculate_enh_lakeffect_snowfall_for_a_datasource(data_mngr=data_manager, label=label, period=period,
                                                          out_folder=out_folder)



@profile
def enh_lakeffect_snfall_calculator_proc(args):

    """
    For multiprocessing
    :param args:
    """
    data_manager, label, period, out_folder = args



    if not isinstance(period, Period):
        p = Period(start=period[0], end=period[1])
        p.months_of_interest = period[2]
        period = p

    print("Start calculations for {} ... {}".format(period.start, period.end))




    calculate_enh_lakeffect_snowfall_for_a_datasource(data_mngr=data_manager, label=label, period=period,
                                                      out_folder=out_folder)

    print("Finish calculations for {} ... {}".format(period.start, period.end))


@profile
def calculate_lake_effect_snowfall_each_year_in_parallel(label_to_config, period=None, months_of_interest=None, nprocs_to_use=None):
    """
    :param label_to_config:
    :param period:  The period of interest defined by the start and the end year of the period (inclusive)
    """


    if months_of_interest is not None:
        period.months_of_interest = months_of_interest

    assert hasattr(period, "months_of_interest")

    for label, the_config in label_to_config.items():
        data_manager = DataManager(store_config=the_config)


        print(the_config)

        if "out_folder" in the_config:
            out_folder = the_config["out_folder"]
        else:
            out_folder = "."

        out_folder = Path(out_folder)



        try:
            # Try to create the output folder if it does not exist
            if not out_folder.exists():
                out_folder.mkdir()

            print("{}: {} created".format(multiprocessing.current_process().name, out_folder))
        except FileExistsError:
            print("{}: {} already exists".format(multiprocessing.current_process().name, out_folder))



        if nprocs_to_use is None:
            # Use a fraction of the available processes
            nprocs_to_use = max(int(multiprocessing.cpu_count() * 0.75), 1)
            nprocs_to_use = min(nprocs_to_use, period.in_years())  # No need for more processes than there is of years
            nprocs_to_use = max(1, nprocs_to_use)  # make sure that nprocs_to_use is not 0

        print("Using {} processes for parallelization".format(nprocs_to_use))

        # Construct the input params for each process
        in_data = []

        for start in period.range("years"):
            end_date = start.add(months=len(period.months_of_interest)).subtract(seconds=1)
            end_date = min(end_date, period.end)
            p = Period(start=start, end=end_date)
            p.months_of_interest = period.months_of_interest
            in_data.append([data_manager, label, [p.start, p.end, period.months_of_interest], out_folder])
        print(in_data)


        if nprocs_to_use > 1:
            pool = Pool(processes=nprocs_to_use)
            pool.map(enh_lakeffect_snfall_calculator_proc, in_data)
        else:
            for current_in_data in in_data:
                enh_lakeffect_snfall_calculator_proc(current_in_data)

        del data_manager





def get_zone_around_lakes_mask(lons, lats, lake_mask, ktree=None, dist_km=100):
    """
    Returns the mask of a zone around lakes (excluding lakes) of a given width
    :type ktree: cKDTree
    :param ktree:
    :param lons:
    :param lats:
    :param lake_mask:
    :param dist_km:
    """


    x, y, z = lat_lon.lon_lat_to_cartesian(lons[lake_mask], lats[lake_mask])

    near_lake_zone = np.zeros_like(lons, dtype=bool)

    nlons = lons.shape[0] * lons.shape[1]
    near_lake_zone.shape = (nlons,)

    for xi, yi, zi in zip(x, y, z):
        dists, inds = ktree.query([[xi, yi, zi],], k=nlons, distance_upper_bound=dist_km * 1000)
        near_lake_zone[inds[inds < nlons]] = True

    near_lake_zone.shape = (lons.shape[0], lons.shape[1])


    # Remove lake points from the mask
    near_lake_zone &= ~lake_mask

    
    return near_lake_zone



@profile
def calculate_enh_lakeffect_snowfall_for_a_datasource(data_mngr, label="", period=None, out_folder: Path=Path(".")):
    months_of_interest = period.months_of_interest



    out_file = "{}_lkeff_snfl_{}-{}_m{}-{}.nc".format(label, period.start.year, period.end.year,
                                                      months_of_interest[0], months_of_interest[-1], out_folder)
    out_file = out_folder.joinpath(out_file)

    if out_file.exists():
        print("{} already exists, won't redo!".format(out_file))
        plot_acc_snowfall_map(data_path=out_file, label=label, period=period, out_folder=out_folder,
                              months_of_interest=months_of_interest)
        return

    # for each period
    #  1. get daily snowfall
    #  2. get sum of daily snowfalls
    lkeff_snow_falls = []
    lkeff_snow_fall_days = []
    lkeff_snow_fall_eventcount = []
    years_index = []

    reg_of_interest = None
    lons = None
    lats = None
    ktree = None
    lake_mask = None
    near_lake_100km_zone_mask = None

    ktree_for_nonlocal_snowfall_calculations = None

    secs_per_day = timedelta(days=1).total_seconds()

    for start in period.range("years"):

        end_date = start.add(months=len(months_of_interest)).subtract(seconds=1)
        end_date = min(period.end, end_date)

        p = Period(start, end_date)



        print("Processing {} ... {} period".format(p.start, p.end))

        try:
            air_temp = data_mngr.read_data_for_period(p, default_varname_mappings.T_AIR_2M)
        except IOError as e:
            print(e)
            continue


        #
        print("Calculating daily mean 2-m air temperature")
        air_temp = air_temp.resample("1D", dim="t", how="mean")

        # try to read snowfall if not available, try to calculate from total precip
        try:
            snfl = data_mngr.read_data_for_period(p, default_varname_mappings.SNOWFALL_RATE)
            snfl = snfl.resample("1D", dim="t", how="mean")
            rhosn = base_utils.get_snow_density_kg_per_m3(tair_deg_c=air_temp.values)

            # convert from water depth to snow depth
            snfl *= base_utils.WATER_DENSITY_KG_PER_M3 / rhosn

        except (IOError, KeyError, Exception):
            print("Could not find snowfall rate in {}".format(data_mngr.base_folder))
            print("Calculating from 2-m air temperature and total precipitation.")

            # use  daily mean precip (to be consistent with the 2-meter air temperature)
            precip_m_s = data_mngr.read_data_for_period(p, default_varname_mappings.TOTAL_PREC)
            precip_m_s = precip_m_s.resample("1D", dim="t", how="mean")

            # Calculate snowfall from the total precipitation and 2-meter air temperature
            snfl = precip_m_s.copy()
            snfl.name = default_varname_mappings.SNOWFALL_RATE
            snfl.values = base_utils.get_snow_fall_m_per_s(precip_m_per_s=precip_m_s.values, tair_deg_c=air_temp.values)

        print("===========air temp ranges=======")
        print(air_temp.min(), " .. ", air_temp.max())

        print("Snowfall values ranges: ")
        print(snfl.min(), snfl.max(), common_params.lower_limit_of_daily_snowfall)

        # set to 0 snowfall lower than 1 cm/day
        snfl.values[snfl.values <= common_params.lower_limit_of_daily_snowfall] = 0
        snfl *= timedelta(days=1).total_seconds()

        assert isinstance(snfl, DataArray)

        years_index.append(start.year)

        if reg_of_interest is None:
            lons, lats = snfl.coords["lon"].values, snfl.coords["lat"].values

            # convert longitudes to the 0..360 range
            lons[lons < 0] += 360

            reg_of_interest = common_params.great_lakes_limits.get_mask_for_coords(lons, lats)


            # temporary
            lake_mask = get_mask(lons, lats, shp_path=common_params.GL_COAST_SHP_PATH) > 0.1
            print("lake_mask shape", lake_mask.shape)


            # mask lake points
            reg_of_interest &= ~lake_mask

            # get the KDTree for interpolation purposes
            ktree = KDTree(data=list(zip(*lat_lon.lon_lat_to_cartesian(lon=lons.flatten(), lat=lats.flatten()))))

            # define the 100km near lake zone
            near_lake_100km_zone_mask = get_zone_around_lakes_mask(lons=lons, lats=lats, lake_mask=lake_mask,
                                                                   ktree=ktree, dist_km=200)

            reg_of_interest &= near_lake_100km_zone_mask




        # check the winds
        print("Reading the winds into memory")
        u_we = data_mngr.read_data_for_period(p, default_varname_mappings.U_WE)
        u_we = u_we.resample("1D", dim="t", how="mean")

        v_sn = data_mngr.read_data_for_period(p, default_varname_mappings.V_SN)
        v_sn = v_sn.resample("1D", dim="t", how="mean")
        print("Successfully imported wind components")


        # Try to get the lake ice fractions
        lake_ice_fraction = None
        try:
            lake_ice_fraction = data_mngr.read_data_for_period(p, default_varname_mappings.LAKE_ICE_FRACTION)
            lake_ice_fraction = lake_ice_fraction.resample("1D", dim="t", how="mean")  # Calculate the daily means
            lake_ice_fraction = lake_ice_fraction.sel(t=v_sn.coords["t"], method="nearest")
            lake_ice_fraction = lake_ice_fraction.where(lake_ice_fraction <= 1)

            # at this point shapes of the arrays should be the same
            assert lake_ice_fraction.shape == u_we.shape

            print(lake_ice_fraction.coords["t"][0], lake_ice_fraction.coords["t"][-1])

        except Exception as e:


            print(e)
            print("WARNING: Could not find lake fraction in {}, diagnosing lake-effect snow without lake ice".format(
                data_mngr.base_folder))
            lake_ice_fraction = None

            # raise e

        # take into account the wind direction
        wind_blows_from_lakes = winds.get_wind_blows_from_lakes_mask(lons, lats, u_we.values, v_sn.values, lake_mask,
                                                                     ktree=ktree,
                                                                     region_of_interest=reg_of_interest,
                                                                     dt_secs=secs_per_day, nneighbours=4,
                                                                     lake_ice_fraction=lake_ice_fraction)
        snfl = wind_blows_from_lakes * snfl



        # take into account nonlocal amplification due to lakes
        if ktree_for_nonlocal_snowfall_calculations is None:
            xs_nnlc, ys_nnlcl, zs_nnlcl = lat_lon.lon_lat_to_cartesian(lons.flatten(), lats.flatten())
            ktree_for_nonlocal_snowfall_calculations = KDTree(list(zip(xs_nnlc, ys_nnlcl, zs_nnlcl)))

        snfl_nonlocal = get_nonlocal_mean_snowfall(lons=lons, lats=lats, region_of_interest=reg_of_interest,
                                                   kdtree=ktree_for_nonlocal_snowfall_calculations,
                                                   snowfall=snfl, lake_mask=lake_mask, outer_radius_km=500)




        snfl = (snfl > (common_params.snfl_local_amplification_m_per_s + snfl_nonlocal)).values * snfl

        # count the number of hles events
        snfl_eventcount = snfl.copy()
        snfl_eventcount.values[snfl.values > 1e-5] = 1
        snfl_eventcount.values[snfl.values <= 1e-5] = 0

        snfl_eventcount = snfl_eventcount.diff("t", n=1)
        snfl_eventcount = (snfl_eventcount < 0).sum(dim="t")
        lkeff_snow_fall_eventcount.append(snfl_eventcount)

        # count the number of days with lake effect snowfall
        lkeff_snow_fall_days.append((snfl > 0).sum(dim="t"))

        #  Get the accumulation of the lake effect snowfall
        snfl_acc = snfl.sum(dim="t")
        
        # takes into account the 100km zone near lakes
        # snfl_acc.values = np.ma.masked_where((~reg_of_interest) | (~near_lake_100km_zone_mask), snfl_acc)
        snfl_acc.values = np.ma.masked_where((~reg_of_interest), snfl_acc)

        lkeff_snow_falls.append(snfl_acc)


    if len(years_index) == 0:
         print("Nothing to plot, exiting.")
         return

    # concatenate the yearly accumulated snowfall and save the result to a netcdf file
    # select the region of interest before saving calculated fields to the file
    years_index = DataArray(years_index, name="year", dims="year")

    i_arr, j_arr = np.where(reg_of_interest)
    i_min, i_max = i_arr.min(), i_arr.max()
    j_min, j_max = j_arr.min(), j_arr.max()



    snfl_yearly = xarray.concat([arr.loc[i_min: i_max + 1, j_min: j_max + 1] for arr in lkeff_snow_falls],
                                dim=years_index)
    snfl_yearly.attrs["units"] = "m"

    snfl_days_yearly = xarray.concat([arr.loc[i_min: i_max + 1, j_min: j_max + 1] for arr in lkeff_snow_fall_days],
                                     dim=years_index)
    snfl_days_yearly.attrs["units"] = "days"


    snfl_eventcounts_yearly = xarray.concat([arr.loc[i_min: i_max + 1, j_min: j_max + 1] for arr in lkeff_snow_fall_eventcount],
                                     dim=years_index)
    snfl_eventcounts_yearly.attrs["units"] = "number of events"




    ds = snfl_yearly.to_dataset()
    assert isinstance(ds, xarray.Dataset)
    ds["lkeff_snowfall_days"] = (("year", "x", "y"), snfl_days_yearly)
    ds["lkeff_snowfall_eventcount"] = (("year", "x", "y"), snfl_eventcounts_yearly)

    ds.to_netcdf(str(out_file))

    ds.close()

    # do the plotting
    plot_acc_snowfall_map(data_path=out_file, label=label, period=period, out_folder=out_folder,
                          months_of_interest=period.months_of_interest)



@profile
def plot_acc_snowfall_map(data_path :Path=None, label="", period :Period=None, out_folder :Path=None, months_of_interest:list=None):
    # Plot snowfall maps for each year
    """
    Data is converted from m to cm before plotting
    :param data_path:
    :param label:
    :param period:
    :param out_folder:
    """
    clevs_total_snowfall = [0, 10, 50, 90, 130, 170, 210, 250, 400, 500]
    clevs_lkeff_snowfall = [0, 1, 2, 10, 15, 20, 40, 80, 120, 160, 200, 250]
    clevs = clevs_lkeff_snowfall


    ds = xarray.open_dataset(str(data_path))

    lkeff_snfl = ds["snow_fall"]
    years_index, lons, lats = ds["year"].values, ds["lon"].values, ds["lat"].values


    b = Basemap(lon_0=180,
                llcrnrlon=common_params.great_lakes_limits.lon_min,
                llcrnrlat=common_params.great_lakes_limits.lat_min,
                urcrnrlon=common_params.great_lakes_limits.lon_max,
                urcrnrlat=common_params.great_lakes_limits.lat_max,
                resolution="i")

    xx, yy = b(lons, lats)


    # print("Basemap corners: ", lons[i_min, j_min] - 360, lons[i_max, j_max] - 360)

    plot_utils.apply_plot_params(font_size=10)
    fig = plt.figure()

    ncols = 3
    nrows = len(years_index) // ncols + 1
    gs = GridSpec(ncols=ncols, nrows=nrows)

    # bn = BoundaryNorm(clevs, len(clevs) - 1)
    # cmap = cm.get_cmap("nipy_spectral")

    cmap, bn = colors.from_levels_and_colors(clevs, ["indigo", "blue", "dodgerblue", "aqua", "lime", "yellow", "gold",
                                                     "orange", "orangered", "red", "firebrick", ])

    area_avg_lkeff_snowfall = []
    for i, y in enumerate(years_index):
        col = i % ncols
        row = i // ncols
        ax = fig.add_subplot(gs[row, col])
        # to_plot = np.ma.masked_where(~reg_of_interest, lkeff_snfl[i].values)


        to_plot = lkeff_snfl[i].to_masked_array()
        print(xx.shape, to_plot.shape)


        to_plot *= 100  # convert to cm
        im = b.contourf(xx, yy, to_plot, norm=bn, cmap=cmap, levels=clevs, extend="max")



        area_avg_lkeff_snowfall.append(to_plot[(~to_plot.mask) & (to_plot > 0)].mean())

        cb = b.colorbar(im, ax=ax)
        cb.ax.set_title("cm")

        b.drawcoastlines()
        b.drawparallels(np.arange(-90, 90, 10), labels=[1, 0, 0, 1])
        b.drawmeridians(np.arange(-180, 180, 10), labels=[1, 0, 0, 1])

        ax.set_title("{}".format(y))

    fig.tight_layout()



    months_of_interest_str = [str(m) for m in months_of_interest] if months_of_interest is not None else ["unknown_months"]
    img_file = "{}_acc_lakeff_snow_{}-{}_m{}.png".format(label, period.start.year, period.end.year, "_".join(months_of_interest_str))

    img_file = str(out_folder.joinpath(img_file))
    print("Saving plot to {}".format(img_file))
    plt.savefig(img_file, bbox_inches="tight")
    # plt.show()
    plt.close(fig)

    # plot area-averaged lake-effect snowfall
    # fig = plt.figure()
    # ax = plt.gca()
    # ax.plot(years_index.values.astype(int), area_avg_lkeff_snowfall, "r", lw=2)
    # ax.set_title("Area averaged annual lake-effect snowfall")
    # sf = ScalarFormatter(useOffset=False)
    # ax.xaxis.set_major_formatter(sf)
    # ax.grid()
    #
    # fig.tight_layout()
    # img_file = "{}_acc_lakeff_snow_area_avg_{}-{}.png".format(label, period.start.year, period.end.year - 1)
    # img_file = str(out_folder.joinpath(img_file))
    # plt.savefig(img_file, bbox_inches="tight")
    # plt.close(fig)



def main():
    # First approximation of the lake-effect snow, by looking at the daily snowfall of more than 1 cm/day
    period = Period(
        datetime(1994, 12, 1), datetime(1995, 3, 1)
    )

    # should be consequent
    months_of_interest = [12, 1, 2]

    period.months_of_interest = months_of_interest

    ERAI_label = "ERA-Interim"

    vname_to_level_erai = {
        default_varname_mappings.T_AIR_2M: VerticalLevel(1, level_kinds.HYBRID),
        default_varname_mappings.U_WE: VerticalLevel(1, level_kinds.HYBRID),
        default_varname_mappings.V_SN: VerticalLevel(1, level_kinds.HYBRID),

    }

    label_to_config = OrderedDict(
        [  # ERA-Interim
            (ERAI_label,
             {
                 "base_folder": "/RECH/data/Driving_data/Offline/ERA-Interim_0.75/6h_Analysis",
                 "data_source_type": data_source_types.ALL_VARS_IN_A_FOLDER_OF_RPN_FILES,
                 "min_dt": timedelta(hours=6),
                 "varname_mapping": default_varname_mappings.vname_map_CRCM5,
                 "level_mapping": vname_to_level_erai,
                 "offset_mapping": default_varname_mappings.vname_to_offset_CRCM5,
                 "multiplier_mapping": default_varname_mappings.vname_to_multiplier_CRCM5
             }
             ),
            # Add additional sources below
        ]
    )

    label = "ECMWF_CRCM5_FLake_0"

    label_to_config_CRCM5 = OrderedDict([(
        label, {
            "base_folder": "/HOME/huziy/skynet3_rech1/ens_simulations_links_diro/ENSSEASF_NorthAmerica_0.22deg_B1_0",
            "data_source_type": data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT,
            "min_dt": timedelta(hours=3),
            "varname_mapping": default_varname_mappings.vname_map_CRCM5,
            "level_mapping": vname_to_level_erai,
            "offset_mapping": default_varname_mappings.vname_to_offset_CRCM5,
            "multiplier_mapping": default_varname_mappings.vname_to_multiplier_CRCM5,
            "filename_prefix_mapping": default_varname_mappings.vname_to_fname_prefix_CRCM5,
            "out_folder": "lake_effect_analysis_{}".format(label)
        }
    )])

    # for i in range(1, 9):
    #     label = "ECMWF_CRCM5_FLake_{}".format(i)
    #     label_to_config_CRCM5[label] = label_to_config_CRCM5[label0]
    #     label_to_config_CRCM5[label]["out_folder"] = "lake_effect_analysis_{}".format(label)


    # ECMWF GCM ensemble member outputs
    label_ECMWF_GCM = "ECMWF_GCM_1"

    multiplier_map_ECMWF_GCM = defaultdict(lambda: 1)
    multiplier_map_ECMWF_GCM[default_varname_mappings.TOTAL_PREC] = 1.0e-3 / (24.0 * 3600.0)  # convert to M/S]

    label_to_config_ECMWF_GCM = OrderedDict(
        [
            (label_ECMWF_GCM, {
                "base_folder": "/RESCUE/skynet3_rech1/huziy/ens_simulations_links_diro/ECMWF_GCM/ensm_1",
                "data_source_type": data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES,
                "out_folder": "lake_effect_analysis_{}".format(label_ECMWF_GCM),
                "varname_mapping": {
                    default_varname_mappings.T_AIR_2M: "tas",
                    default_varname_mappings.TOTAL_PREC: "prlr",
                    default_varname_mappings.U_WE: "uas",
                    default_varname_mappings.V_SN: "vas",
                },
                "multiplier_mapping": multiplier_map_ECMWF_GCM,
                "offset_mapping": defaultdict(lambda: 0),
                "level_mapping": defaultdict(lambda: 0),
             }),
        ]
    )

    calculate_lake_effect_snowfall(label_to_config=label_to_config_CRCM5, period=period)


if __name__ == '__main__':
    import time
    t0 = time.clock()
    main()
    print("Execution time {} seconds".format(time.clock() - t0))



