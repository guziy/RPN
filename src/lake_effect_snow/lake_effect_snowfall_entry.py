from collections import OrderedDict
from datetime import timedelta, datetime

import xarray
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, BoundaryNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.basemap import Basemap
from pendulum import Pendulum
from rpn import level_kinds
from scipy.spatial import KDTree
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


def calculate_lake_effect_snowfall(label_to_config, period=None):
    """

    :param label_to_config:
    :param period:  The period of interest defined by the start and the end year of the period (inclusive)
    """


    assert hasattr(period, "months_of_interest")

    for label, the_config in label_to_config.items():
        data_manager = DataManager(store_config=the_config)
        calculate_enh_lakeffect_snowfall_for_a_datasource(data_mngr=data_manager, label=label, period=period)



def calculate_enh_lakeffect_snowfall_for_a_datasource(data_mngr, label="", period=None):

    out_file = "{}_lkeff_snfl_{}-{}_.nc".format(label, period.start.year, period.end.year)

    # for each period
    #  1. get daily snowfall
    #  2. get sum of daily snowfalls
    lkeff_snow_falls = []
    years_index = []

    reg_of_interest = None
    lons = None
    lats = None
    ktree = None
    lake_mask = None

    months_of_interest = period.months_of_interest
    secs_per_day = timedelta(days=1).total_seconds()

    for start in period.range("years"):
        p = Period(start, start.add(months=len(months_of_interest)))

        air_temp = data_mngr.read_data_for_period(p, default_varname_mappings.T_AIR_2M)

        day_dates = [datetime(d.year, d.month, d.day) for d in pd.to_datetime(air_temp.coords["t"].values)]
        day_dates = DataArray(day_dates, name="time", dims="t")

        air_temp = air_temp.groupby(day_dates).mean(dim="t")

        # try to read snowfall if not available, try to calculate from total precip
        try:
            snfl = data_mngr.read_data_for_period(p, default_varname_mappings.SNOWFALL_RATE)
            snfl = snfl.groupby(day_dates).mean(dim="t")
            rhosn = base_utils.get_snow_density_kg_per_m3(tair_deg_c=air_temp.values)

            # convert from water depth to snow depth
            snfl *= base_utils.WATER_DENSITY_KG_PER_M3 / rhosn

        except IOError:
            print("Could not find snowfall rate in {}".format(data_mngr.base_folder))
            print("Calculating from 2-m air temperature and total precipitation.")

            # use  daily mean precip (to be consistent with the 2-meter air temperature)
            precip_m_s = data_mngr.read_data_for_period(p, default_varname_mappings.TOTAL_PREC)
            precip_m_s = precip_m_s.groupby(day_dates).mean(dim="t")

            # Calculate snowfall from the total precipitation and 2-meter air temperature
            snfl = base_utils.get_snow_fall_m_per_s(precip_m_per_s=precip_m_s.values, tair_deg_c=air_temp.values)



        print("===========air temp ranges=======")
        print(air_temp.min(), " .. ", air_temp.max())


        print("Snowfall values ranges: ")
        print(snfl.values.min(), snfl.values.max(), common_params.lower_limit_of_daily_snowfall)

        # set to 0 snowfall lower than 1 cm/day
        snfl.values[snfl.values <= common_params.lower_limit_of_daily_snowfall] = 0
        snfl *= timedelta(days=1).total_seconds()

        assert isinstance(snfl, DataArray)

        years_index.append(start.year)

        if reg_of_interest is None:
            lons, lats = snfl.coords["lon"].values, snfl.coords["lat"].values
            reg_of_interest = common_params.great_lakes_limits.get_mask_for_coords(lons, lats)

            # mask lake points
            oc_mask = maskoceans(np.where(lons < 180, lons, lons - 360), lats, lons, inlands=True)
            reg_of_interest = reg_of_interest & (~oc_mask.mask)

            # temporary, TODO: make the lake mask using a shape file
            lake_mask = oc_mask.mask

            # get the KDTree for interpolation purposes
            ktree = KDTree(data=list(zip(*lat_lon.lon_lat_to_cartesian(lon=lons.flatten(), lat=lats.flatten()))))

        #  check the winds
        print("Reading the winds into memory")
        u_we = data_mngr.read_data_for_period(p, default_varname_mappings.U_WE)
        u_we = u_we.groupby(day_dates).mean(dim="t")

        v_sn = data_mngr.read_data_for_period(p, default_varname_mappings.V_SN)
        v_sn = v_sn.groupby(day_dates).mean(dim="t")
        print("Successfully imported wind components")

        wind_blows_from_lakes = winds.get_wind_blows_from_lakes_mask(lons, lats, u_we.values, v_sn.values, lake_mask, ktree=ktree,
                                                                     region_of_interest=reg_of_interest, dt_secs=secs_per_day)

        snfl = wind_blows_from_lakes * snfl


        #  Get the accumulation of the lake effect snowfall
        snfl_acc = snfl.sum(dim="time")
        snfl_acc.values = np.ma.masked_where(~reg_of_interest, snfl_acc)

        lkeff_snow_falls.append(snfl_acc)


        # debug
        break

    # concatenate the yearly accumulated snowfall and save the result to a netcdf file
    years_index = DataArray(years_index, name="year", dims="year")

    i_arr, j_arr = np.where(reg_of_interest)
    i_min, i_max = i_arr.min(), i_arr.max()
    j_min, j_max = j_arr.min(), j_arr.max()

    snfl_yearly = xarray.concat([arr.loc[i_min: i_max + 1, j_min: j_max + 1] for arr in lkeff_snow_falls],
                                dim=years_index)
    snfl_yearly.attrs["units"] = "m"
    snfl_yearly.to_dataset().to_netcdf(out_file)


    # Plot snowfall maps for each year
    clevs_total_snowfall = [0, 10, 50, 90, 130, 170, 210, 250, 400, 500]
    clevs_lkeff_snowfall = [0, 1, 2, 10, 15, 20, 40, 80, 120, 160]
    clevs = clevs_lkeff_snowfall

    b = Basemap(lon_0=180,
                llcrnrlon=lons[i_min, j_min],
                llcrnrlat=lats[i_min, j_min],
                urcrnrlon=lons[i_max, j_max],
                urcrnrlat=lats[i_max, j_max],
                resolution="i")

    xx, yy = b(lons, lats)

    plot_utils.apply_plot_params(font_size=10)
    fig = plt.figure()

    ncols = 3
    nrows = len(years_index) // ncols + 1
    gs = GridSpec(ncols=ncols, nrows=nrows)

    # bn = BoundaryNorm(clevs, len(clevs) - 1)
    # cmap = cm.get_cmap("nipy_spectral")

    cmap, bn = colors.from_levels_and_colors(clevs, ["indigo", "blue", "dodgerblue", "aqua", "lime", "yellow", "gold",
                                                     "orange", "red"])

    area_avg_lkeff_snowfall = []
    for i, y in enumerate(years_index.values):
        col = i % ncols
        row = i // ncols
        ax = fig.add_subplot(gs[row, col])
        to_plot = np.ma.masked_where(~reg_of_interest, lkeff_snow_falls[i])

        print(xx.shape, to_plot.shape)

        to_plot *= 100  # convert to cm
        im = b.contourf(xx, yy, to_plot, norm=bn, cmap=cmap, levels=clevs)

        area_avg_lkeff_snowfall.append(to_plot[(~to_plot.mask) & (to_plot > 0)].mean())

        cb = b.colorbar(im, ax=ax)
        cb.ax.set_title("cm")

        b.drawcoastlines()
        b.drawparallels(np.arange(-90, 90, 10), labels=[1, 0, 0, 1])
        b.drawmeridians(np.arange(-180, 180, 10), labels=[1, 0, 0, 1])

        ax.set_title("{}".format(y))

    fig.tight_layout()
    plt.savefig("{}_acc_lakeff_snow_{}-{}.png".format(label, period.start.year, period.end.year - 1), bbox_inches="tight")
    plt.close(fig)

    # plot area-averaged lake-effect snowfall
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(years_index.values.astype(int), area_avg_lkeff_snowfall, "r", lw=2)
    ax.set_title("Area averaged annual lake-effect snowfall")
    sf = ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(sf)

    assert isinstance(ax, Axes)
    ax.grid()

    fig.tight_layout()
    plt.savefig("{}_acc_lakeff_snow_area_avg_{}-{}.png".format(label, period.start.year, period.end.year - 1),
                bbox_inches="tight")
    plt.close(fig)


def main():

    # First approximation of the lake-effect snow, by looking at the daily snowfall of more than 1 cm/day
    period = Period(
        datetime(1991, 12, 1), datetime(2002, 3, 1)
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
        [   # ERA-Interim
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
             )
            # Add additional sources below
         ]
    )



    calculate_lake_effect_snowfall(label_to_config=label_to_config, period=period)


if __name__ == '__main__':
    main()