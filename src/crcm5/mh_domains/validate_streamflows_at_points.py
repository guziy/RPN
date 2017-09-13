import calendar
from collections import OrderedDict
from collections import defaultdict
from datetime import timedelta, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from pendulum import Pendulum
from pendulum import Period
from rpn import level_kinds
from rpn.rpn import RPN

from application_properties import main_decorator
from crcm5.mh_domains import constants
from crcm5.mh_domains import stfl_stations
from crcm5.model_point import ModelPoint
from data.cehq_station import Station
from data.cell_manager import CellManager
from data.robust import data_source_types
from data.robust.data_manager import DataManager
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.default_varname_mappings import vname_to_offset_CRCM5
from util import pandas_utils
from util import plot_utils

img_folder = Path("mh/engage_report/station_data_plots")



date_formatter = DateFormatter("%b")

def format_month_label(x, pos):
    return date_formatter.format_data(x)[0]










@main_decorator
def main():
    direction_file_path = Path("/RECH2/huziy/BC-MH/bc_mh_044deg/Samples/bc_mh_044deg_198001/pm1980010100_00000000p")

    sim_label = "mh_0.44"

    start_year = 1981
    end_year = 2010

    streamflow_internal_name = "streamflow"
    selected_staion_ids = constants.selected_station_ids_for_streamflow_validation

    # ======================================================





    day = timedelta(days=1)
    t0 = datetime(2001, 1, 1)
    stamp_dates = [t0 + i * day for i in range(365)]
    print("stamp dates range {} ... {}".format(stamp_dates[0], stamp_dates[-1]))


    lake_fraction = None

    # establish the correspondence between the stations and model grid points
    with RPN(str(direction_file_path)) as r:
        assert isinstance(r, RPN)
        fldir = r.get_first_record_for_name("FLDR")
        flow_acc_area = r.get_first_record_for_name("FAA")
        lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()
        # lake_fraction = r.get_first_record_for_name("LF1")

    cell_manager = CellManager(fldir, lons2d=lons, lats2d=lats, accumulation_area_km2=flow_acc_area)
    stations = stfl_stations.load_stations_from_csv(selected_ids=selected_staion_ids)
    station_to_model_point = cell_manager.get_model_points_for_stations(station_list=stations, lake_fraction=lake_fraction,
                                                                        nneighbours=8)


    # Update the end year if required
    max_year_st = -1
    for station in station_to_model_point:
        y = max(station.get_list_of_complete_years())
        if y >= max_year_st:
            max_year_st = y


    if end_year > max_year_st:
        print("Updated end_year to {}, because no obs data after...".format(max_year_st))
        end_year = max_year_st



    # read model data
    mod_data_manager = DataManager(
        store_config={
            "varname_mapping": {streamflow_internal_name: "STFA"},
            "base_folder": str(direction_file_path.parent.parent),
            "data_source_type": data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT,
            "level_mapping": {streamflow_internal_name: VerticalLevel(-1, level_type=level_kinds.ARBITRARY)},
            "offset_mapping": vname_to_offset_CRCM5,
            "filename_prefix_mapping": {streamflow_internal_name: "pm"}
    })


    station_to_model_data = defaultdict(list)
    for year in range(start_year, end_year + 1):
        start = Pendulum(year, 1, 1)
        p_test = Period(start, start.add(years=1).subtract(microseconds=1))
        stfl_mod = mod_data_manager.read_data_for_period(p_test, streamflow_internal_name)

        # convert to daily
        stfl_mod = stfl_mod.resample("D", "t", how="mean", closed="left", keep_attrs=True)

        assert isinstance(stfl_mod, xr.DataArray)

        for station, model_point in station_to_model_point.items():
            assert isinstance(model_point, ModelPoint)
            ts1 = stfl_mod[:, model_point.ix, model_point.jy].to_series()
            station_to_model_data[station].append(pd.Series(index=stfl_mod.t.values, data=ts1))





    # concatenate the timeseries for each point, if required
    if end_year - start_year + 1 > 1:
        for station in station_to_model_data:
            station_to_model_data[station] = pd.concat(station_to_model_data[station])
    else:
        for station in station_to_model_data:
            station_to_model_data[station] = station_to_model_data[station][0]



    # calculate observed climatology
    station_to_climatology = OrderedDict()
    for s in sorted(station_to_model_point, key=lambda st: st.latitude, reverse=True):
        assert isinstance(s, Station)
        print(s.id, len(s.get_list_of_complete_years()))

        # Check if there are continuous years for the selected period
        common_years = set(s.get_list_of_complete_years()).intersection(set(range(start_year, end_year + 1)))
        if len(common_years) > 0:
            _, station_to_climatology[s] = s.get_daily_climatology_for_complete_years_with_pandas(stamp_dates=stamp_dates,
                                                                                                  years=common_years)

            _, station_to_model_data[s] = pandas_utils.get_daily_climatology_from_pandas_series(station_to_model_data[s],
                                                                                                stamp_dates,
                                                                                                years_of_interest=common_years)


        else:
            print("Skipping {}, since it does not have enough data during the period of interest".format(s.id))







    # ---- Do the plotting ----
    ncols = 4

    nrows = len(station_to_climatology) // ncols
    nrows += int(not (len(station_to_climatology) % ncols == 0))

    axes_list = []
    plot_utils.apply_plot_params(width_cm=8 * ncols, height_cm=8 * nrows, font_size=8)
    fig = plt.figure()
    gs = GridSpec(nrows=nrows, ncols=ncols)




    for i, (s, clim) in enumerate(station_to_climatology.items()):
        assert isinstance(s, Station)

        row = i // ncols
        col = i % ncols

        print(row, col, nrows, ncols)

        # normalize by the drainage area
        if s.drainage_km2 is not None:
            station_to_model_data[s] *= s.drainage_km2 / station_to_model_point[s].accumulation_area

        if s.id in constants.stations_to_greyout:
            ax = fig.add_subplot(gs[row, col], facecolor="0.45")
        else:
            ax = fig.add_subplot(gs[row, col])

        assert isinstance(ax, Axes)

        ax.plot(stamp_dates, clim, color="k", lw=2, label="Obs.")
        ax.plot(stamp_dates, station_to_model_data[s], color="r", lw=2, label="Mod.")
        ax.xaxis.set_major_formatter(FuncFormatter(format_month_label))
        ax.xaxis.set_major_locator(MonthLocator(bymonthday=15))
        ax.xaxis.set_minor_locator(MonthLocator(bymonthday=1))
        ax.grid()





        ax.annotate(s.get_pp_name(), xy=(1.02, 1), xycoords="axes fraction",
                    horizontalalignment="left", verticalalignment="top", fontsize=8, rotation=-90)


        last_date = stamp_dates[-1]
        last_date = last_date.replace(day=calendar.monthrange(last_date.year, last_date.month)[1])

        ax.set_xlim(stamp_dates[0].replace(day=1), last_date)


        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax)


        if s.drainage_km2 is not None:
            ax.set_title("{}: ({:.1f}$^\circ$E, {:.1f}$^\circ$N, DA={:.0f} km$^2$)".format(s.id, s.longitude, s.latitude, s.drainage_km2))
        else:
            ax.set_title(
                "{}: ({:.1f}$^\circ$E, {:.1f}$^\circ$N, DA not used)".format(s.id, s.longitude, s.latitude))
        axes_list.append(ax)

    # plot the legend
    axes_list[-1].legend()


    if not img_folder.exists():
        img_folder.mkdir()

    fig.tight_layout()
    img_file = img_folder / "{}_{}-{}_{}.png".format(sim_label, start_year, end_year, "-".join(sorted(s.id for s in station_to_climatology)))

    print("Saving {}".format(img_file))
    fig.savefig(str(img_file), bbox_inches="tight", dpi=300)







if __name__ == '__main__':
    main()