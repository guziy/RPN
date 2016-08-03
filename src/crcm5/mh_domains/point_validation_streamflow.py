import calendar
from datetime import datetime

from pathlib import Path

from matplotlib.dates import DateFormatter, DateLocator, MonthLocator
from rpn.rpn import RPN
from rpn.rpn_multi import MultiRPN

from application_properties import main_decorator
from crcm5.mh_domains import default_domains
import data.cehq_station as cehq_station
from crcm5.mh_domains.point_validation_streamflow_from_2cols_per_station import get_model_data
from crcm5.model_point import ModelPoint
from data.cell_manager import CellManager
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from util import plot_utils

img_folder = Path("mh")







@main_decorator
def main():



    model_data_path = Path("/RECH2/huziy/BC-MH/bc_mh_044deg/Diagnostics")
    # model_data_path = Path("/RECH2/huziy/BC-MH/bc_mh_044deg/Samples")

    static_data_file = "/RECH2/huziy/BC-MH/bc_mh_044deg/Samples/bc_mh_044deg_198001/pm1980010100_00000000p"

    r = RPN(static_data_file)

    fldir = r.get_first_record_for_name("FLDR")
    faa = r.get_first_record_for_name("FAA")
    lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()


    gc = default_domains.bc_mh_044

    cell_manager = CellManager(fldir, nx=fldir.shape[0], ny=fldir.shape[1],
                               lons2d=lons, lats2d=lats, accumulation_area_km2=faa)


    selected_station_ids = ["06EA002", ]

    stations = cehq_station.load_from_hydat_db(province="SK", selected_ids=selected_station_ids, natural=None)


    # (06EA002): CHURCHILL RIVER AT SANDY BAY at (-102.31832885742188,55.52333068847656), accum. area is 212000.0 km**2
    # TODO: plot where is this station, compare modelled and observed hydrographs

    for s in stations:
        print(s)

    station_to_model_point = cell_manager.get_model_points_for_stations(stations, drainaige_area_reldiff_limit=0.08, nneighbours=16)

    print(station_to_model_point[stations[0]])

    station = stations[0]
    assert isinstance(station, cehq_station.Station)

    obs_not_corrected = pd.Series(index=station.dates, data=station.values).groupby(by=lambda d: d.replace(day=15)).mean()
    obs_corrected = pd.read_csv("mh/obs_data/Churchill Historic Monthly Apportionable Flow_06EA002.csv", skiprows=2)

    print(obs_corrected.head())
    print(obs_corrected.year.iloc[0], obs_corrected.year.iloc[-1])




    date_index = pd.date_range(start=datetime(obs_corrected.year.iloc[0] - 1, 12, 15),
                               end=datetime(obs_corrected.year.iloc[-1], 12, 15),
                               freq="M")

    date_index = date_index.shift(15, freq=pd.datetools.day)

    print(date_index)
    data = np.concatenate([r for r in obs_corrected.values[:, 1:-1]])


    factor = date_index.map(lambda d: 1000 / (calendar.monthrange(d.year, d.month)[1] * 24 * 3600))
    print(factor[:10])
    obs_corrected = pd.Series(index=date_index, data=data * factor)



    modelled_data = get_model_data(model_point=station_to_model_point[station], station=station, output_path=model_data_path,
                                   grid_config=gc, basins_of_interest_shp=default_domains.MH_BASINS_PATH,
                                   cell_manager=cell_manager, vname="STFL")

    fig = plt.figure()
    ax = obs_corrected.plot(label="obs corrected")

    obs_not_corrected.plot(label="obs not corrected", ax=ax, color="k")

    modelled_data.plot(label="CRCM5", ax=ax, color="r")

    ax.legend(loc="upper left")
    img_file = img_folder.joinpath("{}_validation_monthly.png".format(station.id))
    fig.savefig(str(img_file))
    plt.close(fig)


    # climatology
    start_year = 1980
    end_year = 2010
    fig = plt.figure()
    ax = obs_corrected.select(lambda d: start_year <= d.year <= end_year).groupby(lambda d: d.replace(year=2001)).mean().plot(label="obs corrected")

    obs_not_corrected.select(lambda d: start_year <= d.year <= end_year).groupby(lambda d: d.replace(year=2001)).mean().plot(label="obs not corrected", ax=ax, color="k")

    modelled_data.select(lambda d: start_year <= d.year <= end_year).groupby(lambda d: d.replace(year=2001)).mean().plot(label="CRCM5", ax=ax, color="r")

    ax.xaxis.set_major_locator(MonthLocator(bymonthday=15))
    ax.xaxis.set_major_formatter(DateFormatter("%b"))

    ax.legend(loc="upper left")

    img_file = img_folder.joinpath("{}_validation_clim.png".format(station.id))
    fig.savefig(str(img_file))
    plt.close(fig)


    # Interannual variability
    fig = plt.figure()


    obs_corrected = obs_corrected.select(lambda d: start_year <= d.year <= end_year)
    modelled_data = modelled_data.select(lambda d: start_year <= d.year <= end_year)

    corr_list = []
    for m in range(1, 13):
        obs = obs_corrected.select(lambda d: d.month == m)
        mod = modelled_data.select(lambda d: d.month == m)


        print(obs.head())

        obs.index = obs.index.map(lambda d: d.year)
        mod.index = mod.index.map(lambda d: d.year)

        corr_list.append(obs.corr(mod))

    ax = plt.gca()
    ax.plot(range(1, 13), corr_list)
    ax.set_xlabel("Month")
    ax.set_title("Inter-annual variability")




    img_file = img_folder.joinpath("{}_interannual.png".format(station.id))
    fig.savefig(str(img_file))
    plt.close(fig)




if __name__ == '__main__':
    plot_utils.apply_plot_params()
    main()