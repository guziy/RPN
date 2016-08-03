import calendar
import glob
import re
from datetime import datetime

from pathlib import Path

from matplotlib.dates import DateFormatter, DateLocator, MonthLocator
from rpn.rpn import RPN
from rpn.rpn_multi import MultiRPN

from application_properties import main_decorator
from crcm5.mh_domains import default_domains
import data.cehq_station as cehq_station
from crcm5.model_point import ModelPoint
from data.cell_manager import CellManager
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from util import plot_utils

img_folder = Path("mh")




def get_model_data(model_point, station, output_path=None, grid_config=None, basins_of_interest_shp="",
                   cell_manager=None, vname=None):
    lons, lats, bmp = None, None, None
    data_mask = None

    assert isinstance(model_point, ModelPoint)
    assert isinstance(cell_manager, CellManager)
    assert isinstance(station, cehq_station.Station)


    monthly_diagnostics_case = False
    if output_path.name.lower().endswith("diagnostics"):
        fname_pattern = "pm*_moyenne"
        monthly_diagnostics_case = True
    else:
        fname_pattern = "pm*p"



    pattern = re.compile(".*" + 8 * "0" + ".*")

    flist = [f for f in glob.glob(str(output_path.joinpath("*").joinpath(fname_pattern))) if pattern.match(f) is None]

    r = MultiRPN(flist)

    date_to_field = r.get_all_time_records_for_name_and_level(varname=vname)

    lons, lats = r.get_longitudes_and_latitudes_of_the_last_read_rec()

    # get the basemap object
    bmp, data_mask = grid_config.get_basemap_using_shape_with_polygons_of_interest(
        lons, lats, shp_path=basins_of_interest_shp, mask_margin=5)


    upstream_mask = cell_manager.get_mask_of_upstream_cells_connected_with_by_indices(model_point.ix, model_point.jy)

    # Plot station position
    fig = plt.figure()

    ax = plt.gca()


    lons1, lats1 = lons[upstream_mask > 0.5], lats[upstream_mask > 0.5]
    x1, y1 = bmp(lons1, lats1)

    bmp.drawrivers()
    bmp.drawcoastlines(ax=ax)
    bmp.readshapefile(basins_of_interest_shp[:-4], "basin", linewidth=2, color="m")

    bmp.scatter(x1, y1, c="g", s=100)
    bmp.scatter(*bmp(lons[model_point.ix, model_point.jy], lats[model_point.ix, model_point.jy]), c="b", s=250)


    plt.savefig(str(img_folder.joinpath("{}_position_and_upstream.png".format(station.id))))

    r.close()



    res = pd.Series(index=sorted(date_to_field.keys()),
                    data=[date_to_field[d][model_point.ix, model_point.jy] for d in sorted(date_to_field.keys())])

    # get monthly means
    res = res.groupby(lambda d: d.replace(day=15)).mean()

    if monthly_diagnostics_case:
        # shift to the end of the month before a previous month, and then shift 15 days to later
        res = res.shift(-2, freq="M").shift(15, freq="D")

    print(res.index[:20])
    print(res.index[-20:])
    return res



@main_decorator
def main():

    # model_data_path = Path("/RECH2/huziy/BC-MH/bc_mh_044deg/Diagnostics")
    model_data_path = Path("/RECH2/huziy/BC-MH/bc_mh_044deg/Samples")

    static_data_file = "/RECH2/huziy/BC-MH/bc_mh_044deg/Samples/bc_mh_044deg_198001/pm1980010100_00000000p"

    corrected_obs_data_path = "mh/obs_data/Dauphin River Computed Unregulated Flows 1924 - 2015_ 05LM006.csv"

    r = RPN(static_data_file)

    fldir = r.get_first_record_for_name("FLDR")
    faa = r.get_first_record_for_name("FAA")
    lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()


    gc = default_domains.bc_mh_044

    cell_manager = CellManager(fldir, nx=fldir.shape[0], ny=fldir.shape[1],
                               lons2d=lons, lats2d=lats, accumulation_area_km2=faa)


    selected_station_ids = ["05LM006", ]

    stations = cehq_station.load_from_hydat_db(province=None, selected_ids=selected_station_ids, natural=None)




    # (06EA002): CHURCHILL RIVER AT SANDY BAY at (-102.31832885742188,55.52333068847656), accum. area is 212000.0 km**2
    # TODO: plot where is this station, compare modelled and observed hydrographs

    for s in stations:
        print(s)

    station_to_model_point = cell_manager.get_model_points_for_stations(stations, drainaige_area_reldiff_limit=0.1, nneighbours=16)

    print(station_to_model_point[stations[0]])

    station = stations[0]
    assert isinstance(station, cehq_station.Station)

    obs_not_corrected = pd.Series(index=station.dates, data=station.values).groupby(by=lambda d: d.replace(day=15)).mean()
    obs_corrected = pd.read_csv(corrected_obs_data_path, skiprows=2)

    print(obs_corrected.head())

    if hasattr(obs_corrected, "year"):
        # for monthly data
        date_index = pd.date_range(start=datetime(obs_corrected.year.iloc[0] - 1, 12, 15), end=datetime(obs_corrected.year.iloc[-1], 12, 15), freq="M")
        date_index = date_index.shift(15, freq=pd.datetools.day)
        factor = date_index.map(lambda d: 1000 / (calendar.monthrange(d.year, d.month)[1] * 24 * 3600))
        print(obs_corrected.year.iloc[0], obs_corrected.year.iloc[-1])
    elif hasattr(obs_corrected, "Date"):
        date_index = obs_corrected.Date.map(lambda s: datetime.strptime(s, "%b %d %Y"))
        print(date_index.iloc[0], date_index.iloc[-1])
        factor = date_index.map(lambda d: 1.0 / 0.028316846999968986)
    else:
        print(obs_corrected.head())
        raise IOError("Could not find time information in the obs file: ...")
    print(date_index)
    data = np.concatenate([r for r in obs_corrected.values[:, 1:-1]])



    print(factor[:10])
    obs_corrected = pd.Series(index=date_index, data=data / factor)



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