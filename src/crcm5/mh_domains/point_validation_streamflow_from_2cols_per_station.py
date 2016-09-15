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

from domains.grid_config import GridConfig
from util import plot_utils

img_folder = Path("mh")


def get_model_data(station_to_model_point, output_path=None, grid_config=None, basins_of_interest_shp="",
                   cell_manager=None, vname=None):
    lons, lats, bmp = None, None, None
    data_mask = None

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

    r.close()

    # get the basemap object
    bmp, data_mask = grid_config.get_basemap_using_shape_with_polygons_of_interest(
        lons, lats, shp_path=basins_of_interest_shp, mask_margin=5)

    station_to_model_data = {}  # model data are the pandas timeseries

    stations_to_ignore = []

    for station, model_point in station_to_model_point.items():
        assert isinstance(model_point, ModelPoint)
        assert isinstance(cell_manager, CellManager)
        assert isinstance(station, cehq_station.Station)

        upstream_mask = cell_manager.get_mask_of_upstream_cells_connected_with_by_indices(model_point.ix,
                                                                                          model_point.jy)

        # Skip model points and staions with small number of gridcells upstream
        if upstream_mask.sum() <= 1:
            stations_to_ignore.append(station)
            print("Station {} is ignored, because the number of upstream cells is <= 1.".format(station.id))
            continue

        # Skip model points and stations outside the region of interest
        if not data_mask[model_point.ix, model_point.jy]:
            stations_to_ignore.append(station)
            print("Station {} is ignored, because it is outside of the domain of interest.".format(station.id))
            continue

        # Plot station position
        fig = plt.figure()

        ax = plt.gca()

        lons1, lats1 = lons[upstream_mask > 0.5], lats[upstream_mask > 0.5]
        x1, y1 = bmp(lons1, lats1)

        bmp.drawrivers()
        bmp.drawcoastlines(ax=ax)
        bmp.drawcountries(ax=ax, linewidth=0.2)
        bmp.drawstates(linewidth=0.1)
        bmp.readshapefile(basins_of_interest_shp[:-4], "basin", linewidth=2, color="m")

        bmp.scatter(x1, y1, c="g", s=100)
        bmp.scatter(*bmp(lons[model_point.ix, model_point.jy], lats[model_point.ix, model_point.jy]), c="b", s=250)

        fig.tight_layout()
        plt.savefig(str(img_folder.joinpath("{}_position_and_upstream.png".format(station.id))), bbox_inche="tight")
        plt.close(fig)


        res = pd.Series(index=sorted(date_to_field.keys()),
                        data=[date_to_field[d][model_point.ix, model_point.jy] for d in sorted(date_to_field.keys())])

        # get monthly means
        res = res.groupby(lambda d: d.replace(day=15, hour=0)).mean()

        if monthly_diagnostics_case:
            # shift to the end of the month before a previous month, and then shift 15 days to later
            res = res.shift(-2, freq="M").shift(15, freq="D")

        print(res.index[:20])
        print(res.index[-20:])

        station_to_model_data[station] = res


    # Not enough drainage area
    for s in stations_to_ignore:
        del station_to_model_point[s]

    return station_to_model_data


def plot_validations_for_stations(station_to_model_point=None, cell_manager=None, corrected_obs_data_folder=None,
                                  model_data_path=None, grid_config=None, start_year=None, end_year=None):

    assert isinstance(grid_config, GridConfig)

    station_to_modelled_data = get_model_data(station_to_model_point=station_to_model_point,
                                              output_path=model_data_path,
                                              grid_config=grid_config,
                                              basins_of_interest_shp=default_domains.MH_BASINS_PATH,
                                              cell_manager=cell_manager, vname="STFL")

    for station, model_point in station_to_model_point.items():

        corrected_obs_data_path = None
        for f in corrected_obs_data_folder.iterdir():
            if f.name.upper()[:-4].endswith(station.id):
                corrected_obs_data_path = str(f)
                print("Corrected station data is from {}.".format(f))
                break


        obs_not_corrected = pd.Series(index=station.dates, data=station.values).groupby(lambda d: d.replace(day=15, hour=0)).mean()

        print(obs_not_corrected.head())



        obs_corrected = None
        if corrected_obs_data_path is not None:
            obs_corrected = pd.read_csv(corrected_obs_data_path, skiprows=2)
            print(obs_corrected.head())

            if hasattr(obs_corrected, "year"):
                # for monthly data
                date_index = pd.date_range(start=datetime(obs_corrected.year.iloc[0] - 1, 12, 15),
                                           end=datetime(obs_corrected.year.iloc[-1], 12, 15), freq="M")
                date_index = date_index.shift(15, freq=pd.datetools.day)
                factor = date_index.map(lambda d: 1000 / (calendar.monthrange(d.year, d.month)[1] * 24 * 3600))
                print(obs_corrected.year.iloc[0], obs_corrected.year.iloc[-1])

                data = np.concatenate([r for r in obs_corrected.values[:, 1:-1]]) * factor.values

            elif hasattr(obs_corrected, "Date"):
                obs_corrected.dropna(inplace=True)

                # Check several date formats before giving up
                date_formats = ["%b %d %Y", "%Y-%m-%d"]

                assert isinstance(obs_corrected, pd.DataFrame)

                value_error = None
                for date_format in date_formats:
                    try:
                        date_index = obs_corrected.Date.map(lambda s: datetime.strptime(s, date_format))
                        value_error = None
                        break
                    except ValueError as ve:
                        value_error = ve

                if value_error is not None:
                    raise value_error


                print(date_index.iloc[0], date_index.iloc[-1])

                colnames = list(obs_corrected)
                if "m3/s" in colnames[-1].lower().replace(" ", ""):
                    factor = date_index.map(lambda d: 1)
                elif "cfs" in colnames[-1].lower().replace(" ", ""):
                    factor = date_index.map(lambda d: 0.028316846999968986)
                else:
                    raise IOError("Unknown streamflow units in {}".format(corrected_obs_data_path))

                data = obs_corrected.values[:, -1] * factor.values

            else:
                print(obs_corrected.head())
                raise IOError("Could not find time information in the obs file: ...")

            # Time series of the corrected obs
            obs_corrected = pd.Series(index=date_index, data=data, dtype=np.float64)


        # Get model data corresponding to the station
        modelled_data = station_to_modelled_data[station]

        # Select the time interval
        def date_selector(d):
            return (start_year <= d.year <= end_year) and not ((d.month == 2) and (d.day == 29))



        if obs_corrected is not None:
            obs_corrected = obs_corrected.select(date_selector)

            if len(obs_corrected) == 0:
                obs_corrected = None


        obs_not_corrected = obs_not_corrected.select(date_selector)

        modelled_data = modelled_data.select(date_selector)




        # debug ====
        # import pickle
        # pickle.dump(modelled_data, open("debug_model_ts.bin", "wb"))


        # Monthly means =======================================================================
        fig = plt.figure()

        ax = modelled_data.plot(label="CRCM5", color="r")

        if len(obs_not_corrected) > 0:
            obs_not_corrected.plot(label="obs not corrected", color="k", ax=ax)

        if obs_corrected is not None:
            obs_corrected.plot(label="obs corrected", ax=ax)

        ax.legend(loc="upper left")
        img_file = img_folder.joinpath("{}_validation_monthly.png".format(station.id))
        fig.savefig(str(img_file))
        plt.close(fig)

        # climatology =========================================================================
        fig = plt.figure()
        ax = modelled_data.groupby(lambda d: d.replace(year=2001, day=15, hour=0, second=0)).mean().plot(label="CRCM5", color="r")

        if len(obs_not_corrected) > 0:
            obs_not_corrected.groupby(lambda d: d.replace(year=2001, day=15, hour=0)).mean().plot(label="obs not corrected", ax=ax, color="k")

        if obs_corrected is not None:
            obs_corrected.groupby(lambda d: d.replace(year=2001, day=15, hour=0)).mean().plot(label="obs corrected", ax=ax)

        ax.xaxis.set_major_locator(MonthLocator(bymonthday=15))
        ax.xaxis.set_major_formatter(DateFormatter("%b"))
        ax.legend(loc="upper left")

        img_file = img_folder.joinpath("{}_validation_clim.png".format(station.id))
        fig.savefig(str(img_file))
        plt.close(fig)

        # Interannual variability =================================================================

        corr_list = []
        for m in range(1, 13):
            if obs_corrected is not None:
                obs = obs_corrected.select(lambda d: d.month == m)
            else:
                obs = obs_not_corrected.select(lambda d: d.month == m)

            if len(obs) == 0:
                break

            mod = modelled_data.select(lambda d: d.month == m)

            print(obs.head())

            obs.index = obs.index.map(lambda d: d.year)
            mod.index = mod.index.map(lambda d: d.year)

            corr_list.append(obs.corr(mod))

        if len(corr_list) > 0:
            fig = plt.figure()

            ax = plt.gca()
            ax.plot(range(1, 13), corr_list)
            ax.set_xlabel("Month")
            ax.set_title("Inter-annual variability")

            img_file = img_folder.joinpath("{}_interannual.png".format(station.id))
            fig.savefig(str(img_file))
            plt.close(fig)


@main_decorator
def main():
    start_year = 1980
    end_year = 2010


    # model_data_path = Path("/RECH2/huziy/BC-MH/bc_mh_044deg/Diagnostics")
    model_data_path = Path("/RECH2/huziy/BC-MH/bc_mh_044deg/Samples")

    static_data_file = "/RECH2/huziy/BC-MH/bc_mh_044deg/Samples/bc_mh_044deg_198001/pm1980010100_00000000p"

    corrected_obs_data_folder = Path("mh/obs_data/")

    r = RPN(static_data_file)

    fldir = r.get_first_record_for_name("FLDR")
    faa = r.get_first_record_for_name("FAA")
    lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()

    gc = default_domains.bc_mh_044

    cell_manager = CellManager(fldir, nx=fldir.shape[0], ny=fldir.shape[1],
                               lons2d=lons, lats2d=lats, accumulation_area_km2=faa)

    selected_station_ids = [
        "05LM006",
        "05BN012",
        "05AK001",
        "05QB003"
    ]

    stations = cehq_station.load_from_hydat_db(province=None, selected_ids=selected_station_ids, natural=None, skip_data_checks=True)


    for s in stations:
        assert isinstance(s, cehq_station.Station)
        if s.id == "05AK001":
            s.drainage_km2 *= 2.5

        if s.id == "05BN012":
            pass


    # Manitoba natural stations
    # statons_mnb = cehq_station.load_from_hydat_db(province="MB", natural=True, start_date=datetime(start_year, 1, 1), end_date=datetime(end_year,12,31))
    # statons_ssk = cehq_station.load_from_hydat_db(province="SK", natural=True, start_date=datetime(start_year, 1, 1), end_date=datetime(end_year,12,31))
    # statons_alb = cehq_station.load_from_hydat_db(province="AB", natural=True, start_date=datetime(start_year, 1, 1), end_date=datetime(end_year,12,31))


    # for s in statons_mnb + statons_ssk + statons_alb:
    #     if s not in stations:
    #         stations.append(s)


    # (06EA002): CHURCHILL RIVER AT SANDY BAY at (-102.31832885742188,55.52333068847656), accum. area is 212000.0 km**2
    # TODO: plot where is this station, compare modelled and observed hydrographs

    for s in stations:
        print(s)

    # assert len(stations) == len(selected_station_ids), "Could not find stations for some of the specified ids"

    station_to_model_point = cell_manager.get_model_points_for_stations(stations, drainaige_area_reldiff_limit=0.9,
                                                                        nneighbours=8)


    print("Established the station to model point mapping")


    plot_validations_for_stations(station_to_model_point,
                                  cell_manager=cell_manager,
                                  corrected_obs_data_folder=corrected_obs_data_folder,
                                  model_data_path=model_data_path,
                                  grid_config=gc, start_year=start_year, end_year=end_year)


if __name__ == '__main__':
    plot_utils.apply_plot_params()
    main()
