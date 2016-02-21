from datetime import datetime
from pathlib import Path

import lxml
from mpl_toolkits.basemap import Basemap

from application_properties import main_decorator
from pykml import parser

from data.cehq_station import Station

import pandas as pd

OBS_DATA_FOLDER = Path("/home/huziy/skynet3_rech1/cc-paper-lakeobs-data/ice_thickness_point_data")

STATION_COORDS_FILE = OBS_DATA_FOLDER.joinpath("Originalicethickness.kml")
STATION_DATA_FILE = OBS_DATA_FOLDER.joinpath("original_program_data_20030304.csv")

import matplotlib.pyplot as plt
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis


def get_obs_data():
    df = pd.read_csv(str(STATION_DATA_FILE), sep=";", encoding="ISO-8859-1",
                     skiprows=1, usecols=range(4))
    """
    :type : pd.DataFrame
    """

    df.Date = df.Date.map(lambda s: datetime.strptime(s, "%y-%m-%d"))
    df["ice_depth"] = df.ix[:, -1]
    df["station_id"] = df.ix[:, 0]
    df["station_name"] = df.ix[:, 1]
    df["year"] = df.Date.map(lambda d: d.year)
    return df


def get_station_objects(start_year=1980, end_year=2010, sel_names=None):
    # read ice depth values
    df = get_obs_data()

    lon_min, lon_max = -100, 0
    lat_min, lat_max = 40, 90

    nvals_min = 100

    p = parser.parse(STATION_COORDS_FILE.open())

    root = p.getroot()

    station_elts = root.Document.Placemark

    # select points based on the lat/lon limits?
    stations = []
    for el in station_elts:

        lon, lat, _ = [float(c.strip()) for c in el.Point.coordinates.text.split(",")]

        # Check if the station
        if sel_names is not None:

            is_ok = False

            for sel_name in sel_names:
                if sel_name.lower() in el.name.text.lower():
                    is_ok = True
                    break

            if not is_ok:
                continue

        if (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max):
            print("{}: {}".format(el.name, el.Point.coordinates))

            df_s = df.loc[df.station_name.str.lower().str.startswith(el.name.text.lower())]

            df_s = df_s.loc[(df_s.year >= start_year) & (df_s.year <= end_year)]

            if len(df_s) < nvals_min:
                continue

            print(len(df_s))
            d_to_v = dict(zip(df_s["Date"][:], df_s["ice_depth"][:]))

            # df_s.plot(x="Date", y="ice_depth")
            # plt.title(el.name.text)
            # plt.show()

            # print(df_s.station_name)

            stations.append(Station(st_id=df_s.station_name.iloc[0], lon=lon, lat=lat, date_to_value=d_to_v))

    return stations


@main_decorator
def main():
    selected_names = ["South Baymouth", "Frechette Point",
                      "Kashagawigamog Lake",
                      "MATAGAMI", "Chemung Lake", "Moosonee",
                      "Botwood", "Kuujjuaq",
                      "Kashagawigamog Lake", "Canal Lake"]
    stations = get_station_objects(sel_names=selected_names)

    print(len(stations))

    model_data_path = "/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r.hdf5"

    ice_depth_varname = "LD"

    print([s.id for s in stations])

    lons, lats, nvals = list(zip(*[(s.longitude, s.latitude, len(s)) for s in stations]))
    df = analysis.get_timeseries_for_for_points(lons, lats,
                                                data_path=model_data_path,
                                                varname=ice_depth_varname)



    for i, s in enumerate(stations):
        assert isinstance(s, Station)
        plt.figure()
        plt.title(s.id)

        plt.plot(s.dates, s.values, "o", label="Obs.")
        plt.plot(df.index, df[i] * 100, label="Mod.", lw=2)
        plt.legend()

    plt.show()




    # b = Basemap()

    #
    # x, y = b(lons, lats)
    #
    # cols = b.scatter(x, y, c=nvals)
    # b.drawcoastlines()
    # b.colorbar(cols)
    # plt.show()


if __name__ == '__main__':
    main()
