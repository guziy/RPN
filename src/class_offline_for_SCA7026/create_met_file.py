import calendar
from collections import OrderedDict
import os
from datetime import timedelta, datetime

from rpn.rpn import RPN
from scipy.spatial.ckdtree import cKDTree
from util.geo import lat_lon

import pandas as pd
from multiprocessing import Process

__author__ = 'huziy'

from class_offline_for_SCA7026 import util_class_offline
import numpy as np
import shutil


def read_lon_lat_from_ini_file(path):
    """
    Get the longitude and latitude of the point corresponding to the path
    :param path:
    :return: :raise Exception:
    """

    with open(path) as f:
        for i, line in enumerate(f):
            if i == 3:
                lat, lon = [float(s) for s in line.split()[:2]]
                return lon, lat
        raise Exception("The file is too small: {}".format(path))


def cache_tree_generator(f):
    data = {}

    def cached(arg1, arg2):
        key = (arg1, arg2)
        if not key in data:
            data[key] = f(arg1, arg2)
        return data[key]

    return cached


# Decorator to cache the tree generator
@cache_tree_generator
def generate_kdtree(folder_with_ini_files, filename_prefix):
    lons, lats = [], []
    paths = []
    sel_files = [f for f in os.listdir(folder_with_ini_files) if f.startswith(filename_prefix) and f.lower().endswith(".ini")]
    xll, yll = None, None
    xur, yur = None, None
    for fname in sel_files:
        fpath = os.path.join(folder_with_ini_files, fname)
        lon, lat = read_lon_lat_from_ini_file(fpath)
        lons.append(lon)
        lats.append(lat)
        paths.append(fpath)

    x0, y0, z0 = lat_lon.lon_lat_to_cartesian(np.asarray(lons), np.asarray(lats))
    return cKDTree(list(zip(x0, y0, z0))), paths




def find_closest_ini_point(folder_with_ini_files="/home/huziy/skynet1_rech3/SCA7026_CLASS/offline_initialisation",
                           longitude=None, latitude=None,
                           filename_prefix="ERA40_NEW"):

    ktree, paths = generate_kdtree(folder_with_ini_files, filename_prefix)
    x, y, z = lat_lon.lon_lat_to_cartesian(longitude, latitude)
    dist, i = ktree.query((x, y, z))
    return paths[i]


def get_points_of_interest():
    return OrderedDict([
        ("YUMA",     (245.61, 32.58)),
        ("FLINT",    (276.83, 43.27)),
        ("WINNIPEG", (263.02, 49.79)),
        ("ORLANDO",  (278.50, 28.54)),
        ("PHOENIX",  (248.91, 33.46)),
        ("MONTREAL", (286.56, 45.72)),
        ("FAIRBANKS", (212.66, 64.91)),
        ("QUEBEC", (360 - 71.216667, 46.816667))
        ])


def interpolate_for_the_list_of_points():

    out_folder = "SCA7026"
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)


    for point, coords in get_points_of_interest().items():
        p = Process(target=main,
                    kwargs=dict(point_name=point, dest_lon=coords[0], dest_lat=coords[1], out_folder=out_folder))
        p.start()


def main(point_name="", dest_lon=None, dest_lat=None, out_folder=None):
    source_data_path = "/skynet1_rech3/camille/ERA_1958-2010"
    fname_pattern = "ERA-Interim_1979-2010_{}.rpn"
    # fname_pattern = "ERA_1958-2010_{}.rpn"

    varnames = [
        "TT", "UV", "SD", "AD", "HU", "P0", "PR"
    ]

    ignore_leap_year = True

    # point_name = "YUMA_erai"
    # dest_lat, dest_lon = 32.58, 245.61

    # point_name = "FLINT"
    # dest_lat, dest_lon = 43.27, 276.83

    # longitude should be in the range [-180, 180]
    dest_lon = dest_lon - 360 if dest_lon > 180 else dest_lon



    x0, y0, z0 = lat_lon.lon_lat_to_cartesian(dest_lon, dest_lat)

    start_year = 1980
    end_year = 2010
    out_step_min = 30  # interpolate in time if needed



    # Determine the index of the closest point
    tt_path = os.path.join(source_data_path, fname_pattern.format("TT"))

    robj = RPN(tt_path)
    tt = robj.get_first_record_for_name("TT")
    lons, lats = robj.get_longitudes_and_latitudes_for_the_last_read_rec()

    robj.close()

    lons_1d, lats_1d = lons.flatten(), lats.flatten()
    x, y, z = lat_lon.lon_lat_to_cartesian(lons_1d, lats_1d)

    ktree = cKDTree(list(zip(x, y, z)))
    dist, ind = ktree.query((x0, y0, z0))

    met_format = " {:>2}{:>3}{:>5}{:>6}" + 2 * "{:>9.2f}" + \
                 "{:>14.4E}{:>9.2f}{:>12.3E}{:>8.2f}{:>12.2f}" + "\n"


    shift_min = timedelta(minutes=int(dest_lon / 15.0 * 60))

    # write data to the text file
    with open(os.path.join(out_folder, "{}_{}-{}.MET".format(point_name, start_year, end_year)), "w") as f:
        df = pd.DataFrame()
        dates = None
        for vname in varnames:
            print("Reading {}".format(vname))
            r = RPN(os.path.join(source_data_path, fname_pattern.format(vname)))
            data = r.get_time_records_iterator_for_name_and_level(varname=vname)
            # select only dates from the range of interest and for the position of interest

            data = {k: v.flatten()[ind] for k, v in data if start_year <= k.year <= end_year + 1}

            if dates is None:
                dates = list(sorted(data.keys()))

            ts = pd.DataFrame(data=[data[k] for k in dates], index=dates, columns=[vname, ])

            ts = ts.asfreq(pd.DateOffset(minutes=out_step_min), method=None)
            method = "nearest"
            if vname in ["AD", "SD", "PR"]:
                ts.values[ts.values < 0] = 0.0
            elif vname == "HU":
                ts.values[ts.values <= 0] = 1.0e-10
            else:
                method = "linear"

            ts = ts.interpolate(method=method)

            df = pd.concat([df, ts], axis=1)

            r.close()

        print("Finished reading data into memory")

        df = df.select(crit=lambda d: not (d.month == 2 and d.day == 29 and ignore_leap_year))
        df = df.select(crit=lambda d: start_year <= d.year <= end_year)  # select the time window after interpolation

        for vname in varnames:
            print("{} ranges: min={}; max={}".format(vname, df[vname].min(), df[vname].max()))

        for d, row in df.iterrows():

            t_local = d + shift_min
            cos_zen = util_class_offline.get_cos_of_zenith_angle(dest_lat, t_local)
            # swrad = 0.0
            # if cos_zen > 0.0:
            swrad = row["SD"]

            day_of_year = d.timetuple().tm_yday
            if calendar.isleap(d.year):
                if d > datetime(d.year, 2, 29) and ignore_leap_year:
                    day_of_year -= 1

            line = met_format.format(
                d.hour, d.minute, day_of_year, d.year,
                swrad, row["AD"], row["PR"], row["TT"], row["HU"], row["UV"],
                row["P0"] * 100
            )
            f.write(line)

    # Get ini file
    src_ini = find_closest_ini_point(longitude=dest_lon, latitude=dest_lat)
    shutil.copyfile(src_ini, "{}/{}.INI".format(out_folder, point_name))


    print(dist, ind)
    print("Finished processing: {}".format(point_name))


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()
    #main()
    interpolate_for_the_list_of_points()

    # print find_closest_ini_point(longitude=245.61, latitude=32.58)
    # print find_closest_ini_point(longitude=276.83, latitude=43.27)
    # print find_closest_ini_point(longitude=360 - 73.5667, latitude=45.5)
    # print find_closest_ini_point(longitude=4.749146, latitude=27.708638)
    # print find_closest_ini_point(longitude=360 - 147.699890, latitude=64.848733)