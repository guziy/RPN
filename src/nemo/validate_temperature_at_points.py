from datetime import datetime
from netCDF4 import num2date
import os
import re
import iris
from iris.analysis import interpolate
from scipy.spatial.ckdtree import cKDTree
from util.geo import lat_lon
import matplotlib.pyplot as plt

__author__ = 'huziy'

import pandas as pd
import numpy as np

EXP_DIR = "/home/huziy/nemo_glk/test_fwb_my"
#EXP_DIR = "/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/EXP_Luis_fwb2"


T_FILE_PATH, U_FILE_PATH, V_FILE_PATH = None, None, None

import application_properties
application_properties.set_current_directory()

for fname in os.listdir(EXP_DIR):
    if fname.endswith("_grid_T.nc"):
        T_FILE_PATH = os.path.join(EXP_DIR, fname)
    elif fname.endswith("_grid_U.nc"):
        U_FILE_PATH = os.path.join(EXP_DIR, fname)
    elif fname.endswith("_grid_V.nc"):
        V_FILE_PATH = os.path.join(EXP_DIR, fname)

from . import nemo_commons

EXP_NAME = os.path.basename(EXP_DIR)
NEMO_IMAGES_DIR = os.path.join("nemo", EXP_NAME, "temp_ts_validate")


if not os.path.isdir(NEMO_IMAGES_DIR):
    os.makedirs(NEMO_IMAGES_DIR)


class LakeObsStation(object):
    def __init__(self, path = None):
        print(path)
        self.longitude = None
        self.latitude = None
        self.id = None
        self.name = None

        if path is None:
            self.df = pd.DataFrame()
        else:
            self._load_from_file(path)


    def get_dates_and_values_to_plot(self):
        return self.df.index.to_pydatetime(), self.df["value"]


    def plot(self):
        return self.df.plot(style = "-")

    def _load_from_file(self, path):
        with open(path) as f:
            f.readline()
            line = f.readline()
            fields = line.split()

            self.id = fields[0]
            self.longitude = float(fields[2])
            self.latitude = float(fields[1])
            self.name = fields[3]

            line = f.readline()


            f.readline()
            data = []
            dates = []

            line = f.readline()
            while line:

                if line.strip() == "":
                    line = f.readline()
                    continue


                if "missing" in line.lower():
                    line = f.readline()
                    continue

                fields = line.split()



                if len(fields) < 13:
                    line = f.readline()
                    continue


                year = int(fields[0])


                vals = [float(x) for x in fields[1:]]
                data.extend(vals)
                dates.extend([datetime(year, m, 1) for m in range(1, 13)])

                line = f.readline()


            print(dates[0], dates[-1], self.latitude, self.longitude)

            data = (np.asarray(data) - 32.0) * 5.0 / 9.0

            self.df = pd.DataFrame(index=dates, data=data, columns=["value"])
            self.df = self.df[self.df["value"] > -99]


    def compare_with_modelled(self, data_cube, img_folder = None, ktree = None):
        """

        :param data_cube:
        :param img_folder:
        """

        assert isinstance(data_cube, iris.cube.Cube)
        print(self.id)

        time = data_cube.coord("time")
        dates_model = [num2date(t, units=str(time.units)) for t in time.points[:]]


        xt, yt, zt = lat_lon.lon_lat_to_cartesian(self.longitude, self.latitude)

        dists, inds = ktree.query(list(zip([xt,], [yt,], [zt,])))

        data_model = data_cube.data
        ntimes = data_model.shape[0]

        data_model.shape = (ntimes, -1)
        data_model = data_model[:, inds[0]]
        fig = plt.figure()
        ax = plt.gca()
        #self.df.plot(label="Obs", ax = ax)
        dates_obs, vals_obs = self.get_dates_and_values_to_plot()
        ax.plot(dates_obs, vals_obs, label = "Obs")
        ax.plot(dates_model, data_model, label = "Mod")
        print(data_model.min(), data_model.max())

        if data_model.sum() < 0.0001:
            plt.gcf().savefig(os.path.join(img_folder, "{0}.jpeg".format(self.id)))







        pass


def get_obs_data(data_folder = ""):
    """
    Read observation data into the list of stations
    """
    st_list = []
    for f_name in os.listdir(data_folder):
        if not f_name.lower().endswith(".dat"):
            continue

        f_path = os.path.join(data_folder, f_name)
        st_list.append(LakeObsStation(path=f_path))

    return st_list




def main():

    name_constraint = iris.Constraint(cube_func=lambda c: c.var_name == "sosstsst")
    data_cube = iris.load_cube(T_FILE_PATH, constraint=name_constraint)
    assert isinstance(data_cube, iris.cube.Cube)

    lons = data_cube.coord("longitude").points[:]
    lats = data_cube.coord("latitude").points[:]
    print(lons.shape, lats.shape)

    x, y, z = lat_lon.lon_lat_to_cartesian(lons.flatten(), lats.flatten())
    ktree = cKDTree(data=list(zip(x, y, z)))


    for st in get_obs_data(data_folder="/home/huziy/skynet3_rech1/nemo_obs_for_validation/temperature_at_points_ts"):
        st.compare_with_modelled(data_cube, img_folder=NEMO_IMAGES_DIR, ktree=ktree)


    #st.plot()
    import matplotlib.pyplot as plt
    plt.show()

if __name__ == "__main__":
    main()