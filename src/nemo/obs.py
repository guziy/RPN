from datetime import datetime, timedelta
import os
from iris.aux_factory import AuxCoordFactory
from iris.coords import AuxCoord
from iris.cube import Cube
import iris
from matplotlib.axes import Axes
from matplotlib.dates import date2num, DateFormatter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from iris.analysis import interpolate


__author__ = 'san'


class ObservationPoint(object):
    def __init__(self):
        pass


class AdcpProfileObs(ObservationPoint):
    def __init__(self):
        """
        Advanced Doppler current profiler data interface
        """
        super(AdcpProfileObs, self).__init__()

    # TODO: implement




class TempProfileObs(ObservationPoint):
    first_data_column_index = 4
    time_column_index = 3
    year_day_column_index = 2
    year_column_index = 1

    def __init__(self):
        super(TempProfileObs, self).__init__()
        self.longitude = None
        self.latitude = None
        self.data_frame = None
        self.levels = None

    def read_metadata_and_data(self, data_path="", header_path=""):
        self._read_meta_data(path=header_path)
        self._read_data(path=data_path)

    def _read_data(self, path=""):
        self.data_frame = pd.read_csv(path, sep=r"\s+", header=None, na_values=[-999.0, ])
        self.data_frame = self.data_frame.iloc[:, 1:]

        self.data_frame["date"] = self.data_frame[self.year_column_index].map(lambda x: datetime(x, 1, 1)) + \
                                  self.data_frame[self.year_day_column_index].map(lambda x: timedelta(days=x - 1))

        self.data_frame = self.data_frame.groupby(by="date").mean()

        del self.data_frame[3], self.data_frame[2], self.data_frame[1]

        assert isinstance(self.data_frame, pd.DataFrame)
        self.data_frame.columns = self.levels


    def get_dates(self):
        return self.data_frame.index

    def get_start_date(self):
        return self.data_frame.index.to_pydatetime()[0]

    def get_end_date(self):
        return self.data_frame.index.to_pydatetime()[-1]



    def _read_meta_data(self, path=""):
        """
        read header file
        :param path:
        :return:
        """
        look_for_coords = True

        self.levels = []
        look_for_levels = False
        with open(path) as f:
            for line in f:
                if look_for_coords:
                    look_for_coords = not self._extract_lonlat_from_line(line)

                if look_for_levels:
                    try:
                        self.levels.append(float(line.split()[4]))
                    except ValueError, e:
                        pass

                if line.startswith("***"):
                    look_for_levels = True

        print self.levels

    def _extract_lonlat_from_line(self, line):
        finished_reading = False
        if "LONGITUDE" in line:
            self.longitude = self._parse_lon_or_lat(line)
        elif "LATITUDE" in line:
            self.latitude = self._parse_lon_or_lat(line)
        return finished_reading

    @staticmethod
    def _parse_lon_or_lat(line):
        fields = line.split()
        deg, minutes, sec = [float(s) for s in fields[1].split("-")]

        multiplier = -1 if fields[2] in ["W", "S"] else 1
        return (deg + minutes / 60. + sec / 3600.0) * multiplier

    def __str__(self):
        return "Obs. point at (lon={}; lat={})".format(self.longitude, self.latitude)

    def plot_vertical_section(self):
        plt.figure()

        dates_num = date2num(self.data_frame.index.to_pydatetime())
        zz, tt = np.meshgrid(self.levels, dates_num)

        data = self.data_frame.values

        ax = plt.gca()
        im = ax.contourf(tt, zz, data, levels=np.arange(4, 30, 1))

        assert isinstance(ax, Axes)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(DateFormatter("%Y\n%b\n%d"))
        plt.colorbar(im)

    def get_tz_section_data(self):
        dates_num = date2num(self.data_frame.index.to_pydatetime())
        zz, tt = np.meshgrid(self.levels, dates_num)
        data = self.data_frame.values
        return tt, zz, data



def get_profile_for_prefix(prefix="", folder=""):
    po = TempProfileObs()
    folder = os.path.expanduser(folder)

    header_path = os.path.join(folder, "{}.header".format(prefix))
    data_path = os.path.join(folder, "{}.data".format(prefix))
    po.read_metadata_and_data(data_path=data_path, header_path=header_path)
    return po


def get_profile_for_testing():
    """
    Get a profile for testing
    :rtype : TempProfileObs
    """

    po = TempProfileObs()
    folder = os.path.expanduser("/RESCUE/skynet3_rech1/huziy/nemo_obs_for_validation/data_from_Ram_Yerubandi/Erie")

    # header_path = os.path.join(folder, "08-01T-004A024.120.290.header")
    # data_path = os.path.join(folder, "08-01T-004A024.120.290.data")
    # print data_path
    # po.read_metadata_and_data(header_path=header_path, data_path=data_path)
    # po.plot_vertical_section()

    header_path = os.path.join(folder, "08-01T-013A054.120.290.header")
    data_path = os.path.join(folder, "08-01T-013A054.120.290.data")
    print data_path
    po.read_metadata_and_data(header_path=header_path, data_path=data_path)
    # po.plot_vertical_section()
    return po

if __name__ == '__main__':
    po = TempProfileObs()
    folder = os.path.expanduser("~/NEMO/validation/from_Ram_Yerubandi")


    # header_path = os.path.join(folder, "08-01T-004A024.120.290.header")
    # data_path = os.path.join(folder, "08-01T-004A024.120.290.data")
    # print data_path
    # po.read_metadata_and_data(header_path=header_path, data_path=data_path)
    # po.plot_vertical_section()

    header_path = os.path.join(folder, "08-01T-013A054.120.290.header")
    data_path = os.path.join(folder, "08-01T-013A054.120.290.data")
    print data_path
    po.read_metadata_and_data(header_path=header_path, data_path=data_path)
    po.plot_vertical_section()

    # plt.show()



    model_data_path = "/Users/san/NEMO/outputs/GLK_1d_19790101_19790131_grid_T.nc"
    model_cube = iris.load_cube(model_data_path,
                                constraint=iris.Constraint(cube_func=lambda c: c.var_name == "votemper"))


    # model_cube.add_aux_coord(AuxCoord(depth_cube.data, units="m"))

    # print model_cube.coord("model_level_number").points[:]
    #po.get_vertical_section_of_model_bias(model_cube)

