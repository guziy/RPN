from datetime import datetime
from netCDF4 import Dataset
import os
import pickle
import re
from collections import OrderedDict
import pandas as pd
import numpy as np
__author__ = 'huziy'

PRECIP_NAME = "precip"
PRECIP_UNITS = "mm/s"


def _get_year_list_from_name(filename = ""):

    """
    >>> folder_path = "/skynet3_rech1/huziy/NEMO_OFFICIAL/DFS5.2/precip"
    >>> for fname in os.listdir(folder_path):
    ...        if "-" not in fname: continue
    ...        _get_year_list_from_name(fname)
    [1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978]

    >>> for fname in os.listdir(folder_path):
    ...     if "1979" not in fname: continue
    ...     _get_year_list_from_name(fname)
    [1979]

    :param filename:
    :return:
    """
    groups = re.findall(r"\d+", filename)
    if "-" not in filename:
        year = int(groups[-1])
        return [year, ]

    start, end = [int(token) for token in groups[-2:]]
    return range(start, end + 1)





class DFSDataManager(object):

    def __init__(self, folder_path = "/skynet3_rech1/huziy/NEMO_OFFICIAL/DFS5.2", var_name = "t2"):
        self.folder_path = folder_path
        self.var_folder = os.path.join(self.folder_path, var_name)
        self.var_name = var_name
        self.year_2_path = {}

        for fname in os.listdir(self.var_folder):
            for y in _get_year_list_from_name(fname):
                self.year_2_path[y] = os.path.join(self.var_folder, fname)


    def get_lons_and_lats_2d(self):
        fname = os.listdir(self.var_folder)[0]
        ds = Dataset(os.path.join(self.var_folder, fname))
        lons = None
        lats = None
        for vname, varnc in ds.variables.iteritems():
            print vname
            if vname.lower().startswith("lon"):
                lons = varnc[:]
            elif vname.lower().startswith("lat"):
                lats = varnc[:]

        if lons.ndim == 1:
            lons, lats = np.meshgrid(lons, lats)


        print lons.shape
        print lats.shape
        assert lons.shape == lats.shape
        ds.close()
        return lons, lats

    def get_seasonal_means(self, season_name_to_months = None,
                           start_year = None,
                           end_year = None):
        """
        Year range is inclusive i.e. [start_year, end_year]
        :param season_name_to_months:
        :param start_year:
        :param end_year:
        """
        if season_name_to_months is None:
            season_name_to_months = OrderedDict([
                ("Winter", (1, 2, 12)),
                ("Spring", range(3, 6)),
                ("Summer", range(6, 9)),
                ("Fall", range(9, 12))])

        if None in [start_year, end_year]:
            start_year = min(self.year_2_path.keys())
            end_year = max(self.year_2_path.keys())

        seasonal_cache_file_name = "seasonal_{0}_".format("-".join(season_name_to_months.keys()))
        seasonal_cache_file_name += "{0}_{1}-{2}.cache".format(self.var_name, start_year, end_year)
        cache_path = os.path.join(self.folder_path, seasonal_cache_file_name)
        print "Cache file {0}".format(cache_path)
        if os.path.isfile(cache_path):
            return pickle.load(open(cache_path))

        #create month to season map
        month_to_season = {}
        for sname, months in season_name_to_months.iteritems():
            for m in months:
                month_to_season[m] = sname


        seasonal_panels = []
        for the_year in range(start_year, end_year + 1):
            print "DFS: processing year {0}".format(the_year)
            ds = Dataset(self.year_2_path[the_year])
            data = ds.variables[self.var_name][:]
            ds.close()
            nt, ny, nx = data.shape
            print "nt = {0}, ny = {1}, nx = {2}".format(nt, ny, nx)
            year_start = datetime(the_year, 1, 1)
            year_end = datetime(the_year + 1, 1, 1)

            dt = (year_end - year_start) / nt
            panel = pd.Panel(data=data, items=[year_start + i * dt for i in range(nt)],
                             major_axis=range(ny), minor_axis=range(nx))
            panel_seasons = panel.groupby(lambda d: month_to_season[d.month], axis = "items").mean()
            seasonal_panels.append(panel_seasons)


        season_to_mean = OrderedDict()
        for sname, _ in season_name_to_months.iteritems():
            season_to_mean[sname] = np.asarray([p[sname].values for p in seasonal_panels]).mean(axis = 0)
            print season_to_mean[sname].shape


        pickle.dump(season_to_mean, open(cache_path, "w"))
        return season_to_mean



    def get_daily_climatology(self, start_year = None, end_year = None, var_name = "t2"):

        for the_year in range(start_year, end_year + 1):
            ds = Dataset(self.year_2_path[the_year])
            data = ds.variables[var_name][:]
            nt, ny, nx = data.shape
            #TODO: implement

        #TODO: implement
        pass


def check():
    dm = DFSDataManager()
    s2m = dm.get_seasonal_means(start_year=1979, end_year=1980)
    import os
    print os.getcwd()


    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    b = Basemap(lon_0=180)
    lons, lats = dm.get_lons_and_lats_2d()
    x, y = b(lons, lats)
    for sname, smean in s2m.iteritems():
        plt.figure()
        plt.title(sname)
        im = b.pcolormesh(x, y, smean - 273.15)
        b.drawcoastlines()
        plt.colorbar(im)

    plt.show()



if __name__ == "__main__":
    check()