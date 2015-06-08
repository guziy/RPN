from netCDF4 import num2date, Dataset
import os
import pickle
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.spatial.kdtree import KDTree
import application_properties
from cru.temperature import CRUDataManager

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import numpy as np
from util.geo import lat_lon

__author__ = 'huziy'


class SweDataManager(CRUDataManager):
    def __init__(self, path="data/swe_ross_brown/swe.nc", var_name=""):
        self.lons2d, self.lats2d = None, None
        self.times = None
        self.var_data = None
        CRUDataManager.__init__(self, path=path, var_name=var_name)
        print(list(self.nc_dataset.variables.keys()))

        pass

    def _init_fields(self, nc_dataset):
        print("init_fields")
        nc_vars = nc_dataset.variables
        times = nc_vars["time"][:]

        lons = nc_vars["longitude"][:]
        lats = nc_vars["latitude"][:]

        self.lons2d, self.lats2d = lons, lats

        time_units_s = nc_vars["time"].units
        self.times_var = nc_vars["time"]
        self.times_num = nc_vars["time"][:]
        self.times = num2date(times, time_units_s)
        if not self.lazy:
            self.var_data = nc_vars[self.var_name][:]

        x_in, y_in, z_in = lat_lon.lon_lat_to_cartesian(self.lons2d.flatten(), self.lats2d.flatten())
        self.kdtree = KDTree(list(zip(x_in, y_in, z_in)))
        print("SWE obs time limits: ", self.times[0], self.times[-1])

    def get_mean_for_year_and_months(self, year, months=None):
        bool_vector = np.array([(x.year == year) and (x.month in months) for x in self.times])
        assert self.var_data is not None
        return np.mean(self.var_data[bool_vector, :, :], axis=0)


    def save_period_means_to_file(self, months=None, year_range=range(1980, 1997),
                                  path="djf_swe_ross_brown.nc"):
        ds = Dataset(path, mode="w", format="NETCDF3_CLASSIC")
        ds.createDimension('year', len(year_range))
        ds.createDimension('lon', self.lons2d.shape[0])
        ds.createDimension('lat', self.lons2d.shape[1])

        lonVariable = ds.createVariable('longitude', 'f4', ('lon', 'lat'))
        latVariable = ds.createVariable('latitude', 'f4', ('lon', 'lat'))
        yearVariable = ds.createVariable("year", "i4", ("year",))

        altVariable = ds.createVariable("SWE", "f4", ('year', 'lon', 'lat'))
        altVariable.units = "mm"

        for i, the_year in enumerate(year_range):
            altVariable[i, :, :] = self.get_mean_for_year_and_months(the_year, months=months)

        lonVariable[:, :] = self.lons2d[:, :]
        latVariable[:, :] = self.lats2d[:, :]
        yearVariable[:] = year_range

        ds.close()


    def save_projected_means_to_file(self, months=None, year_range=range(1980, 1997),
                                     path="djf_swe_ross_brown_on_cordex.nc",
                                     dest_lons2d=None, dest_lats2d=None):

        ds = Dataset(path, mode="w", format="NETCDF3_CLASSIC")
        ds.createDimension('year', len(year_range))
        ds.createDimension('lon', dest_lons2d.shape[0])
        ds.createDimension('lat', dest_lons2d.shape[1])

        lon_variable = ds.createVariable('longitude', 'f4', ('lon', 'lat'))
        lat_variable = ds.createVariable('latitude', 'f4', ('lon', 'lat'))
        year_variable = ds.createVariable("year", "i4", ("year",))

        swe_variable = ds.createVariable("SWE", "f4", ('year', 'lon', 'lat'))
        swe_variable.units = "mm"

        for i, the_year in enumerate(year_range):
            data = self.get_mean_for_year_and_months(the_year, months=months)
            swe = self.interpolate_data_to(data, dest_lons2d, dest_lats2d, nneighbours=1)
            swe_variable[i, :, :] = swe

        lon_variable[:, :] = dest_lons2d[:, :]
        lat_variable[:, :] = dest_lats2d[:, :]
        year_variable[:] = year_range

        ds.close()

        pass


    def get_mean(self, start_year, end_year, months=None):
        """
        Get mean using pandas, overriden, not implemented for the lazy case
        :param start_year:
        :param end_year:
        :param months:
        :return:
        """
        import pandas as pd
        # Use cache file for performance
        cache_file = "swe_obs_{0}-{1}_".format(start_year, end_year) + "-".join([str(x) for x in months]) + ".cache"
        if os.path.isfile(cache_file):
            print("Using cached SWE data from {0}".format(cache_file))
            return pickle.load(open(cache_file, "rb"))
        if self.lazy:
            # TODO: implement
            raise NotImplementedError()
        else:
            nx, ny = self.lons2d.shape
            data_panel = pd.Panel(data=self.nc_vars[self.var_name][:], items=self.times,
                                  major_axis=list(range(nx)), minor_axis=list(range(ny)))
            data_panel = data_panel.select(
                lambda d: (d.month in months) and (d.year >= start_year) and d.year <= end_year)
            df = data_panel.mean(axis="items")
        mean_field = df.values
        pickle.dump(mean_field, open(cache_file, "wb"))
        return mean_field

    def getMeanFieldForMonthsInterpolatedTo(self, months=None,
                                            lons_target=None, lats_target=None,
                                            start_year=None, end_year=None):
        """
        Get mean over months and interpolate the result to the target longitudes and latitudes
        :param months:
        :param lons_target:
        :param lats_target:
        :param start_year:
        :param end_year:
        :return:
        """

        print(self.var_name)

        mean_field = self.get_mean(start_year, end_year, months = months)
        assert mean_field.shape == self.lons2d.shape, "data shape: ({0}, {1})".format(*mean_field.shape) + \
                                                      "coordinates shape: ({0}, {1})".format(*self.lons2d.shape)
        interp_field = self.interpolate_data_to(mean_field, lons_target, lats_target, nneighbours=1)
        return interp_field


def main():
    from permafrost import draw_regions

    dm = SweDataManager(var_name="SWE")

    b, lons2d, lats2d = draw_regions.get_basemap_and_coords()

    x, y = b(dm.lons2d, dm.lats2d)

    fig = plt.figure()

    start_year = 1981
    end_year = 1997

    levels = [10, ] + list(range(20, 120, 20)) + [150, 200, 300, 500, 1000]
    cmap = mpl.cm.get_cmap(name="jet_r", lut=len(levels))
    norm = colors.BoundaryNorm(levels, cmap.N)

    gs = gridspec.GridSpec(1, 2)
    ax = fig.add_subplot(gs[0, 0])
    data = dm.get_mean(start_year, end_year, months=[3])
    img = b.contourf(x, y, data.copy(), ax=ax, cmap=cmap, norm=norm, levels=levels)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = fig.colorbar(img, cax=cax)
    b.drawcoastlines(ax=ax)
    ax.set_title("SWE (not interp.), \n DJF period: {0} - {1}".format(start_year, end_year))

    ax = fig.add_subplot(gs[0, 1])
    data_projected = dm.interpolate_data_to(data, lons2d, lats2d, nneighbours=1)
    x, y = b(lons2d, lats2d)
    img = b.contourf(x, y, data_projected, ax=ax, levels=img.levels)

    #add pretty colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = fig.colorbar(img, cax=cax)

    b.drawcoastlines(ax=ax)
    ax.set_title("SWE ( interp.), \n DJF period: {0} - {1}".format(start_year, end_year))

    plt.savefig("swe_rb_djf.png")

    pass


def test1():
    dm = SweDataManager(var_name="SWE")
    #b, lons2d, lats2d = draw_regions.get_basemap_and_coords()
    #dm.save_projected_means_to_file(months=[12,1,2], dest_lons2d=lons2d, dest_lats2d=lats2d)
    print(dm.kdtree)


if __name__ == "__main__":
    application_properties.set_current_directory()
    #main()
    test1()
    print("Hello world")
  