from datetime import timedelta, datetime
import itertools
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.spatial.kdtree import KDTree
import application_properties
from data.timeseries import TimeSeries

from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np
from netCDF4 import Dataset, num2date, date2num
import matplotlib.pyplot as plt


class CRUDataManager:
    def __init__(self, path = "data/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.tmp.dat.nc", var_name = "tmp", lazy = False):
        self.lazy = lazy
        ds = Dataset(path)
        self.var_name = var_name
        self._init_fields(ds)
        self.nc_dataset = ds

        self.nc_vars = ds.variables

        pass

    def _init_fields(self, nc_dataset):
        nc_vars = nc_dataset.variables
        lons = nc_vars["lon"][:]
        lats = nc_vars["lat"][:]

        lats2d, lons2d = np.meshgrid(lats, lons)

        self.lons2d, self.lats2d = lons2d, lats2d

        self.timesVar = nc_vars["time"]
        self.times_num = nc_vars["time"][:]
        self.times = num2date(self.times_num, self.timesVar.units, self.timesVar.calendar)
        if not self.lazy:
            self.var_data = np.transpose( nc_vars[self.var_name][:], axes=(0,2,1))

        x_in,y_in,z_in = lat_lon.lon_lat_to_cartesian(self.lons2d.flatten(), self.lats2d.flatten())
        self.kdtree = KDTree(zip(x_in, y_in, z_in))


    def get_mean(self, start_year, end_year, months = None):
        """
        returns the mean for the period [start_year, end_year], over the months
        :type months: list
        months = list of month numbers over which the averaging is done
        """

        if months is None:
            months = list(range(1,13))


        start_date = datetime(start_year,1,1)
        end_date = datetime(end_year + 1,1,1)

        start_date_num = date2num(start_date, self.timesVar.units)
        end_date_num = date2num(end_date, self.timesVar.units)

        sel_query = (self.times_num >= start_date_num) & (self.times_num < end_date_num)
        sel_dates = self.times_num[sel_query]
        sel_data = np.transpose( self.nc_vars[self.var_name][sel_query,:,:], axes=(0,2,1))


        sel_dates = num2date(sel_dates,self.timesVar.units)

        bool_vector = np.where(map( lambda x: (x.month in months), sel_dates))[0]
        return np.mean(sel_data[bool_vector, :, :], axis=0)


    def get_daily_climatology(self, start_year, end_year, stamp_year = 2001):
        """
        returns a numpy array of shape (365, nx, ny) with daily climatological means
        """
        day = timedelta(days = 1)
        the_date = datetime(stamp_year,1,1)
        stamp_days = [the_date + i * day for i in xrange(365)]
        result = []
        for the_date in stamp_days:
            bool_vector = np.array(map(lambda x: (x.day == the_date.day) and
                                                 (x.month == the_date.month) and
                                                 (x.year <= end_year) and (x.year >= start_year), self.times))
            result.append(np.mean( self.var_data[bool_vector,:,:], axis=0))
        return np.array(result)
        pass

    def get_thawing_index_from_climatology(self, daily_temps_clim, t0 = 0.0 ):

        nt, nx, ny = daily_temps_clim.shape
        result = np.zeros((nx, ny))

        for t in xrange(nt):
            tfield = daily_temps_clim[t,:,:]
            result += tfield * np.array(tfield >= t0).astype(int)
        return result


        pass



    def create_monthly_means_file(self, start_year, end_year):
        fname = "{0}_monthly_means.nc".format(self.var_name)
        year_range = range(start_year, end_year+1)
        dsm = Dataset(fname, "w", format="NETCDF3_CLASSIC")
        dsm.createDimension('year', len(year_range))
        dsm.createDimension("month", 12)
        dsm.createDimension('lon', self.lons2d.shape[0])
        dsm.createDimension('lat', self.lons2d.shape[1])

        lonVariable = dsm.createVariable('longitude', 'f4', ('lon', 'lat'))
        latVariable = dsm.createVariable('latitude', 'f4', ('lon', 'lat'))
        yearVariable = dsm.createVariable("year", "i4", ("year",))

        variable = dsm.createVariable(self.var_name, "f4", ('year', "month" ,'lon', 'lat'))
        for i, the_year in enumerate(year_range):
            print the_year
            for j, the_month in enumerate(xrange(1,13)):
                variable[i,j,:,:] = self.get_mean(the_year, the_year, months=[the_month])

        lonVariable[:] = self.lons2d
        latVariable[:] = self.lats2d
        yearVariable[:] = np.array(year_range)
        dsm.close()

        pass



    def _interp_and_sum(self, data1d, mults_1d, x, y, z, nneighbors = 1):
        data_interp = self.interpolate_data_to_cartesian(data1d, x, y,z, nneighbours=nneighbors)
        return np.sum(mults_1d * data_interp)

    def get_monthly_timeseries_using_mask(self, mask, lons2d_target, lats2d_target, multipliers_2d, start_date = None, end_date = None):
        """
        multipliers_2d used to multiply the values when aggregating into a single timeseries
        sum(mi * vi) - in space
        """


        bool_vect = np.array( map( lambda t: start_date <= t <= end_date, self.times) )

        new_times = list(itertools.ifilter(lambda t: start_date <= t <= end_date, self.times))
        new_vals = self.var_data[bool_vect, :, :]
        x_out, y_out, z_out = lat_lon.lon_lat_to_cartesian(lons2d_target.flatten(), lats2d_target.flatten())

        print len(new_times)
        flat_mask = mask.flatten()
        x_out = x_out[flat_mask == 1]
        y_out = y_out[flat_mask == 1]
        z_out = z_out[flat_mask == 1]
        mults = multipliers_2d.flatten()[flat_mask == 1]

        data_interp = map(lambda t: self._interp_and_sum(new_vals[t,:,:].flatten(), mults, x_out, y_out, z_out), range( len(new_times)))

        print "Interpolated data", data_interp

        print "Interpolated all"
        return TimeSeries(time=new_times, data=data_interp).get_ts_of_monthly_means()


    def get_daily_timeseries_using_mask(self, mask, lons2d_target, lats2d_target, multipliers_2d, start_date = None, end_date = None):
        """
        multipliers_2d used to multiply the values when aggregating into a single timeseries
        sum(mi * vi) - in space
        """


        bool_vect = np.array( map( lambda t: start_date <= t <= end_date, self.times) )

        new_times = list(itertools.ifilter(lambda t: start_date <= t <= end_date, self.times))
        new_vals = self.var_data[bool_vect, :, :]
        x_out, y_out, z_out = lat_lon.lon_lat_to_cartesian(lons2d_target.flatten(), lats2d_target.flatten())

        print len(new_times)

        flat_mask = mask.flatten()
        x_out = x_out[flat_mask == 1]
        y_out = y_out[flat_mask == 1]
        z_out = z_out[flat_mask == 1]
        mults = multipliers_2d.flatten()[flat_mask == 1]
        data_interp = map(lambda t: self._interp_and_sum(new_vals[t,:,:].flatten(), flat_mask, x_out, y_out, z_out), range( len(new_times)))

        print "Interpolated all"
        return TimeSeries(time=new_times, data=data_interp).get_ts_of_daily_means()


    def interpolate_data_to_cartesian(self, data_in_flat, x, y, z, nneighbours = 4):
        """
        len(data_in_flat) , len(x) == len(y) == len(z) == len(data_out_flat) - all 1D
        """
        print "start query"
        dst, ind = self.kdtree.query(zip(x, y, z), k=nneighbours)
        print "end query"

        inverse_square = 1.0 / dst ** 2
        if len(dst.shape) > 1:
            norm = np.sum(inverse_square, axis=1)
            norm = np.array( [norm] * dst.shape[1] ).transpose()
            coefs = inverse_square / norm

            data_out_flat = np.sum( coefs * data_in_flat[ind], axis= 1)
        elif len(dst.shape) == 1:
            data_out_flat = data_in_flat[ind]
        else:
            raise Exception("Could not find neighbor points")
        return data_out_flat




    def interpolate_data_to(self, data_in, lons2d, lats2d, nneighbours = 4):
        """
        Interpolates data_in to the grid defined by (lons2d, lats2d)
        assuming that the data_in field is on the initial CRU grid

        interpolate using 4 nearest neighbors and inverse of squared distance
        """

        x_out, y_out, z_out = lat_lon.lon_lat_to_cartesian(lons2d.flatten(), lats2d.flatten())
        dst, ind = self.kdtree.query(zip(x_out, y_out, z_out), k=nneighbours)

        data_in_flat = data_in.flatten()

        inverse_square = 1.0 / dst ** 2
        if len(dst.shape) > 1:
            norm = np.sum(inverse_square, axis=1)
            norm = np.array( [norm] * dst.shape[1] ).transpose()
            coefs = inverse_square / norm

            data_out_flat = np.sum( coefs * data_in_flat[ind], axis= 1)
        elif len(dst.shape) == 1:
            data_out_flat = data_in_flat[ind]
        else:
            raise Exception("Could not find neighbor points")
        return np.reshape(data_out_flat, lons2d.shape)


def main():
    from permafrost import draw_regions
    dm = CRUDataManager()

    b, lons2d, lats2d = draw_regions.get_basemap_and_coords()

    x, y = b(dm.lons2d, dm.lats2d)


    fig = plt.figure()

    gs = gridspec.GridSpec(1,2)
    ax = fig.add_subplot(gs[0,0])
    data = dm.get_mean(1981, 2009, months = [6,7,8])
    img = b.contourf(x, y, data.copy(), ax = ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = fig.colorbar(img,  cax = cax)
    b.drawcoastlines(ax = ax)
    ax.set_title("CRU (not interp.), \n JJA period: {0} - {1}".format(1981, 2009))


    ax = fig.add_subplot(gs[0,1])
    data_projected = dm.interpolate_data_to(data, lons2d, lats2d)
    x, y = b(lons2d, lats2d)
    img = b.contourf(x, y, data_projected, ax = ax, levels = img.levels)

    #add pretty colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = fig.colorbar(img,  cax = cax)

    b.drawcoastlines(ax = ax)
    ax.set_title("CRU ( interp.), \n JJA period: {0} - {1}".format(1981, 2009))

    plt.show()
    plt.savefig("t_cru_jja.png")


    pass

def create_monthly_means():
    #tmp
    #dm = CRUDataManager()
    #dm.create_monthly_means_file(1901, 2009)

    #pre
    dm = CRUDataManager(path="data/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.pre.dat.nc", var_name="pre")
    dm.create_monthly_means_file(1901, 2009)



def plot_thawing_index():
    dm = CRUDataManager()
    clim = dm.get_daily_climatology(1981, 2010)
    thi = dm.get_thawing_index_from_climatology(clim)

    plt.pcolormesh(thi.transpose())
    plt.colorbar()
    plt.show()



if __name__ == "__main__":
    application_properties.set_current_directory()
    plot_thawing_index()
    #create_monthly_means()
    #main()
    print "Hello world"
  
