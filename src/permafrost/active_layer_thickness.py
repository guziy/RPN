from datetime import datetime
from netCDF4 import Dataset
import os
import pickle
import re
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import application_properties
from rpn.rpn import RPN
import matplotlib.pyplot as plt
from util import plot_utils

__author__ = 'huziy'

import numpy as np
from matplotlib import cm, gridspec
from multiprocessing import Pool
import draw_regions

class CRCMDataManager:

    def __init__(self, data_folder = None, file_prefix = "pm"):
        """
        If data folder contains several folder it descends into each one and
        looks for the Samples directory, then it assumes that month directories
        are in those Samples directories
        """
        self.data_folder = data_folder
        self._samples = "Samples"
        self.file_prefix = file_prefix
        self._init_yearmonth_to_data_path()
        self.T0 = 273.15


        layer_widths = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                            1.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

        bottoms = []
        tops = []
        b = 0
        t = 0
        for wi in layer_widths:
            t = b
            b += wi
            tops.append(t)
            bottoms.append(b)

        #self.level_heights = np.array(tops)
        self.level_heights = np.array( (np.array(tops) + np.array(bottoms)) * 0.5 )
        self.yearmonth_to_mm_soiltemp = None
        pass


    def get_alt_using_monthly_mean_climatology(self, year_range):
        tc = self.get_soiltemp_climatology(year_range)
        tc_max = np.max(tc, axis=0)
        return self._get_alt(tc_max)

    def get_soiltemp_climatology(self, year_range = None):

        monthly_climatology = [[] for m in xrange(12)]
        for the_year in year_range:
            for the_month in xrange(1, 13):
                the_mean = self._get_monthly_mean_soiltemp(the_year, the_month)
                if the_mean is None:
                    continue
                monthly_climatology[the_month - 1].append(the_mean)

        for m in xrange(12):
            monthly_climatology[m] = np.mean(monthly_climatology[m], axis=0)

        return monthly_climatology




    def get_alt_using_files_in(self, folder = "", file_name_prefix = "pm"
                               ):
        all_temps = []
        for the_file in os.listdir(folder):
            if the_file.startswith(file_name_prefix):
                print the_file
                times, temps = self._read_profiles_from_file(os.path.join(folder, the_file), var_name="I0")

                mt = np.mean(temps, axis = 0)
                all_temps.append(mt)

        max_temp = np.max(all_temps, axis = 0)

        return self._get_alt(max_temp)




    def _get_monthly_mean_soiltemp(self, year, month, var_name = ""):
        """
        returns None if the data for the specifide year and month
        were not found
        """
        key = (year, month)

        if not self.yearmonth_to_data_path.has_key(key):
            print "Warning could not find data for month/year = {0}/{1}".format(month, year)
            return None
        data_path = self.yearmonth_to_data_path[key]

        times, temp = self._read_profiles_from_file(data_path, var_name=var_name)

        mmean = np.mean(temp, axis=0)
        return mmean



    def _read_profiles_from_file(self, file_path, var_name = ""):
        """
        returns [times(t), T(t,x, y, z)]
        """
        rpn_obj = RPN(file_path)

        data = rpn_obj.get_4d_field(name = var_name)
        times = sorted( data.keys() )
        nt = len(times)
        levels = sorted( data.items()[0][1].keys() )


        nz = len(levels)

        field = data.items()[0][1].items()[0][1]
        nx, ny = field.shape

        temperature = np.zeros((nt, nx, ny, nz))
        for ti, t in enumerate(times):
            for k, level in enumerate(levels):
                temperature[ti, :, :, k] = data[t][level]

        rpn_obj.close()
        return times, temperature
        pass

    def _get_daily_means(self, times, soilt_temps):
        start_day = times[0].day
        end_day = times[-1].day

        daily_means = []
        for the_day in xrange(start_day, end_day + 1):
            bool_vector = map( lambda x: x.day == the_day, times )
            bool_vector = np.array(bool_vector)
            daily_means.append(np.mean(soilt_temps[bool_vector,:,:,:], axis = 0))
        return daily_means

        pass

    def get_Tmax_profiles_for_year_using_daily_means(self, year, var_name = ""):
        """
        returns matrix T(x, y, z)
        Temeperature is taken as mean of the temperatures during a day
        """
        profiles = []
        for month in xrange(1,13):
            key = (year, month)

            if not self.yearmonth_to_data_path.has_key(key):
                print "Warning: could not find data for year/month = {0}/{1} ".format(year, month)
                continue

            path = self.yearmonth_to_data_path[key]



            times, temp = self._read_profiles_from_file(path, var_name=var_name)
            profiles.extend(self._get_daily_means(times, temp))
            pass

        return np.max(np.array(profiles), axis=0)


        pass

    def get_Tmax_profiles_for_year_using_monthly_means(self, year, var_name = ""):
        """
        returns matrix T(x, y, z)
        Temeperature is taken as mean of the temperatures during a day
        """
        profiles = []
        for month in xrange(1,13):
            key = (year, month)

            if not self.yearmonth_to_data_path.has_key(key):
                print "Warning: could not find data for year/month = {0}/{1} ".format(year, month)
                continue

            profiles.append(self._get_monthly_mean_soiltemp(year, month, var_name=var_name))
            pass

        return np.max(np.array(profiles), axis=0)



    def _get_alt(self, soiltemp_3d):
        nx, ny, nz = soiltemp_3d.shape
        alt = -np.ones((nx, ny))

        #if t_max allways < 0 then alt = 0
        all_neg = np.all(soiltemp_3d <= self.T0, axis=2)
        alt[all_neg] = 0.0


        #if tmax intersects 0
        for k in xrange(0, nz - 1):
            t2 = soiltemp_3d[:,:,k + 1]
            t1 = soiltemp_3d[:,:,k]
            intersection = (t2 - self.T0) * (t1 - self.T0) <= 0
            #intersection = (t2 >= self.T0) & (t1 <= self.T0)
            first_intersection = intersection & (alt < 0)

            h2 = self.level_heights[k + 1]
            h1 = self.level_heights[k]

            possible_div_by_0 = (t1 == t2)

            ind1 = first_intersection & (~possible_div_by_0)
            alt[ind1] = h2 + (h1 - h2) * (self.T0 - t2[ind1]) / (t1[ind1] - t2[ind1])
            alt[first_intersection & possible_div_by_0 & (t1 <= self.T0)] = h1


        return alt


    def get_active_layer_thickness(self, year, mean_temps_to_use = "monthly"):
        """
        return 2D field of the active layer thickness for the year
        returns ALT(x, y)
        """
        #t(x,y,z) - annual maximums
        if mean_temps_to_use == "daily" :
            t = self.get_Tmax_profiles_for_year_using_daily_means(year, var_name="I0")
        elif mean_temps_to_use == "monthly":
            t = self.get_Tmax_profiles_for_year_using_monthly_means(year, var_name="I0")
        else:
            raise Exception("Unknown averaging interval: {0}".format(mean_temps_to_use))
        return self._get_alt(t)


    def _init_year_month_to_data_path_imp(self, samples_dir):
        for month_folder in os.listdir(samples_dir):
            suffix = month_folder.split("_")[-1]
            if len(suffix) != len(re.findall("\d+", suffix)[0]):  #check that the suffix consists only of digits
                continue
            d = datetime.strptime(suffix, "%Y%m")
            key = (d.year, d.month)
            month_path = os.path.join(samples_dir, month_folder)


            data_path = None
            if not os.path.isdir(month_path):
                continue

            for data_file in os.listdir(month_path):
                if data_file.startswith(self.file_prefix):
                    data_path = os.path.join(month_path, data_file)

            self.yearmonth_to_data_path[key] = data_path


    def _init_yearmonth_to_data_path(self):
        """
        inits a map {(year, month) : path_to_data_file}
        """
        self.yearmonth_to_data_path = {}


        #if using a single experiment
        if self._samples in os.listdir(self.data_folder):
            samples_dir = os.path.join(self.data_folder, self._samples)
            self._init_year_month_to_data_path_imp(samples_dir)
            return



        #if need to merge several experiments
        for folder in os.listdir(self.data_folder):
            parent_i = os.path.join(self.data_folder, folder)
            if not self._samples in os.listdir(parent_i): continue # skip parent which do not contain Samples folder
            samples_dir = os.path.join(self.data_folder, folder, self._samples)
            self._init_year_month_to_data_path_imp(samples_dir)



        pass

    def get_mean_over_months_of_2d_var(self, start_year, end_year, months = None, var_name = ""):
        monthly_means = []
        for the_year in xrange(start_year, end_year + 1):
            for the_month in months:
                path = self.yearmonth_to_data_path[(the_year, the_month)]
                rpn_obj = RPN(path)

                records = rpn_obj.get_all_time_records_for_name(varname=var_name)

                monthly_means.append(np.mean(records.values(), axis=0))

                rpn_obj.close()

        return np.mean(monthly_means, axis=0)
        pass


    def get_seasonal_mean_for_year_of_2d_var(self, the_year, months = None, var_name = ""):
        """
        Return mean over months of a given 2d field
        returns numpy array of dimensions (x, y)
        """
        monthly_means = []
        for the_month in months:

            key = (the_year, the_month)
            if not self.yearmonth_to_data_path.has_key(key):
                print("Warning donot have data for {0}/{1}".format(the_year, the_month))
                continue

            path = self.yearmonth_to_data_path[key]
            rpn_obj = RPN(path)
            records = rpn_obj.get_all_time_records_for_name(varname=var_name)
            monthly_means.append(np.mean(records.values(), axis=0))
            rpn_obj.close()

        return np.mean(monthly_means, axis=0)



    @classmethod
    def get_mean_2d_from_climatologies(cls, path = "", file_prefixes = None,
                                    file_suffixes = None, var_name = ""):
        """
        When you have a folder with climatologies, use this method
        """

        field_list = []

        if file_prefixes is None:
            file_prefixes = os.listdir(path)

        if file_suffixes is None:
            file_suffixes = os.listdir(path)

        for file_name in os.listdir(path):
            prefix_ok = False
            suffix_ok = False

            for p in file_prefixes:
                if file_name.startswith(p):
                    prefix_ok = True
                    break

            for s in file_suffixes:
                if file_name.endswith(s):
                    suffix_ok = True
                    break

            if prefix_ok and suffix_ok:
                rpn_obj = RPN(os.path.join(path, file_name))
                data = rpn_obj.get_first_record_for_name(var_name)
                rpn_obj.close()
                field_list.append(data)
        return np.array(field_list).mean(axis = 0)


def get_alt_for_year(year):
    cache_file = "year_to_alt.bin"
    year_to_alt = {}
    if os.path.isfile(cache_file):
        year_to_alt = pickle.load(open(cache_file))

    if year_to_alt.has_key(year):
        return year_to_alt[year]
    else:
        dm = CRCMDataManager(data_folder="data/CORDEX")
        h = dm.get_active_layer_thickness(year)
        year_to_alt[year] = h
        pickle.dump(year_to_alt, open(cache_file, mode="w"))
        return h


def get_alt_for_year(args):
    """
    """
    year, data_manager, mean_temps_to_use = args

    return data_manager.get_active_layer_thickness(year, mean_temps_to_use=mean_temps_to_use)


def save_alts_to_netcdf_file(path = "alt.nc"):
    data_path = "data/cordex_e1"
    year_range = xrange(1981, 2101)
    ds = Dataset(path, mode = "w", format="NETCDF3_CLASSIC")
    b, lons2d, lats2d = draw_regions.get_basemap_and_coords()
    ds.createDimension('year', len(year_range))
    ds.createDimension('lon', lons2d.shape[0])
    ds.createDimension('lat', lons2d.shape[1])

    lonVariable = ds.createVariable('longitude', 'f4', ('lon', 'lat'))
    latVariable = ds.createVariable('latitude', 'f4', ('lon', 'lat'))
    yearVariable = ds.createVariable("year", "i4", ("year",))

    altVariable = ds.createVariable("alt", "f4", ('year','lon', 'lat'))

    lonVariable[:,:] = lons2d[:,:]
    latVariable[:,:] = lats2d[:,:]
    yearVariable[:] = year_range

    dm = CRCMDataManager(data_folder=data_path)
    dm_list = len(year_range) * [dm]
    mean_types = len(year_range) * ["monthly"]
    pool = Pool(processes=6)
    alts = pool.map(get_alt_for_year, zip(year_range, dm_list, mean_types))
    alts = np.array(alts)
    altVariable[:,:,:] = alts[:,:,:]
    ds.close()





    pass

def plot_means_and_stds_for_period(year_range = range(1981,2011),
                                   plot_grid = None,
                                   row = 0, mean_upper_limit = 10.0,
                                   figure = None, basemap = None,
                                   x = None, y = None, mask = None,
                                   permafrost_kind_field = None
    ):

    """
    Calculates fields of means and standard deviations of ALT
    figure - is the figure to which subplots are added,
    x, y - 2d coordinates in the basemap projection
    :type plot_grid: gridspec.GridSpec
    """
    alts = [ get_alt_for_year(the_year) for the_year in year_range ]

    mean_alt = np.mean(alts, axis = 0)
    std_alt = np.std(alts, axis = 0)

    mean_alt = np.ma.masked_where(mask, mean_alt)
    std_alt = np.ma.masked_where(mask, std_alt)


    alt_levels = range(0, int(mean_upper_limit + 1))
    cmap = cm.get_cmap("jet", len(alt_levels))

    ax0 = figure.add_subplot(plot_grid[row, 0])
    cs0 = basemap.contourf(x, y, mean_alt, ax = ax0, levels = alt_levels, cmap = cmap )
    ax0.set_title("mean over {0} - {1}".format(year_range[0], year_range[-1]))

    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = plt.colorbar(cs0,  cax = cax)
    basemap.contour(x,y, permafrost_kind_field,
        levels = xrange(1,4), colors = "k", linewidths = 0.5, ax = ax0)

    ax1 = figure.add_subplot(plot_grid[row, 1])
    std_alt = np.ma.masked_where(mean_alt > mean_upper_limit, std_alt)
    cs1 = basemap.contourf(x, y, std_alt, ax = ax1 )

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = plt.colorbar(cs1,  cax = cax)
    basemap.contour(x,y, permafrost_kind_field,
        levels = xrange(1,4), colors = "k", linewidths = 0.5, ax = ax1)
    ax1.set_title("Interannual variability")

    #draw coast lines
    basemap.drawcoastlines(ax = ax1, linewidth = 0.5)
    basemap.drawcoastlines(ax = ax0, linewidth = 0.5)




def plot_alt_from_monthly_climatologies():
    plot_utils.apply_plot_params(width_pt=None, height_cm=20, width_cm=16, font_size=12)
    figure = plt.figure()
    b, lons2d, lats2d = draw_regions.get_basemap_and_coords(llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-10)
    x, y = b(lons2d, lats2d)
    permafrost_mask = draw_regions.get_permafrost_mask(lons2d, lats2d)
    dm = CRCMDataManager(data_folder="data/CORDEX")

    year_ranges = [range(1981, 1985), range(2041, 2045), range(2071, 2075)]

    gs = gridspec.GridSpec(len(year_ranges),1)

    pf_mask = (permafrost_mask == 1) | (permafrost_mask == 2)
    pf_mask = ~pf_mask

    permafrost_mask = np.ma.masked_where(permafrost_mask <= 0, permafrost_mask)
    for i, year_range in enumerate(year_ranges):
        ax = figure.add_subplot(gs[i, 0])
        alt = dm.get_alt_using_monthly_mean_climatology(year_range)
        alt = np.ma.masked_where(pf_mask, alt)

        img = b.contourf(x, y, alt, levels = xrange(11), cmap = cm.get_cmap("jet", ), ax = ax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = plt.colorbar(img,  cax = cax)

        b.contour(x, y, permafrost_mask, levels = xrange(5), linewidth = 0.1, colors = "k", ax = ax)
        b.drawcoastlines(ax = ax, linewidth = 0.5)
        ax.set_title("period: {0} - {1}".format(year_range[0], year_range[-1]))
    plt.savefig("alt_from_clim.png")



def plot_alt_for_different_e_scenarios():

    labels = ("E1", "E2", "E3", "E4")
    p_format =  "pmNorthAmerica_0.44deg_CanHistoE{0}"
    prefixes = map(lambda x: p_format.format(x), xrange(1,5))
    b, lons2d, lats2d = draw_regions.get_basemap_and_coords(llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-10)
    x, y = b(lons2d, lats2d)
    permafrost_mask = draw_regions.get_permafrost_mask(lons2d, lats2d)

    dm = CRCMDataManager(data_folder="data/CORDEX") #needed here only for verticla levels


    plot_utils.apply_plot_params(width_pt=None, height_cm=12, width_cm=16, font_size=12)
    fig = plt.figure()

    gs = gridspec.GridSpec(2,2)

    scen_index = 0
    for row in xrange(2):
        for col in xrange(2):
            sc = labels[scen_index]
            ax = fig.add_subplot(gs[row, col])
            h = dm.get_alt_using_files_in(folder="data/CORDEX/na/means_month", file_name_prefix=prefixes[scen_index])
            h = np.ma.masked_where((permafrost_mask == 0) |
                                       (permafrost_mask >= 3) | (h < 0), h)

            plot_for_scenario(sc, ax, basemap=b, x = x, y= y, alt = h, permafrost_mask=permafrost_mask,
                start_year=1950, end_year=1954
            )

            scen_index += 1
    gs.tight_layout(fig, h_pad = 0.9, w_pad = 16)
    fig.savefig("alt_diff_scenarios.png")
    pass


def plot_for_scenario(scen_id, ax, basemap = None,
                      x = None, y = None, alt = None,
                      permafrost_mask = None, start_year = None, end_year = None
                      ):
    """

    """
    levels = np.arange(0, 11, 1)

    img = basemap.contourf( x,y, alt, ax = ax, levels = levels, cmap = cm.get_cmap("jet", len(levels) - 1))
    #img = b.pcolormesh(x,y, h, vmin = levels[0], vmax = levels[-1])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = plt.colorbar(img,  cax = cax)

    ax.set_title("{0}: {1}-{2}".format(scen_id, start_year, end_year))
    basemap.drawcoastlines(ax = ax)
    basemap.contour(x,y, permafrost_mask, ax = ax, levels = xrange(1,4), colors = "k", linewidths = 0.5)



def test():
    fig = plt.figure()
    b, lons2d, lats2d = draw_regions.get_basemap_and_coords(llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-10)
    x, y = b(lons2d, lats2d)
    permafrost_mask = draw_regions.get_permafrost_mask(lons2d, lats2d)

    dm = CRCMDataManager(data_folder="data/CORDEX") #needed here only for verticla levels

    h = dm.get_alt_using_files_in(folder="data/CORDEX/na/era40_2")
    h = np.ma.masked_where((permafrost_mask == 0) |
                           (permafrost_mask >= 3) | (h < 0), h)

    levels = np.arange(0, 11, 1)
    ax = plt.gca()
    img = b.contourf(x,y, h, levels = levels, cmap = cm.get_cmap("jet", len(levels) - 1), ax = ax)
    #img = b.pcolormesh(x,y, h, vmin = levels[0], vmax = levels[-1])


    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = fig.colorbar(img,  cax = cax)

    ax.set_title("ERA40-2 1958-1961")

    b.drawcoastlines(ax = ax)
    b.contour(x,y, permafrost_mask, levels = xrange(1,4), colors = "k", linewidths = 0.5, ax = ax)
    fig.savefig("alt_ERA40-2.png")


def main():
    #test()
    plot_utils.apply_plot_params(width_pt=None, height_cm=20, width_cm=16, font_size=12)
    figure = plt.figure()
    #year_ranges = [range(1981, 2011), range(2041, 2071), range(2071, 2101)]
    year_ranges = [range(1981, 1985), range(2041, 2045), range(2071, 2075)]
    b, lons2d, lats2d = draw_regions.get_basemap_and_coords()
    x, y = b(lons2d, lats2d)
    permafrost_mask = draw_regions.get_permafrost_mask(lons2d, lats2d)

    mask = (permafrost_mask == 0) | (permafrost_mask >= 3)

    gs = gridspec.GridSpec(len(year_ranges), 2)
    for i, the_range in enumerate(year_ranges):
        plot_means_and_stds_for_period(year_range=the_range, figure=figure,
            x = x, y = y, basemap=b, mask = mask, row=i, plot_grid=gs,
            permafrost_kind_field=permafrost_mask
        )

    figure.savefig("permafrost_mean_and_std.png")
    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    #main()
    test()
    #save_alts_to_netcdf_file()
    #plot_alt_from_monthly_climatologies()
    #plot_alt_for_different_e_scenarios()
    print "Hello world"
  