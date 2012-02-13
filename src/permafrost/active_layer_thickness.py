from datetime import datetime
import os
import itertools
import pickle
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import application_properties
from rpn import RPN
import matplotlib.pyplot as plt
from util import plot_utils

__author__ = 'huziy'

import numpy as np
from matplotlib import cm, gridspec
from matplotlib import ticker

import draw_regions

class DataManager:

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

        self.level_heights = np.array( (np.array(tops) + np.array(bottoms)) * 0.5 )

        pass




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

    def get_Tmax_profiles_for_year(self, year, var_name = ""):
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

    def get_active_layer_thickness(self, year):
        """
        return 2D field of the active layer thickness for the year
        returns ALT(x, y)
        """
        #t(x,y,z) - annual maximums
        t = self.get_Tmax_profiles_for_year(year, var_name="I0")

        nx, ny, nz = t.shape
        alt = -np.ones((nx, ny))

        #if t_max allways < 0 then alt = 0
        all_neg = np.all(t <= self.T0, axis=2)
        alt[all_neg] = 0.0


        #if tmax intersects 0
        for k in xrange(0, nz - 1):
            t2 = t[:,:,k + 1]
            t1 = t[:,:,k]
            intersection = (t2 - self.T0) * (t1 - self.T0) <= 0
            first_intersection = intersection & (alt == -1)

            h2 = self.level_heights[k + 1]
            h1 = self.level_heights[k]

            possible_div_by_0 = (t1 == t2)

            ind1 = first_intersection & (~possible_div_by_0)
            alt[ind1] = h2 + (h1 - h2) * (self.T0 - t2[ind1]) / (t1[ind1] - t2[ind1])
            alt[first_intersection & possible_div_by_0 & (t1 <= self.T0)] = h1

        return alt


        pass







    def _init_yearmonth_to_data_path(self):
        """
        inits a map {(year, month) : path_to_data_file}
        """
        self.yearmonth_to_data_path = {}

        for folder in os.listdir(self.data_folder):
            samples_dir = os.path.join(self.data_folder, folder, self._samples)

            for month_folder in os.listdir(samples_dir):
                d = datetime.strptime(month_folder.split("_")[-1], "%Y%m")
                key = (d.year, d.month)
                month_path = os.path.join(samples_dir, month_folder)


                data_path = None
                if not os.path.isdir(month_path):
                    continue

                for data_file in os.listdir(month_path):
                    if data_file.startswith(self.file_prefix):
                        data_path = os.path.join(month_path, data_file)

                self.yearmonth_to_data_path[key] = data_path


        pass



def get_alt_for_year(year):
    cache_file = "year_to_alt.bin"
    year_to_alt = {}
    if os.path.isfile(cache_file):
        year_to_alt = pickle.load(open(cache_file))

    if year_to_alt.has_key(year):
        return year_to_alt[year]
    else:
        dm = DataManager(data_folder="data/CORDEX")
        h = dm.get_active_layer_thickness(year)
        year_to_alt[year] = h
        pickle.dump(year_to_alt, open(cache_file, mode="w"))
        return h


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








def test():



    plt.figure()

    b, lons2d, lats2d = draw_regions.get_basemap_and_coords()
    x, y = b(lons2d, lats2d)
    permafrost_mask = draw_regions.get_permafrost_mask(lons2d, lats2d)

    h = get_alt_for_year(1981)
    h = np.ma.masked_where((permafrost_mask == 0) |
                           (permafrost_mask >= 3)

        , h)
    dm = DataManager(data_folder="data/CORDEX") #needed here only for verticla levels

    levels = range(11)
    img = b.contourf(x,y, h, levels = levels, cmap = cm.get_cmap("jet", len(levels) - 1))

    plt.colorbar(img)

    b.drawcoastlines()
    b.contour(x,y, permafrost_mask, levels = xrange(1,4), colors = "k", linewidths = 0.5)
    plt.savefig("alt.png")


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
    main()
    #test()
    print "Hello world"
  