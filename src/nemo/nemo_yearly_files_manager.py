from collections import defaultdict
from netCDF4 import Dataset, OrderedDict, num2date, date2num, MFDataset
import os
import pandas as pd
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.basemap import Basemap

# from rpn.domains.rotated_lat_lon import RotatedLatLon

import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.ckdtree import cKDTree
from domains.rotated_lat_lon import RotatedLatLon
from util import plot_utils
from util.geo import lat_lon
import numpy as np
import matplotlib.dates as mdates
from datetime import timedelta

__author__ = 'huziy'

try:
    import iris
    from iris import unit as iunit
    from iris.cube import Cube
    from iris.time import PartialDateTime
    import iris.quickplot as qplt
    from iris import analysis as ianalysis
except ImportError as ierr:
    print(ierr)
    print("Iris is not installed.")


class NemoYearlyFilesManager(object):
    def __init__(self, folder="", bathymetry_file="bathy_meter.nc",
                 proj_file="gemclim_settings.nml", suffix="_T.nc"):
        self.bathymetry = None
        self.data_folder = folder
        self.bathymetry_file = bathymetry_file
        self.proj_file = proj_file
        self.suffix = suffix

        self.model_kdtree = None
        self.lons = None
        self.lats = None
        self.basemap = None
        self.lake_mask = None
        self.get_coords_and_basemap()
        self.define_lake_mask()

        self.year_to_path = None
        self._build_year_to_datapath_map()

        self.ccrs = None


    def get_tz_crosssection_for_the_point(self, lon=None, lat=None, zlist=None, var_name="",
                                          start_date=None, end_date=None):

        """
        get t-z cross section matrix for the point on the zlist levels
        Note: if zlist is None, the profiles are returned on model levels
        :param lon:
        :param lat:
        :param zlist:
        :param var_name:
        :param start_date:
        :param end_date:
        """
        if self.model_kdtree is None:
            xs, ys, zs = lat_lon.lon_lat_to_cartesian(self.lons.flatten(), self.lats.flatten())
            self.model_kdtree = cKDTree(list(zip(xs, ys, zs)))

        xt, yt, zt = lat_lon.lon_lat_to_cartesian(lon, lat)

        start_year = start_date.year
        end_year = end_date.year

        # Get 4 nearest neighbors for interpolation
        dists_from, inds_from = self.model_kdtree.query([(xt, yt, zt), ], k=1)


        # Calculate the inverse of squre of distance for weighted average
        weights = 1.0 / dists_from ** 2
        weights /= weights.sum()

        inds_from = inds_from.squeeze()
        weights = weights.squeeze()
        if len(weights.shape) == 0:
            weights = [weights, ]

        neighbor_lons = self.lons.flatten()[inds_from]
        neighbor_lats = self.lats.flatten()[inds_from]

        i_list, j_list = [], []

        if dists_from.ndim > 1:
            for the_lon, the_lat in zip(neighbor_lons, neighbor_lats):
                i, j = np.where((self.lons == the_lon) & (self.lats == the_lat))
                i_list.append(i[0])
                j_list.append(j[0])
        else:
            i, j = np.where((self.lons == neighbor_lons) & (self.lats == neighbor_lats))
            i_list.append(i[0])
            j_list.append(j[0])

        profiles = []
        dates = []
        ztarget = np.asarray(zlist) if zlist is not None else None
        vert_kdtree = None
        for the_year in range(start_year, end_year + 1):


            # cube dimensions (t, z, y, x)

            with MFDataset(self.year_to_path[the_year]) as ds:
                data = ds.variables[var_name]

                time_var = ds.variables["time_counter"]


                time_data = time_var[:]

                assert np.all(time_data == np.array(sorted(time_data))), "Time data is not sorted: {}".format(time_data)

                if end_date.hour == 0:
                    end_date += timedelta(days=1)

                d1 = date2num(start_date, time_var.units)
                d2 = date2num(end_date, time_var.units)

                current_dates = num2date([t for t in time_var[:] if d1 <= t <= d2], units=time_var.units)
                data = data[np.where((d1 <= time_var[:]) & (time_var[:] <= d2))[0], :, :, :]


                # Use inverse squared distances to interpolate in horizontal
                prof = data[:, :, j_list[0], i_list[0]] * weights[0]
                for i, j, weight in zip(i_list[1:], j_list[1:], weights[1:]):
                    prof += data[:, :, j, i] * weight

                # Linear interpolation in vertical

                if "deptht" in ds.variables:
                    zsource = ds.variables["deptht"][:]
                elif "depthu" in ds.variables:
                    zsource = ds.variables["depthu"][:]
                elif "depthv" in ds.variables:
                    zsource = ds.variables["depthv"][:]
                elif "depthw" in ds.variables:
                    zsource = ds.variables["depthw"][:]
                else:
                    raise Exception("Could not find vertical coordinate")

                if vert_kdtree is None:
                    vert_kdtree = cKDTree([[z, ] for z in zsource])

                # No interpolation if the vertical levels are not supplied
                ztarget = zsource if ztarget is None else ztarget

                zdists, zinds = vert_kdtree.query([[z, ] for z in ztarget], k=2)
                zdists = zdists.squeeze()
                zinds = zinds.squeeze()

                zweights = zdists / zdists.sum(axis=1)[:, np.newaxis]  # weight1 = d2/(d1 + d2)

                prof = prof[:, zinds[:, 0]] * zweights[np.newaxis, :, 1] + prof[:, zinds[:, 1]] * zweights[np.newaxis, :, 0]
                profiles.extend(prof)



                print("Selected data for the time range: ", current_dates[0], current_dates[-1])
                print("The limits are ", start_date, end_date)

                dates.extend(current_dates)

        # Calculate model bottom
        bottom = 0
        for i, j in zip(i_list, j_list):
            bottom += self.bathymetry[i, j]
        bottom /= float(len(i_list))

        dates_num = mdates.date2num(dates)

        # mask everything below the model bottom
        if zlist is None and False:
            profiles = np.asarray(profiles)
            profiles = profiles[:, np.where(ztarget <= bottom)]
            profiles = profiles.squeeze()
            ztarget = ztarget[ztarget <= bottom]

        zz, tt = np.meshgrid(ztarget, dates_num)

        # print("nemo tt-ranges: ", tt.min(), tt.max())
        # profiles = np.ma.masked_where(zz > bottom, profiles)


        # plot for debug
        #
        # plt.figure()
        # ax = plt.gca()
        # profiles = np.ma.masked_where(zz >= bottom, profiles)
        # im = ax.contourf(tt, zz, profiles, levels=np.arange(4, 30, 1))
        #
        # xlimits = ax.get_xlim()
        # ax.plot(xlimits, [bottom, bottom], "k-", lw=2)
        # print bottom
        #
        # assert isinstance(ax, Axes)
        # ax.invert_yaxis()
        # ax.xaxis.set_major_formatter(DateFormatter("%Y\n%b\n%d"))
        #
        # plt.colorbar(im)
        # plt.show()

        return tt, zz, profiles


    def get_seasonal_clim_field(self, start_year=None, end_year=None, season_to_months=None,
                                varname="sosstsst", level_index=0):

        """
        Get seasonal mean climatology for a field
        :param start_year:
        :param end_year:
        :param season_to_months:
        :param varname:
        """
        if start_year is None:
            start_year = min(self.year_to_path.keys())

        if end_year is None:
            end_year = max(self.year_to_path.keys())

        # Set up month to season relation
        month_to_season = defaultdict(lambda: "no-season")
        for m in range(1, 13):
            for s, months in season_to_months.items():
                if m in months:
                    month_to_season[m] = s
                    break

        season_to_field_list = defaultdict(list)
        for y in range(start_year, end_year + 1):
            fpath = self.year_to_path[y]

            with MFDataset(fpath) as ds:

                data_var = ds.variables[varname]

                if len(data_var.shape) == 3:
                    nt, ny, nx = data_var.shape
                    data = data_var[:]
                elif len(data_var.shape) == 4:
                    nt, nz, ny, nx = data_var.shape
                    data = data_var[:, level_index, :, :]
                else:
                    raise Exception("Do not know how to handle {}-dimensional fields".format(len(data_var.shape)))

                time_var = ds.variables["time_counter"]

                dates = num2date(time_var[:], time_var.units)

                panel = pd.Panel(data=data, items=dates, major_axis=range(ny), minor_axis=range(nx))

                seas_mean = panel.groupby(lambda d: month_to_season[d.month], axis="items").mean()

                for the_season in seas_mean:
                    season_to_field_list[the_season].append(seas_mean[the_season].values)

        result = {}
        for the_season, field_list in season_to_field_list.items():
            mean_field = np.mean(field_list, axis=0).transpose()
            print(mean_field.shape)

            result[the_season] = np.ma.masked_where(~self.lake_mask, mean_field)

        return result

    def get_max_yearly_ice_fraction(self, start_year, end_year):
        """
        Note the time interval [start_year, end_year] is inclusive
        :param start_year:
        :param end_year:
        """
        varname = "iiceconc"
        data = []
        lake_avg = []
        for the_year in range(start_year, end_year + 1):
            fpath = self.year_to_path[the_year]
            with Dataset(fpath) as ds:
                field = ds.variables[varname][:].max(axis=0)
                lake_avg.append(field.transpose()[self.lake_mask].mean())
                data.append(field)

        return np.mean(data, axis=0).transpose(), lake_avg

    def plot_comparisons_with_glerl_ice_cover(self, path_to_obs_file=""):
        # TODO:  ....
        pass


    def define_lake_mask(self):
        c = Dataset(os.path.join(self.data_folder, self.bathymetry_file)).variables["Bathymetry"][:]
        self.bathymetry = c.transpose()
        self.lake_mask = self.bathymetry > 0.5


    def _build_year_to_datapath_map(self):
        """

        build the relation {year => data path}
        """

        self.year_to_path = defaultdict(list)
        data_dir = Path(self.data_folder)

        for f in data_dir.iterdir():

            if not f.name.endswith(self.suffix):
                continue

            y = int(f.name.split("_")[2][:-4])
            self.year_to_path[y].append(str(f))






    def get_seasonal_mean_sst(self, start_year=None, end_year=None, season_to_months=None):

        """

        :param start_year:
        :param end_year:
        :param season_to_months:
        :return: dict(year -> season -> field)
        """

        raise NotImplementedError()


    def get_seasonal_mean_lst(self, start_year=None, end_year=None, season_to_months=None):

        """

        :param start_year:
        :param end_year:
        :param season_to_months:
        :return: dict(year -> season -> field)
        """

        import pandas as pd


        month_to_season = defaultdict(lambda: "no-season")

        for season, months in season_to_months.items():
            for the_month in months:
                month_to_season[the_month] = season


        print("months_to_season = {}".format(month_to_season))

        result = {}
        for the_year in range(start_year, end_year + 1):
            result[the_year] = {}
            data_path = self.year_to_path[the_year]
            with MFDataset(data_path) as ds:
                # sst = ds.variables["isstempe"][:]
                # ist = ds.variables["isnotem2"][:]
                # ice_f = ds.variables["iiceconc"][:]

                # if hasattr(ice_f, "mask"):
                #     ice_f[ice_f.mask] = 0

                # Calculate lake surface temperature
                # lst = sst * (1.0 - ice_f) + ist * ice_f

                lst = ds.variables["sosstsst"][:]

                time_var = ds.variables["time_counter"]
                dates = num2date(time_var[:], time_var.units).tolist()
                print(", ".join([str(d) for d in dates[:10]]) + ", ..., {}".format(dates[-1]))

                panel = pd.Panel(data=lst, items=dates, major_axis=range(lst.shape[1]), minor_axis=range(lst.shape[2]))


                seasonal_panel = panel.groupby(
                    lambda d: month_to_season[d.month], axis="items").mean()

                for the_season in season_to_months:
                    print(seasonal_panel)
                    # in the files the dimensions are ordered as (t, y, x) -> hence the transpose below
                    result[the_year][the_season] = seasonal_panel[the_season, :, :].values.T


        return result


    def read_and_interpolate_homa_data(self, path="", start_year=None, end_year=None, season_to_months=None):
        """
        :param path:
        :param target_cube:
        """
        import pandas as pd

        ds = Dataset(path)
        sst = ds.variables["sst"][:]

        # read longitudes and latitudes from a file
        lons_source = ds.variables["lon"][:]
        lats_source = ds.variables["lat"][:]



        month_to_season = defaultdict(lambda: "no-season")

        for seas, mths in season_to_months.items():
            for m in mths:
                month_to_season[m] = seas


        # time variable
        time_var = ds.variables["time"]
        dates = num2date(time_var[:], time_var.units)

        if hasattr(sst, "mask"):
            sst[sst.mask] = np.nan

        panel = pd.Panel(data=sst, items=dates, major_axis=range(sst.shape[1]), minor_axis=range(sst.shape[2]))



        seasonal_sst = panel.groupby(
            lambda d: (d.year, month_to_season[d.month]), axis="items").mean()


        # source grid
        xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons_source.flatten(), lats_source.flatten())
        kdtree = cKDTree(data=list(zip(xs, ys, zs)))

        # target grid
        xt, yt, zt = lat_lon.lon_lat_to_cartesian(self.lons.flatten(), self.lats.flatten())

        dists, inds = kdtree.query(list(zip(xt, yt, zt)))



        assert isinstance(seasonal_sst, pd.Panel)

        result = {}
        for the_year in range(start_year, end_year + 1):
            result[the_year] = {}
            for the_season in list(season_to_months.keys()):
                the_mean = seasonal_sst.select(lambda item: item == (the_year, the_season), axis="items")
                result[the_year][the_season] = the_mean.values.flatten()[inds].reshape(self.lons.shape)

        return result


    def get_nemo_and_homa_seasonal_mean_sst(self, start_year=None, end_year=None, season_to_months=None, use_noaa_oisst=False):
        """

        :param start_year:
        :param end_year:
        :param season_to_months:
        :return: {year: {season: field}}
        """
        model_data = self.get_seasonal_mean_lst(season_to_months=season_to_months,
                                                start_year=start_year, end_year=end_year)


        obs_data = None
        if not use_noaa_oisst:
            obs_sst_path = os.path.expanduser("~/skynet3_rech1/nemo_obs_for_validation/GreatLakes_2003_5km-2/sst-glk.nc")

            obs_data = self.read_and_interpolate_homa_data(path=obs_sst_path, start_year=start_year, end_year=end_year,
                                                           season_to_months=season_to_months)


        return model_data, obs_data, self.basemap, self.lons, self.lats

    def plot_comparisons_of_seasonal_sst_with_homa_obs(self, start_year=None, end_year=None, season_to_months=None,
                                                       exp_label=""):

        model_data = self.get_seasonal_mean_lst(season_to_months=season_to_months,
                                                start_year=start_year, end_year=end_year)

        obs_sst_path = os.path.expanduser("~/skynet3_rech1/nemo_obs_for_validation/GreatLakes_2003_5km-2/sst-glk.nc")

        obs_data = self.read_and_interpolate_homa_data(path=obs_sst_path, start_year=start_year, end_year=end_year,
                                                       season_to_months=season_to_months)

        plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=20, height_cm=10)
        # calculate climatologic differences
        diff = {}
        for season in list(season_to_months.keys()):
            diff[season] = np.mean(
                [model_data[y][season] - obs_data[y][season] for y in range(start_year, end_year + 1)], axis=0)
            diff[season] = np.ma.masked_where(~self.lake_mask, diff[season])
            the_field = diff[season]
            print("diff stats({}): min={}; max={}; avg={}".format(
                season, the_field.min(), the_field.max(), the_field.mean()))


        # plot seasonal biases
        xx, yy = self.basemap(self.lons.copy(), self.lats.copy())


        # calculate difference ranges
        diff_max = 0
        for season, the_diff in diff.items():
            diff_max = max(np.percentile(np.abs(the_diff[~the_diff.mask]), 90), diff_max)
        diff_max = 5

        locator = MaxNLocator(nbins=12, symmetric=True)
        bounds = locator.tick_values(-diff_max, diff_max)
        bn = BoundaryNorm(bounds, len(bounds) - 1)
        cmap = cm.get_cmap("RdBu_r", len(bounds) - 1)

        im = None
        fig = plt.figure()
        ncols = 2
        # fig.suptitle(r"LST $\left({\rm ^\circ C}\right)$", font_properties=FontProperties(weight="bold"))
        gs = GridSpec(len(season_to_months) // ncols, ncols + 1, width_ratios=[1.0, ] * ncols + [0.05, ])
        for i, season in enumerate(season_to_months.keys()):
            ax = fig.add_subplot(gs[i // ncols, i % ncols])
            im = self.basemap.pcolormesh(xx, yy, diff[season][:], ax=ax, cmap=cmap, norm=bn)
            ax.set_title(season)
            self.basemap.drawcoastlines(ax=ax, linewidth=0.5)
            if not i:
                ax.set_ylabel("NEMO - Obs")

        cb = plt.colorbar(im, ticks=locator, cax=fig.add_subplot(gs[:, -1]), extend="both")

        nemo_img_dir = "nemo"
        if not os.path.isdir(nemo_img_dir):
            os.mkdir(nemo_img_dir)

        plt.tight_layout()
        fig.savefig(os.path.join(nemo_img_dir, "sst_homa_validation_{}.pdf".format(exp_label)))
        plt.show()

    def get_cartopy_proj_and_coords(self):
        """
        :return: lons2d, lats2d, basemap [based on the bathymetry file and gemclim_settings.nml]
        """
        from cartopy import crs
        # Read longitudes and latitudes and create the basemap only if they are not initialized
        if self.ccrs is None:

            with Dataset(os.path.join(self.data_folder, self.bathymetry_file)) as ds:
                self.lons, self.lats = ds.variables["nav_lon"][:].transpose(), ds.variables["nav_lat"][:].transpose()

            import re

            lon1, lat1 = None, None
            lon2, lat2 = None, None
            with open(os.path.join(self.data_folder, self.proj_file)) as f:
                for line in f:
                    if "Grd_xlat1" in line and "Grd_xlon1" in line:
                        groups = re.findall(r"-?\b\d+.?\d*\b", line)
                        lat1, lon1 = [float(s) for s in groups]

                    if "Grd_xlat2" in line and "Grd_xlon2" in line:
                        groups = re.findall(r"-?\b\d+.?\d*\b", line)
                        lat2, lon2 = [float(s) for s in groups]

            rll = RotatedLatLon(lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2)
            # self.basemap = rll.get_basemap_object_for_lons_lats(lons2d=self.lons, lats2d=self.lats)


            lon0, _ = rll.get_true_pole_coords_in_rotated_system()
            o_lon_p, o_lat_p = rll.get_north_pole_coords()
            print(lon0, o_lat_p)
            self.ccrs = crs.RotatedPole(pole_longitude=lon0, pole_latitude=o_lat_p)

        self.lons[self.lons > 180] -= 360

        return self.lons, self.lats, self.ccrs


    def get_coords_and_basemap(self, resolution="c"):
        """
        :return: lons2d, lats2d, basemap [based on the bathymetry file and gemclim_settings.nml]
        """

        # Read longitudes and latitudes and create the basemap only if they are not initialized
        if self.lons is None:
            with Dataset(os.path.join(self.data_folder, self.bathymetry_file)) as ds:

                if "nav_lon" in ds.variables:
                    self.lons, self.lats = ds.variables["nav_lon"][:].transpose(), ds.variables["nav_lat"][:].transpose()
                else:
                    for vname, v in ds.variables.items():
                        if "lon" in vname.lower():
                            self.lons = v[:].T
                            continue

                        if "lat" in vname.lower():
                            self.lats = v[:].T
                            continue

                        if self.lons is not None and self.lats is not None:
                            break


                import re

                lon1, lat1 = None, None
                lon2, lat2 = None, None
                with open(os.path.join(self.data_folder, self.proj_file)) as f:
                    for line in f:
                        if "Grd_xlat1" in line and "Grd_xlon1" in line:
                            groups = re.findall(r"-?\b\d+.?\d*\b", line)
                            lat1, lon1 = [float(s) for s in groups]

                        if "Grd_xlat2" in line and "Grd_xlon2" in line:
                            groups = re.findall(r"-?\b\d+.?\d*\b", line)
                            lat2, lon2 = [float(s) for s in groups]

                rll = RotatedLatLon(lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2)
                self.basemap = rll.get_basemap_object_for_lons_lats(lons2d=self.lons, lats2d=self.lats, resolution=resolution)
                print(lon1, lat1, lon2, lat2)


        # self.basemap.drawcoastlines()
        # xx, yy = self.basemap(self.lons, self.lats)
        # self.basemap.pcolormesh(xx, yy, ds.variables["Bathymetry"][:].transpose())
        # plt.show()

        self.lons[self.lons > 180] -= 360

        return self.lons, self.lats, self.basemap


def main():
    # nemo_manager = NemoYearlyFilesManager(folder="/home/huziy/skynet3_rech1/offline_glk_output_daily_1979-2012",
    # suffix="icemod.nc")
    # exp_label = "default-offline"

    nemo_manager = NemoYearlyFilesManager(folder="/home/huziy/skynet3_rech1/output_NEMO_offline_1979-2012_nosnow",
                                          suffix="icemod.nc")
    exp_label = "offline-nosnow"

    # Study period
    start_year = 2003
    end_year = 2006

    season_to_months = OrderedDict([
        ("Winter", (12, 1, 2)),
        ("Spring", list(range(3, 6))),
        ("Summer", list(range(6, 9))),
        ("Fall", list(range(9, 12)))
    ])

    nemo_manager.plot_comparisons_of_seasonal_sst_with_homa_obs(
        start_year=start_year, end_year=end_year, season_to_months=season_to_months,
        exp_label=exp_label
    )


def validate_max_ice_cover_with_glerl():
    """
    For validations of maximum annual ice concentrations with GLERL obs

    """
    nemo_manager = NemoYearlyFilesManager(folder="/home/huziy/skynet3_rech1/offline_glk_output_daily_1979-2012",
                                          suffix="icemod.nc")

    nemo_manager_nosnow = NemoYearlyFilesManager(
        folder="/home/huziy/skynet3_rech1/output_NEMO_offline_1979-2012_nosnow",
        suffix="icemod.nc")

    # Study period
    start_year = 2003
    end_year = 2012

    lon2d, lat2d, bmp = nemo_manager.get_coords_and_basemap()
    model_yearmax_ice_conc, model_lake_avg_ts = nemo_manager.get_max_yearly_ice_fraction(
        start_year=start_year, end_year=end_year)
    model_yearmax_ice_conc = np.ma.masked_where(~nemo_manager.lake_mask, model_yearmax_ice_conc)

    model_yearmax_ice_conc_nosnow, model_lake_avg_ts_no_snow = nemo_manager_nosnow.get_max_yearly_ice_fraction(
        start_year=start_year, end_year=end_year)
    model_yearmax_ice_conc_nosnow = np.ma.masked_where(~nemo_manager.lake_mask, model_yearmax_ice_conc_nosnow)





    # plt.figure()
    xx, yy = bmp(lon2d.copy(), lat2d.copy())
    # im = bmp.pcolormesh(xx, yy, model_yearmax_ice_conc)
    # bmp.colorbar(im)

    # Read and interpolate obs
    path_to_obs = "/RESCUE/skynet3_rech1/huziy/nemo_obs_for_validation/glerl_icecov.nc"

    obs_varname = "ice_cover"
    obs_lake_avg_ts = []
    with Dataset(path_to_obs) as ds:
        time_var = ds.variables["time"]

        lons_obs = ds.variables["lon"][:]
        lats_obs = ds.variables["lat"][:]

        dates = num2date(time_var[:], time_var.units)
        nx, ny = lons_obs.shape

        data = ds.variables[obs_varname][:]
        data = np.ma.masked_where((data > 100) | (data < 0), data)
        print(data.min(), data.max())
        panel = pd.Panel(data=data, items=dates, major_axis=range(nx), minor_axis=range(ny))

        panel = panel.select(lambda d: start_year <= d.year <= end_year)
        the_max_list = []
        for key, g in panel.groupby(lambda d: d.year, axis="items"):
            the_max_field = np.ma.max(np.ma.masked_where((g.values > 100) | (g.values < 0), g.values), axis=0)
            obs_lake_avg_ts.append(the_max_field.mean())
            the_max_list.append(the_max_field)

        obs_yearmax_ice_conc = np.ma.mean(the_max_list, axis=0) / 100.0

        xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons_obs.flatten(), lats_obs.flatten())
        ktree = cKDTree(list(zip(xs, ys, zs)))

        xt, yt, zt = lat_lon.lon_lat_to_cartesian(lon2d.flatten(), lat2d.flatten())
        dists, inds = ktree.query(list(zip(xt, yt, zt)))

        obs_yearmax_ice_conc_interp = obs_yearmax_ice_conc.flatten()[inds].reshape(lon2d.shape)
        obs_yearmax_ice_conc_interp = np.ma.masked_where(~nemo_manager.lake_mask, obs_yearmax_ice_conc_interp)


    # plt.figure()
    # b = Basemap()
    # xx, yy = b(lons_obs, lats_obs)
    # im = b.pcolormesh(xx, yy, obs_yearmax_ice_conc)
    # b.colorbar(im)
    # b.drawcoastlines()

    # Plot as usual: model, obs, model - obs
    img_folder = Path("nemo")
    if not img_folder.is_dir():
        img_folder.mkdir()
    img_file = img_folder.joinpath("validate_yearmax_icecov_glerl_{}-{}.png".format(start_year, end_year))

    plot_utils.apply_plot_params(height_cm=9, width_cm=45, font_size=12)

    fig = plt.figure()
    gs = GridSpec(1, 5, width_ratios=[1, 1, 0.05, 1, 0.05])
    all_axes = []

    cmap = cm.get_cmap("jet", 10)
    diff_cmap = cm.get_cmap("RdBu_r", 10)

    # Model
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("NEMO-offline")
    bmp.pcolormesh(xx, yy, model_yearmax_ice_conc, cmap=cmap, vmin=0, vmax=1)
    all_axes.append(ax)

    # Model
    # ax = fig.add_subplot(gs[0, 1])
    # ax.set_title("NEMO-offline-nosnow")
    # bmp.pcolormesh(xx, yy, model_yearmax_ice_conc_nosnow, cmap=cmap, vmin=0, vmax=1)
    # all_axes.append(ax)


    # Obs
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("NIC")
    im = bmp.pcolormesh(xx, yy, obs_yearmax_ice_conc_interp, cmap=cmap, vmin=0, vmax=1)
    all_axes.append(ax)

    plt.colorbar(im, cax=fig.add_subplot(gs[0, 2]))



    # Biases
    ax = fig.add_subplot(gs[0, 3])
    ax.set_title("NEMO - NIC")
    im = bmp.pcolormesh(xx, yy, model_yearmax_ice_conc - obs_yearmax_ice_conc_interp, cmap=diff_cmap, vmin=-1, vmax=1)
    plt.colorbar(im, cax=fig.add_subplot(gs[0, -1]))
    all_axes.append(ax)

    for the_ax in all_axes:
        bmp.drawcoastlines(ax=the_ax)
        the_ax.set_aspect("auto")

    fig.savefig(str(img_file), bbox_inches="tight")
    plt.close(fig)


    # Plot lake aversged ice concentrations
    fig = plt.figure()
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.plot(range(start_year, end_year + 1), model_lake_avg_ts, "b", lw=2, label="NEMO")
    plt.plot(range(start_year, end_year + 1), model_lake_avg_ts_no_snow, "g", lw=2, label="NEMO-nosnow")
    plt.plot(range(start_year, end_year + 1), np.asarray(obs_lake_avg_ts) / 100.0, "r", lw=2, label="NIC")
    plt.grid()
    plt.legend(ncol=2)
    fig.savefig(str(img_folder.joinpath("lake_avg_iceconc_nemo_offline_vs_NIC.pdf")), bbox_inches="tight")


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()
    # main()

    validate_max_ice_cover_with_glerl()