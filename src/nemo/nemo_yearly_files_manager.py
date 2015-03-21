from netCDF4 import Dataset, OrderedDict
import os
import cartopy
from iris.cube import Cube
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.basemap import Basemap
from rpn.domains.rotated_lat_lon import RotatedLatLon
import matplotlib.pyplot as plt

import iris.quickplot as qplt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from iris import analysis as ianalysis
from scipy.spatial.ckdtree import cKDTree
from util import plot_utils
from util.geo import lat_lon
import numpy as np

__author__ = 'huziy'

import iris
from iris import coord_categorisation


class NemoYearlyFilesManager(object):
    def __init__(self, folder="", bathymetry_file="bathy_meter.nc",
                 proj_file="gemclim_settings.nml", suffix="_T.nc"):
        self.data_folder = folder
        self.bathymetry_file = bathymetry_file
        self.proj_file = proj_file
        self.suffix = suffix


        self.kdtree = None
        self.lons = None
        self.lats = None
        self.basemap = None
        self.lake_mask = None
        self.get_basemap_and_coords()
        self.define_lake_mask()

        self.year_to_path = None
        self._build_year_to_datapath_map()
        pass


    def define_lake_mask(self):
        c = iris.load_cube(os.path.join(self.data_folder, self.bathymetry_file),
                           constraint=iris.Constraint(cube_func=lambda f: f.var_name == "Bathymetry"))
        self.lake_mask = (c.data > 0.5).transpose()


    def _build_year_to_datapath_map(self):
        """

        build the relation {year => data path}
        """
        self.year_to_path = {
            int(fn.split("_")[2][:-4]): os.path.join(self.data_folder, fn)
            for fn in os.listdir(self.data_folder) if fn.endswith(self.suffix)
        }


    def get_seasonal_means(self, start_year=None, end_year=None, season_to_months=None):

        """

        :param start_year:
        :param end_year:
        :param season_to_months:
        :return: dict(year -> season -> field)
        """

        def group_key(c, val):
            for k, months in season_to_months.iteritems():
                if val in months:
                    return k

        result = {}
        for the_year in range(start_year, end_year + 1):
            result[the_year] = {}
            data_path = self.year_to_path[the_year]
            cube = iris.load_cube(data_path, "Sea Surface temperature")
            print cube
            coord_categorisation.add_month_number(cube, "time")
            coord_categorisation.add_categorised_coord(cube, "season", "month_number", group_key)

            assert isinstance(cube, Cube)
            seas_mean = cube.aggregated_by(["season"], iris.analysis.MEAN)


            assert isinstance(seas_mean, Cube)
            assert isinstance(self.basemap, Basemap)

            rotpole = ccrs.RotatedPole(pole_longitude=self.basemap.projparams["lon_0"] + 180,
                                       pole_latitude=self.basemap.projparams["o_lat_p"])

            xll, yll = rotpole.transform_point(self.lons[0, 0], self.lats[0, 0], ccrs.Geodetic())
            xur, yur = rotpole.transform_point(self.lons[-1, -1], self.lats[-1, -1], ccrs.Geodetic())

            # And add some more features.
            scale = "50m"
            rivers = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale,
                                                         edgecolor='k', facecolor='none', linewidth=0.8)
            lakes = cartopy.feature.NaturalEarthFeature('physical', 'lakes', scale,
                                                        edgecolor='k', facecolor='none')

            # xx, yy = self.basemap(self.lons, self.lats)

            for the_season in season_to_months.keys():
                c = iris.Constraint(season=the_season)
                the_mean = seas_mean.extract(c)
                assert isinstance(the_mean, Cube)
                result[the_year][the_season] = the_mean.data.transpose()

                # im = self.basemap.pcolormesh(xx, yy, the_mean.data.transpose())
                # self.basemap.colorbar(im)
                # plt.show()


                # ax = plt.subplot(1, 1, 1, projection=rotpole)
                # ax.set_extent([xll, xur, yll, yur], crs=rotpole)
                #
                #
                # ax.contourf(self.lons, self.lats, the_mean.data.transpose(), 10, transform=rotpole)
                # ax.coastlines(resolution="50m")
                # # ax.add_feature(cartopy.feature.LAKES, resolution="50m")
                # # ax.add_feature(rivers)
                # ax.add_feature(lakes)
                # plt.show()

        return result


    def read_and_interpolate_homa_data(self, path="", start_year=None, end_year=None, season_to_months=None):
        """
        :param path:
        :param target_cube:
        """
        sst = iris.load_cube(path, constraint=iris.Constraint(cube_func=lambda f: f.var_name == "sst"))
        # result_sst = sst.regrid(self.model_cube, ianalysis.Linear())
        print sst


        def group_key(c, val):
            for k, months in season_to_months.iteritems():
                if val in months:
                    return k

        coord_categorisation.add_year(sst, "time")

        result_sst = sst.extract(iris.Constraint(year=lambda y: start_year <= y <= end_year))
        coord_categorisation.add_month_number(result_sst, "time")
        coord_categorisation.add_categorised_coord(result_sst, "season", "month_number", group_key)

        assert isinstance(result_sst, Cube)
        result_sst = result_sst.aggregated_by(["season", "year"], ianalysis.MEAN)

        # read longitudes and latitudes from a file
        lons_source = iris.load_cube(path,
                                     constraint=iris.Constraint(cube_func=lambda f: f.var_name == "lon")).data.flatten()

        lats_source = iris.load_cube(path,
                                     constraint=iris.Constraint(cube_func=lambda f: f.var_name == "lat")).data.flatten()

        # source grid
        xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons_source, lats_source)
        self.kdtree = cKDTree(data=zip(xs, ys, zs))

        # target grid
        xt, yt, zt = lat_lon.lon_lat_to_cartesian(self.lons.flatten(), self.lats.flatten())

        dists, inds = self.kdtree.query(zip(xt, yt, zt))


        print len(inds)

        result = {}
        for the_year in range(start_year, end_year + 1):
            result[the_year] = {}
            for the_season in season_to_months.keys():
                c = iris.Constraint(season=the_season) & iris.Constraint(year=the_year)
                the_mean = result_sst.extract(c)
                assert isinstance(the_mean, Cube)

                print the_mean.data.shape

                result[the_year][the_season] = the_mean.data.flatten()[inds].reshape(self.lons.shape) - 273.15

        return result




    def plot_comparisons_of_seasonal_sst_with_homa_obs(self, start_year=None, end_year=None, season_to_months=None):
        model_data = self.get_seasonal_means(season_to_months=season_to_months,
                                             start_year=start_year, end_year=end_year)

        obs_sst_path = os.path.expanduser("~/skynet3_rech1/nemo_obs_for_validation/GreatLakes_2003_5km-2/sst-glk.nc")

        obs_data = self.read_and_interpolate_homa_data(path=obs_sst_path, start_year=start_year, end_year=end_year,
                                                       season_to_months=season_to_months)

        plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=20, height_cm=10)
        # calculate climatologic differences
        diff = {}
        for season in season_to_months.keys():
            diff[season] = np.mean(
                [model_data[y][season] - obs_data[y][season] for y in range(start_year, end_year + 1)], axis=0)
            diff[season] = np.ma.masked_where(~self.lake_mask, diff[season])


        # plot seasonal biases
        xx, yy = self.basemap(self.lons, self.lats)


        # calculate difference ranges
        diff_max = 0
        for season, the_diff in diff.iteritems():
            diff_max = max(np.percentile(np.abs(the_diff[~the_diff.mask]), 90), diff_max)
        diff_max = 5

        locator = MaxNLocator(nbins=12, symmetric=True)
        bounds = locator.tick_values(-diff_max, diff_max)
        bn = BoundaryNorm(bounds, len(bounds) - 1)
        cmap = cm.get_cmap("RdBu_r", len(bounds) - 1)

        im = None
        fig = plt.figure()
        ncols = 2
        fig.suptitle(r"SST $\left({\rm ^\circ C}\right)$", font_properties=FontProperties(weight="bold"))
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
        fig.savefig(os.path.join(nemo_img_dir, "sst_homa_validation.pdf"))
        plt.show()


    def get_basemap_and_coords(self):
        """
        :return: lons2d, lats2d, basemap [based on the bathymetry file and gemclim_settings.nml]
        """

        # Read longitudes and latitudes and create the basemap only if they are not initialized
        if self.lons is None:
            ds = Dataset(os.path.join(self.data_folder, self.bathymetry_file))
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
            self.basemap = rll.get_basemap_object_for_lons_lats(lons2d=self.lons, lats2d=self.lats)
            print lon1, lat1, lon2, lat2


        # self.basemap.drawcoastlines()
        # xx, yy = self.basemap(self.lons, self.lats)
        # self.basemap.pcolormesh(xx, yy, ds.variables["Bathymetry"][:].transpose())
        # plt.show()

        self.lons[self.lons > 180] -= 360


        return self.lons, self.lats, self.basemap


def main():
    nemo_manager = NemoYearlyFilesManager(folder="/home/huziy/skynet3_rech1/offline_glk_output_daily_1979-2012")

    # Study period
    start_year = 2003
    end_year = 2006

    season_to_months = OrderedDict([
        ("Winter", (12, 1, 2)),
        ("Spring", range(3, 6)),
        ("Summer", range(6, 9)),
        ("Fall", range(9, 12))
    ])

    nemo_manager.plot_comparisons_of_seasonal_sst_with_homa_obs(
        start_year=start_year, end_year=end_year, season_to_months=season_to_months
    )


if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()
    main()