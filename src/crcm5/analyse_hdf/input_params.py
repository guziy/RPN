__author__ = 'huziy'

from datetime import datetime
import os
import re

from matplotlib.patches import Rectangle

from crcm5 import infovar
from util.geo.index_shapes import IndexRectangle, IndexPoint
from . import do_analysis_using_pytables as analysis


class InputParams(object):
    def __init__(self, **kwargs):
        #Path to the input hdf file
        default_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup_ITFS.hdf5"
        self.hdf_path = kwargs.get("hdf_path", default_path)

        #Start of the period of interest
        self.start_date = kwargs.get("start_date", datetime(1980, 1, 1))

        #End of the period of interest
        self.end_date = kwargs.get("end_date", datetime(2010, 12, 31))

        #Variable list to be processed
        self.var_list = kwargs.get("var_list", ["I0", "I1", "I2", "I1+I2"])


        #where to store plotted images
        self.img_folder = "images_for_lake-river_paper"
        self.img_folder = os.path.join(self.img_folder, os.path.basename(self.hdf_path))


        #Define averaging region
        default_rectangle = IndexRectangle(
            lower_left_point=IndexPoint(50, 50),
            width=10, height=10
        )
        self.rectangle = kwargs.get("rectangle", default_rectangle)

        #If not for comparison that the object should take care about creating
        #folders for generated figures
        self.is_for_comparison = kwargs.get("is_for_comparison", False)
        if not self.is_for_comparison:
            self._create_image_folder()
        self._check_myself()

        self.lons2d, self.lats2d, self.basemap = self._get_lats2d_lons2d_basemap()


    def _create_image_folder(self):
        if not os.path.isdir(self.img_folder):
            os.makedirs(self.img_folder)
        print("image files will be stored in {0}".format(self.img_folder))

    def _check_myself(self):
        assert self.start_date <= self.end_date, "End date should be after the start date !"
        assert self.hdf_path is not None, "Path to the hdf file should not be None"
        assert len(self.hdf_path.strip()) > 0, "Path to the hdf file should not be empty"
        assert self.start_date.month == 1, "The month of the start_date should be January (1)"
        assert os.path.isfile(self.hdf_path), "Input file '{0}' does not exist...".format(self.hdf_path)

    def get_imfilename_for_var(self, var_name=""):
        """
        :param var_name:
        :return: file name of the image based on varname and the region of interest
        """
        params = [
            var_name,
            self.rectangle.lower_left_point.i,
            self.rectangle.lower_left_point.j,
            self.rectangle.width, self.rectangle.height,
            self.start_date.year, self.end_date.year
        ]
        return "{0}_{1}_{2}_{3}_{4}_{5}-{6}.jpg".format(*params)

    def _get_lats2d_lons2d_basemap(self):
        return analysis.get_basemap_from_hdf(file_path=self.hdf_path)


    def get_start_end_indices_of_selected_region(self):
        """
        :return: imin, jmin, w, h of the selected rectangle in index space
        """
        imin, jmin = self.rectangle.lower_left_point.i, self.rectangle.lower_left_point.j
        w, h = self.rectangle.width, self.rectangle.height
        return imin, jmin, w, h

    def get_land_mask_using_flow_dirs(self):
        fld = analysis.get_array_from_file(self.hdf_path, infovar.HDF_FLOW_DIRECTIONS_NAME)
        return fld > 0


    def calculate_mean_clim_for_3d_var(self, var_name=None):
        """
        :param var_name:
        :return: dates, levels, data(t, lev, x, y)
        """
        start_year = self.start_date.year
        end_year = self.end_date.year

        if ("+" in var_name) or ("-" in var_name) or ("*" in var_name):
            a_name, b_name = [s.strip() for s in re.split(r"[\+/\-\*]", var_name)]
            _, _, a = analysis.get_daily_climatology_of_3d_field(path_to_hdf_file=self.hdf_path,
                                                                 var_name=a_name,
                                                                 start_year=start_year, end_year=end_year)

            dates, levels, b = analysis.get_daily_climatology_of_3d_field(
                path_to_hdf_file=self.hdf_path,
                var_name=b_name,
                start_year=start_year, end_year=end_year)
            data = eval(var_name, {a_name: a, b_name: b})
        else:
            dates, levels, data = analysis.get_daily_climatology_of_3d_field(path_to_hdf_file=self.hdf_path,
                                                                             var_name=var_name,
                                                                             start_year=start_year,
                                                                             end_year=end_year)
        return dates, levels, data


    def get_mpl_rectangle_for_selected_region(self):
        x, y = self.basemap(self.lons2d, self.lats2d)

        #calculate the profile
        imin, jmin, w, h = self.get_start_end_indices_of_selected_region()
        wx = x[imin + w, jmin] - x[imin, jmin]
        hy = y[imin, jmin + h] - y[imin, jmin]
        return Rectangle((x[imin, jmin], y[imin, jmin]), wx, hy, fc="none", linewidth=2)

