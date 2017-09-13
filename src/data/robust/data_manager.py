from collections import defaultdict
from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np
import xarray
from pendulum import Pendulum
from pendulum import Period
from rpn.rpn import RPN
from scipy.spatial import KDTree
from xarray import DataArray

from data.robust import data_source_types
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.default_varname_mappings import LAKE_ICE_FRACTION
from util.geo import lat_lon


class DataManager(object):

    def __init__(self, store_config=None):

        """
        :param store_config: a dictionary containing the information about the input data layout
         store_config["min_dt"] - is a minimum time step between the data
         important limitation: All data files should be on the same horizontal grid
        """

        # mapping date -> data, path
        self.yearmonth_to_path = {}
        if "min_dt" in store_config:
            self.min_dt = store_config["min_dt"]

        self.varname_mapping = store_config["varname_mapping"]
        self.level_mapping = store_config["level_mapping"]

        self.store_config = store_config
        self.data_source_type = store_config["data_source_type"]

        self.base_folder = self.store_config["base_folder"]


        self.offsets = store_config["offset_mapping"] if "offset_mapping" in store_config else defaultdict(lambda: 0)
        self.multipliers = store_config["multiplier_mapping"] if "multiplier_mapping" in store_config else defaultdict(lambda: 1)


        self.varname_to_file_path = None

        # Do the prliminary mappings for faster access ...
        if self.data_source_type == data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT:
            self.varname_to_file_prefix = store_config["varname_to_filename_prefix_mapping"]
            self.init_mappings_samples_folder_crcm_output()
        elif self.data_source_type == data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT_VNAME_IN_FNAME:
            self.init_mappings_samples_folder_crcm_output()
        elif self.data_source_type == data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES:
            pass
        elif self.data_source_type == data_source_types.ALL_VARS_IN_A_FOLDER_OF_RPN_FILES:
            self.init_mappings_all_vars_in_a_folder_of_rpn_files()
        elif self.data_source_type == data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES_OPEN_EACH_FILE_SEPARATELY:
            self.init_mappings_all_vars_in_a_folder_in_netcdf_files_open_each_file()
            pass
        else:
            raise IOError("Unrecognized input data layout")


        self.lons, self.lats = None, None
        self.kdtree = None






    def init_mappings_all_vars_in_a_folder_in_netcdf_files_open_each_file(self):

        base_folder_p = Path(self.base_folder)

        self.varname_to_file_path = {}
        for f in base_folder_p.iterdir():
            if not f.name.endswith(".nc"):
                continue

            with netCDF4.Dataset(f) as ds:
                for internal_vname, file_vname in self.varname_mapping.items():
                    for vname, var in ds.variables.items():
                        if vname == file_vname:
                            self.varname_to_file_path[internal_vname] = str(f)



    def init_mappings_samples_folder_crcm_output(self):
        """
        maps (year, month) to the folder path with files
        """
        base_folder = Path(self.base_folder)

        for f in base_folder.iterdir():

            if not f.is_dir():
                continue

            tok = f.name.split("_")[-1]
            try:
                d = datetime.strptime(tok, "%Y%m")
                self.yearmonth_to_path[(d.year, d.month)] = f
            except ValueError:
                print("Skipping {}".format(f))
                continue

    def init_mappings_all_vars_in_a_folder_of_rpn_files(self):
        """
        map (year, month) pairs to the paths for quicker access later
        """
        base_folder = Path(self.base_folder)

        for f in base_folder.iterdir():
            tok = f.name.split("_")[-1]
            try:
                d = datetime.strptime(tok, "%Y%m")
                self.yearmonth_to_path[(d.year, d.month)] = f
            except ValueError:
                print("Skipping {}".format(f))
                continue


    def read_data_for_period(self, period: Period, varname_internal: str) -> DataArray:

        """
        Read the data for period and varname into memory, and return it as xarray DataArray
        :param period:
        :param varname_internal:

        Note: this method will read everything into memory, please be easy on the period duration for large datasets
        """
        assert isinstance(period, Period)

        level, level_kind = -1, -1

        if varname_internal in self.level_mapping:
            lvl = self.level_mapping[varname_internal]
            assert isinstance(lvl, VerticalLevel)
            level, level_kind = lvl.get_value_and_kind()


        data = {}
        lons, lats = None, None
        data_list = None
        dates = None


        # for each datasource type the following arrays should be defined:
        #       data(t, x, y), dates(t), lons(x, y), lats(x, y)
        if self.data_source_type == data_source_types.ALL_VARS_IN_A_FOLDER_OF_RPN_FILES:

            assert isinstance(period, Period)

            for month_start in period.range("months"):
                f = self.yearmonth_to_path[(month_start.year, month_start.month)]

                r = RPN(str(f))
                # read the data into memory
                data1 = r.get_all_time_records_for_name_and_level(varname=self.varname_mapping[varname_internal],
                                                                      level=level, level_kind=level_kind)

                if self.lons is None:
                   self.lons, self.lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()

                data.update(data1)

                r.close()

            dates = list(sorted(data))[:-1]  # Ignore the last date because it is from the next month
            data_list = [data[d] for d in dates]

        elif self.data_source_type == data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT:


            assert varname_internal in self.varname_to_file_prefix, "Could not find {} in {}".format(
                varname_internal, self.varname_to_file_prefix
            )

            filename_prefix = self.varname_to_file_prefix[varname_internal]

            for month_start in period.range("months"):

                year, m = month_start.year, month_start.month
              
                print(year, m)

                # Skip years or months that are not available
                if (year, m) not in self.yearmonth_to_path:
                    print("Skipping {}-{}".format(year, m))
                    continue

                month_dir = self.yearmonth_to_path[(year, m)]

                for f in month_dir.iterdir():
                    # Skip the file for time step 0
                    if f.name[-9:-1] == "0" * 8:
                        continue

                    # read only files with the specified prefix
                    if not f.name.startswith(filename_prefix):
                        continue

                    r = RPN(str(f))

                    data.update(r.get_all_time_records_for_name_and_level(varname=self.varname_mapping[varname_internal],
                                                                              level=level, level_kind=level_kind))

                    if self.lons is None:
                       self.lons, self.lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()

                    r.close()


            dates = list(sorted(data))[:-1]  # Ignore the last date because it is from the next month
            data_list = [data[d] for d in dates]

        elif self.data_source_type == data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT_VNAME_IN_FNAME:
            for month_start in period.range("months"):

                year, m = month_start.year, month_start.month

                print(year, m)

                # Skip years or months that are not available
                if (year, m) not in self.yearmonth_to_path:
                    print("Skipping {}-{}".format(year, m))
                    continue

                month_dir = self.yearmonth_to_path[(year, m)]

                for f in month_dir.iterdir():
                    # read only files containing the variable name in the name, i.e. *TT*.rpn
                    if not "_" + self.varname_mapping[varname_internal] in f.name:
                        continue

                    r = RPN(str(f))

                    data.update(
                        r.get_all_time_records_for_name_and_level(varname=self.varname_mapping[varname_internal],
                                                                  level=level, level_kind=level_kind))

                    if self.lons is None:
                        self.lons, self.lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()

                    r.close()

            dates = list(sorted(data))[:-1]  # Ignore the last date because it is from the next month
            data_list = [data[d] for d in dates]

        elif self.data_source_type in [data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES,
                                       data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES_OPEN_EACH_FILE_SEPARATELY]:

            if self.varname_to_file_path is None:
                base_folder = Path(self.base_folder)
                ds = xarray.open_mfdataset(str(base_folder.joinpath("*.nc")))
            else:
                ## In the case of very different netcdf files in the folder
                ## i.e. data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES_OPEN_EACH_FILE_SEPARATELY
                ds = xarray.open_dataset(self.varname_to_file_path[varname_internal])
                print("reading {} from {}".format(varname_internal, self.varname_to_file_path[varname_internal]))

            # select the variable by name and time
            var = ds[self.varname_mapping[varname_internal]].sel(time=slice(period.start, period.end)).squeeze()

            for cname, cvals in var.coords.items():
                if "time" in cname.lower():
                    dates = cvals

            if self.lons is None:
                need_to_create_meshgrid = False
                for cname, cvals in var.coords.items():

                    if "lon" in cname.lower():
                        lons = cvals.values

                        if lons.ndim == 1:
                            need_to_create_meshgrid = True

                    if "lat" in cname.lower():
                        lats = cvals.values

                if need_to_create_meshgrid:
                    lats, lons = np.meshgrid(lats, lons)

                self.lons, self.lats = lons, lats


            # if still could not find longitudes and latitudes
            if self.lons is None:

                for vname, ncvar in ds.items():
                    if "lon" in vname.lower():
                        self.lons = ncvar.values

                    if "lat" in vname.lower():
                        self.lats = ncvar.values



            if var.ndim > 3:
                var = var[:, self.level_mapping[varname_internal], :, :]

            if var.shape[-2:] == self.lons.shape:
                data_list = var.values
            else:
                print(self.lons.shape, var.shape, var.name)
                if var.ndim == 3:
                    data_list = np.transpose(var.values, axes=(0, 2, 1))
                elif var.ndim == 2:
                    data_list = np.transpose(var.values)
                else:
                    raise Exception("{}-dimensional variables are not supported".format(var.ndim))

            # close the dataset
            ds.close()

        else:

            raise NotImplementedError("reading of the layout type {} is not implemented yet.".format(self.data_source_type))


        # print(dates[0], dates[1], "...", dates[-1], len(dates))

        # Construct a dictionary for xarray.DataArray ...
        vardict = {
            "coords": {
                "t": {"dims": "t", "data": dates},
                "lon": {"dims": ("x", "y"), "data": self.lons},
                "lat": {"dims": ("x", "y"), "data": self.lats},
            },
            "dims": ("t", "x", "y"),
            "data": data_list,
            "name": varname_internal
        }


        if len(data_list) == 0:
            print("retreived dates: {}".format(dates))
            raise IOError("Could not find any {} data for the period {}..{} in {}".format(self.varname_mapping[varname_internal],
                                                                                    period.start, period.end, self.base_folder))
        # Convert units based on supplied mappings
        return self.multipliers[varname_internal] * DataArray.from_dict(vardict) + self.offsets[varname_internal]



    def get_kdtree(self):

        if self.lons is None:
            raise Exception("The coordinates (lons and lats) are not yet set for the manager, please read some data first")

        if self.kdtree is None:
            xs, ys, zs = lat_lon.lon_lat_to_cartesian(self.lons.flatten(), self.lats.flatten())
            self.kdtree = KDTree(list(zip(xs, ys, zs)))

        return self.kdtree


    def get_seasonal_means(self, start_year:int, end_year:int, season_to_months:dict, varname_internal:str):

        """
        returns a dictionary {season:{year: mean_field}}
        :param start_year:
        :param end_year:
        :param season_to_months:

        (order of months in the list of months is important, i.e. for DJF the order should be [12, 1, 2])
        """
        result = defaultdict(dict)



        for season, months in season_to_months.items():

            for y in range(start_year, end_year + 1):
                d1 = Pendulum(y, months[0], 1)
                d2 = d1.add(months=len(months)).subtract(seconds=1)

                if d2.year > end_year:
                    continue

                current_period = Period(d1, d2)
                print("calculating mean for [{}, {}]".format(current_period.start, current_period.end))
                data = self.read_data_for_period(current_period, varname_internal)


                result[season][y] = data.mean(dim="t").values

        return result


    def get_seasonal_maxima(self, start_year:int, end_year:int, season_to_months:dict, varname_internal:str):

        """
        returns a dictionary {season:{year: field of maxima}}
        :param start_year:
        :param end_year:
        :param season_to_months:

        (order of months in the list of months is important, i.e. for DJF the order should be [12, 1, 2])
        """
        result = defaultdict(dict)



        for season, months in season_to_months.items():

            for y in range(start_year, end_year + 1):
                d1 = Pendulum(y, months[0], 1)
                d2 = d1.add(months=len(months)).subtract(seconds=1)

                if d2.year > end_year:
                    continue

                current_period = Period(d1, d2)
                print("calculating mean for [{}, {}]".format(current_period.start, current_period.end))
                data = self.read_data_for_period(current_period, varname_internal)

                if varname_internal == LAKE_ICE_FRACTION:
                    result[season][y] = np.ma.masked_where(data.values > 1, data.values).max(axis=0)
                else:
                    result[season][y] = data.max(dim="t").values


        return result



    def get_min_max_avg_for_short_period(self, start_time:Pendulum, end_time:Pendulum, varname_internal:str):
        """
        The short period means that all the data from the period fits into RAM
        :param start_time:
        :param end_time:
        :param varname_internal:
        :return:
        """
        p = Period(start_time, end_time)
        data = self.read_data_for_period(p, varname_internal=varname_internal)


        min_current = data.min(dim="t").values
        max_current = data.max(dim="t").values
        avg_current = data.mean(dim="t").values


        min_dates = _get_dates_for_extremes(min_current, data)
        min_dates.name = "min_dates"

        max_dates = _get_dates_for_extremes(max_current, data)
        max_dates.name = "max_dates"



        # assign names
        min_vals = xarray.DataArray(name="min_{}".format(varname_internal), data=min_current, dims=("x", "y"))
        max_vals = xarray.DataArray(name="max_{}".format(varname_internal), data=max_current, dims=("x", "y"))
        avg_vals = xarray.DataArray(name = "avg_{}".format(varname_internal), data=avg_current, dims=("x", "y"))


        result = {
            min_vals.name: min_vals,
            min_dates.name: min_dates,
            max_vals.name: max_vals,
            max_dates.name: max_dates,
            avg_vals.name: avg_vals
        }

        return result


    def get_min_max_avg_for_period(self, start_year:int, end_year:int, varname_internal:str):
        """

        :param start_year:
        :param end_year:
        :param varname_internal:
        """

        min_vals = None
        max_vals = None
        avg_vals = None

        min_dates = None
        max_dates = None


        avg_n = 0


        for y in range(start_year, end_year + 1):

            p_start = Pendulum(y, 1, 1)
            p_end = Pendulum(y + 1, 1, 1).subtract(microseconds=1)
            p = Period(p_start, p_end)
            data = self.read_data_for_period(p, varname_internal=varname_internal)


            min_current = data.min(dim="t").values
            max_current = data.max(dim="t").values
            avg_current = data.mean(dim="t").values



            # Find extremes and dates when they are occurring
            if min_vals is None:
                min_vals = min_current
            else:
                min_vals = np.where(min_vals <= min_current, min_vals, min_current)


            if max_vals is None:
                max_vals = max_current
            else:
                max_vals = np.where(max_vals >= max_current, max_vals, max_current)

            min_dates = _get_dates_for_extremes(min_vals, data, min_dates)

            assert min_dates is not None

            max_dates = _get_dates_for_extremes(max_vals, data, max_dates)


            # calculate the mean
            if avg_vals is None:
                avg_vals = avg_current
                avg_n = data.shape[0]
            else:
                incr = data.shape[0]
                # calculate the mean incrementally to avoid overflow
                avg_vals = avg_vals * (avg_n / (avg_n + incr)) + (incr / (avg_n + incr)) * avg_current



        # assign names
        min_vals = xarray.DataArray(name="min_{}".format(varname_internal), data=min_vals, dims=("x", "y"))
        min_dates.name = "min_dates"

        max_vals = xarray.DataArray(name="max_{}".format(varname_internal), data=max_vals, dims=("x", "y"))
        max_dates.name = "max_dates"

        avg_vals = xarray.DataArray(name = "avg_{}".format(varname_internal), data=avg_vals, dims=("x", "y"))




        result = {
            min_vals.name: min_vals,
            min_dates.name: min_dates,
            max_vals.name: max_vals,
            max_dates.name: max_dates,
            avg_vals.name: avg_vals
        }

        return result



def _get_dates_for_extremes(extr_vals: xarray.DataArray, current_data_chunk: xarray.DataArray, extr_dates:xarray.DataArray=None):

    """
    Helper method to determine the times when the extreme values are occurring
    :param extr_vals:
    :param current_data_chunk:
    :param result_dates:
    """
    t3d, _ = xarray.broadcast(current_data_chunk.t, current_data_chunk)

    if extr_dates is None:
        result_dates = t3d[0, :, :].copy()
    else:
        result_dates = extr_dates


    tis, xis, yis = np.where(extr_vals == current_data_chunk)


    npvals = t3d.values
    # for ti, xi, yi in zip(tis, xis, yis):
    #     result_dates[xi, yi] = npvals[ti, xi, yi]

    result_dates.values[xis, yis] = npvals[tis, xis, yis]

    # debug


    return result_dates







