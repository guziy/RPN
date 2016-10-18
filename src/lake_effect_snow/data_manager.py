from collections import defaultdict
from datetime import datetime
from pathlib import Path

import xarray
from pandas._period import Period
from rpn.rpn import RPN
from xarray import DataArray
from xarray import Dataset

from lake_effect_snow import data_source_types
from lake_effect_snow.base_utils import VerticalLevel
from pendulum import Period
import numpy as np


class DataManager(object):

    def __init__(self, store_config=None):

        """
        :param store_config: a dictionary containing the information about the input data layout
         store_config["min_dt"] - is a minimum time step between the data
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


        # Do the prliminary mappings for faster access ...
        if self.data_source_type == data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT:
            self.varname_to_file_prefix = store_config["filename_prefix_mapping"]
            self.init_mappings_samples_folder_crcm_output()
            pass
        elif self.data_source_type == data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES:
            # TODO: implement
            # Construct the dictionary {varname: {date range: path}}
            #
            pass
        elif self.data_source_type == data_source_types.ALL_VARS_IN_A_FOLDER_OF_RPN_FILES:
            self.init_mappings_all_vars_in_a_folder_of_rpn_files()
        else:
            raise IOError("Unrecognized input data layout")




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


            for month_start in period.range("months"):
                f = self.yearmonth_to_path[(month_start.year, month_start.month)]

                r = RPN(str(f))
                # read the data into memory
                data1 = r.get_all_time_records_for_name_and_level(varname=self.varname_mapping[varname_internal],
                                                                      level=level, level_kind=level_kind)

                if lons is None:
                   lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()

                data.update(data1)

                r.close()

            dates = list(sorted(data))[:-1]  # Ignore the last date because it is from the next month
            data_list = [data[d] for d in dates]

        elif self.data_source_type == data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT:

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

                    if lons is None:
                       lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()

                    r.close()


            dates = list(sorted(data))[:-1]  # Ignore the last date because it is from the next month
            data_list = [data[d] for d in dates]

        elif self.data_source_type == data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES:
            base_folder = Path(self.base_folder)
            ds = xarray.open_mfdataset(str(base_folder.joinpath("*")))

            # select the variable by name and time
            var = ds[self.varname_mapping[varname_internal]].loc[period.start:period.end].squeeze()

            need_to_create_meshgrid = False
            for cname, cvals in var.coords.items():
                if "time" in cname.lower():
                    dates = cvals

                if "lon" in cname.lower():
                    lons = cvals

                    if lons.ndim == 1:
                        need_to_create_meshgrid = True

                if "lat" in cname.lower():
                    lats = cvals

            if need_to_create_meshgrid:
                lats, lons = np.meshgrid(lats.values, lons.values)


            if var.ndim > 3:
                var = var[:, self.level_mapping[varname_internal], :, :]

            if var.shape[-2:] == lons.shape:
                data_list = var.values
            else:
                data_list = np.transpose(var.values, axes=(0, 2, 1))

        else:
            raise NotImplementedError("reading of the layout type {} is not implemented yet.".format(self.data_source_type))


        # print(dates[0], dates[1], "...", dates[-1], len(dates))

        # Construct a dictionary for xarray.DataArray ...
        vardict = {
            "coords": {
                "t": {"dims": "t", "data": dates},
                "lon": {"dims": ("x", "y"), "data": lons},
                "lat": {"dims": ("x", "y"), "data": lats},
            },
            "dims": ("t", "x", "y"),
            "data": data_list,
            "name": varname_internal
        }


	if len(data_list) == 0:
            raise IOError("Could not find any data for the period {}".format(period))
        # Convert units based on supplied mappings
        return self.multipliers[varname_internal] * DataArray.from_dict(vardict) + self.offsets[varname_internal]





