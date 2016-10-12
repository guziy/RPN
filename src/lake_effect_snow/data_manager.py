from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pandas._period import Period
from rpn.rpn import RPN
from xarray import DataArray
from xarray import Dataset

from lake_effect_snow import data_source_types
from lake_effect_snow.base_utils import VerticalLevel
from pendulum import Period

class DataManager(object):

    def __init__(self, store_config=None):

        """
        :param store_config: a dictionary containing the information about the input data layout
         store_config["min_dt"] - is a minimum time step between the data
        """

        # mapping date -> data, path
        self.yearmonth_to_path = {}
        self.min_dt = store_config["min_dt"]
        self.varname_mapping = store_config["varname_mapping"]
        self.level_mapping = store_config["level_mapping"]

        self.store_config = store_config
        self.data_source_type = store_config["data_source_type"]

        self.base_folder = self.store_config["base_folder"]


        self.offsets = store_config["offset_mapping"] if "offset_mapping" in store_config else defaultdict(lambda: 0)
        self.multipliers = store_config["multiplier_mapping"] if "multiplier_mapping" in store_config else defaultdict(lambda: 1)


        if self.data_source_type == data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT:
            # TODO: implement
            pass
        elif self.data_source_type == data_source_types.ALL_VARS_IN_A_SINGLE_NETCDF_FILE:
            # TODO: implement
            # Construct the dictionary {varname: {date range: path}}
            #
            pass
        elif self.data_source_type == data_source_types.ALL_VARS_IN_A_FOLDER_OF_RPN_FILES:
            self.init_mappings_all_vars_in_a_folder_of_rpn_files()
        else:
            raise IOError("Unrecognized input data layout")


    def init_mappings_for_all_vars_in_a_single_netcdf_file(self):
        pass


    def init_mappings_all_vars_in_a_folder_of_rpn_files(self):
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
        """
        assert isinstance(period, Period)

        level, level_kind = -1, -1

        if varname_internal in self.level_mapping:
            lvl = self.level_mapping[varname_internal]
            assert isinstance(lvl, VerticalLevel)
            level, level_kind = lvl.get_value_and_kind()


        data = {}
        lons, lats = None, None


        # for each datasource type the following arrays should be defined:
        #       data(t, x, y), dates(t), lons(x, y), lats(x, y)
        if self.data_source_type == data_source_types.ALL_VARS_IN_A_FOLDER_OF_RPN_FILES:


            for i, year in enumerate(range(period.start.year, period.end.year + 1)):
                month_min = 1
                month_max = 12

                if year == period.start.year:
                    month_min = period.start.month

                if year == period.end.year:
                    month_max = period.end.month


                for m in range(month_min, month_max + 1):
                    f = self.yearmonth_to_path[(year, m)]

                    r = RPN(str(f))
                    # read the data into memory
                    data1 = r.get_all_time_records_for_name_and_level(varname=self.varname_mapping[varname_internal], level=level, level_kind=level_kind)

                    if lons is None:
                        lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()

                    data.update(data1)

                    r.close()

            dates = list(sorted(data))[:-1]  # Ignore the last date because it is from the next month
            data = [data[d] for d in dates]
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
            "data": data,
            "name": varname_internal
        }

        # Convert units based on supplied mappings
        return self.multipliers[varname_internal] * DataArray.from_dict(vardict) + self.offsets[varname_internal]





