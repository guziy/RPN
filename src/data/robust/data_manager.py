from collections import defaultdict
from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np
import xarray
from mpl_toolkits.basemap import Basemap
from pendulum import Pendulum
from pendulum import Period
from rpn.domains.rotated_lat_lon import RotatedLatLon
from rpn.rpn import RPN
from scipy.spatial import KDTree
from xarray import DataArray

import util.stat_helpers
from data.robust import data_source_types
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.default_varname_mappings import LAKE_ICE_FRACTION
from util.geo import lat_lon
from lake_effect_snow import default_varname_mappings
import pandas as pd


def _get_period_for_year(y):
    start = Pendulum(y, 1, 1)
    end = Pendulum(y + 1, 1, 1).subtract(microseconds=1)
    return Period(start, end)


class DataManager(object):
    # Names of the storage properties
    SP_INTERNAL_TO_INPUT_VNAME_MAPPING = "varname_mapping"
    SP_LEVEL_MAPPING = "level_mapping"
    SP_DATASOURCE_TYPE = "datasource_type"
    SP_BASE_FOLDER = "base_folder"
    SP_OFFSET_MAPPING = "offset_mapping"
    SP_MULTIPLIER_MAPPING = "multiplier_mapping"
    SP_VARNAME_TO_FILENAME_PREFIX_MAPPING = "varname_to_filename_prefix_mapping"

    STORE_PROPS = [
        SP_INTERNAL_TO_INPUT_VNAME_MAPPING,
        SP_LEVEL_MAPPING,
        SP_DATASOURCE_TYPE,
        SP_BASE_FOLDER,
        SP_OFFSET_MAPPING,
        SP_MULTIPLIER_MAPPING
    ]

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

        self.varname_mapping = store_config[self.SP_INTERNAL_TO_INPUT_VNAME_MAPPING]
        self.level_mapping = store_config[self.SP_LEVEL_MAPPING]

        self.store_config = store_config
        self.data_source_type = store_config[self.SP_DATASOURCE_TYPE]

        self.base_folder = self.store_config[self.SP_BASE_FOLDER]

        key = self.SP_OFFSET_MAPPING
        self.offsets = store_config[key] if key in store_config else defaultdict(lambda: 0)

        key = self.SP_MULTIPLIER_MAPPING
        self.multipliers = store_config[key] if key in store_config else defaultdict(lambda: 1)

        self.varname_to_file_path = None

        # Do the prliminary mappings for faster access ...
        if self.data_source_type == data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT:
            self.varname_to_file_prefix = store_config[self.SP_VARNAME_TO_FILENAME_PREFIX_MAPPING]
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

        # 1) when data is extracted from rpn files contains
        # {rlon: rlon, rlat: rlat, }
        self.basemap_info_of_the_last_imported_field = {}



    def export_to_netcdf(self, output_dir_path=None, field_names=None, label="",
                         start_year=1980, end_year=2014, field_metadata=None, global_metadata=None):
        """

        :param output_dir_path:
        :param field_names:
        :param label:
        :param start_year:
        :param end_year:
        :param field_metadata: {field_name: {"units": "mm/day"}, ...} -
                         things you want to attach to the converted netcdf variable
        """
        # TODO: implement getting all the fields, when the argument is None
        assert field_names is not None

        if output_dir_path is None:
            output_dir_path = Path(self.base_folder) / ".." / f"Netcdf_exports_{label}"

            output_dir_path.mkdir(parents=True, exist_ok=True)

        print(field_names)
        default_io_settings = {
                "zlib": True, "dtype": "f4"
        }


        for vname in field_names:

            out_file = output_dir_path / f"{label}_{vname}_{start_year}-{end_year}.nc"

            if out_file.exists():
                print(f"Nothing to do for {out_file}, skipping ...")
                continue


            tmp_files = []
            read_at_least_once = False  # need to read the input data at least once to get the coordinates information
            for y in range(start_year, end_year + 1):

                chunk_out_file = output_dir_path / f"{label}_{vname}_{start_year}-{end_year}_{y}.nc"
                tmp_files.append(str(chunk_out_file))

                if chunk_out_file.exists() and read_at_least_once:
                    print(f"{chunk_out_file} already exists, skipping {y}")
                    continue

                da = self.read_data_for_period(_get_period_for_year(y), vname)

                print(f"{_get_period_for_year(y)}")

                assert isinstance(da, xarray.DataArray)

                da = da.rename(vname)

                # attach some info to the variable
                if field_metadata is not None and y == start_year:
                    da.attrs.update(field_metadata[vname])

                da.to_netcdf(str(chunk_out_file), unlimited_dims=["t"])
                read_at_least_once = True

            with xarray.open_mfdataset(tmp_files, data_vars="minimal", coords="minimal", chunks={"t": 500}) as ds_in:

                if len(self.basemap_info_of_the_last_imported_field) > 0:
                    da = xarray.DataArray(data=0)

                    if "rlon" in self.basemap_info_of_the_last_imported_field:
                        rlon = xarray.DataArray(data=self.basemap_info_of_the_last_imported_field["rlon"], dims=("x",), name="rlon")
                        rlat = xarray.DataArray(data=self.basemap_info_of_the_last_imported_field["rlat"], dims=("y",), name="rlat")

                        ds_in["rlon"] = rlon
                        ds_in["rlat"] = rlat

                    da.attrs.update(
                        {k: v for k, v in self.basemap_info_of_the_last_imported_field.items() if k not in ["rlon", "rlat"]})

                    da.attrs["description"] = ""

                    ds_in["projection"] = da



                encoding = {v: default_io_settings.copy() for v in ["rlon", "rlat", vname]}

                # add some global attrs
                if global_metadata is not None:
                    ds_in.attrs.update(global_metadata)


                ds_in.to_netcdf(str(out_file), unlimited_dims=["t"], encoding=encoding)

            # cleanup, remove temporary files
            for f in tmp_files:
                Path(f).unlink()




    def get_basemap(self, varname_internal, **bmap_kwargs):
        if self.data_source_type == data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT:
            for month_dir in self.base_folder.iterdir():
                if not month_dir.is_dir():
                    continue

                for data_file in month_dir.iterdir():

                    try:
                        # skip files that do not contain the variable
                        if varname_internal in self.varname_to_file_prefix:
                            if not data_file.name.startswith(self.varname_to_file_prefix[varname_internal]):
                                continue

                        with RPN(str(data_file)) as r:
                            r.get_first_record_for_name(self.varname_mapping[varname_internal])
                            lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()
                            rll = RotatedLatLon(**r.get_proj_parameters_for_the_last_read_rec())
                            return rll.get_basemap_object_for_lons_lats(lons, lats, **bmap_kwargs)
                    except Exception:
                        # Try to look into several files before giving up
                        pass
        else:
            raise NotImplementedError("Not impelmented for the data_source_type = {}".format(self.data_source_type))




    def compute_annual_number_of_summer_days(self, temp_treshold_degC=25, start_year: int = 1980, end_year: int=1998,
                                             lons_target=None, lats_target=None, nneighbors=1):
        pass



    def compute_climatological_quantiles(self, q: float = 0.5, rolling_mean_window_days=5,
                                         varname_internal=default_varname_mappings.TOTAL_PREC,
                                         start_year: int = 1980, end_year: int = 2016, daily_agg_func=np.mean,
                                         lons_target=None, lats_target=None, nneighbors=1) -> xarray.DataArray:

        """

        :param rolling_mean_window_days:
        :param q:
        :param varname_internal:
        :param start_year:
        :param end_year:
        :param daily_agg_func:
        :param lats_target: optional, if specify the data is spatially interpolated to these target coordinates
        :param lons_target:
        :param nneighbors: Number of nearest neighbors to consider during spatial interpolation
        """

        # implement caching
        import hashlib
        cache_file = "data_manager_cache/compute_climatological_quantiles/{}_{}_lons{}_lats{}_nneighbors{}_years_{}-{}_daily_agg_{}_rollmeandays{}_q{}.nc".format(
            hashlib.sha1(str(self.base_folder).encode()).hexdigest(), varname_internal,
            hashlib.sha1(str(lons_target).encode()).hexdigest(),
            hashlib.sha1(str(lats_target).encode()).hexdigest(),
            nneighbors,
            start_year, end_year,
            daily_agg_func.__name__, rolling_mean_window_days, q
        )

        out_var_name = "{}_daily_{}_q_{}".format(varname_internal, daily_agg_func.__name__, q)

        # create parent folder if required
        cache_file = Path(cache_file)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():
            print("Using cache file: {}".format(cache_file))
            return xarray.open_dataset(str(cache_file))[out_var_name]

        daily_data = self.__get_daily_aggregates(varname_internal=varname_internal,
                                                 start_year=start_year, end_year=end_year,
                                                 agg_func=daily_agg_func,
                                                 lons_target=lons_target, lats_target=lats_target,
                                                 nneighbors=nneighbors)

        daily_perc_ma = util.stat_helpers.clim_day_percentile_calculator(daily_data.values, daily_data.t,
                                                                      np.nan,
                                                                      rolling_mean_window_days=rolling_mean_window_days,
                                                                      percentile=q, start_year=start_year,
                                                                      end_year=end_year)

        new_coords = {cn: ca for cn, ca in daily_data.coords.items() if cn != "t"}
        t_out = pd.date_range(start="2001-01-01", end="2001-12-31", freq="D")
        new_coords.update({"t": t_out})


        print("calculated percentile shape: {}, n={} masked points".format(daily_perc_ma.shape, daily_perc_ma.mask.sum()))

        daily_perc = xarray.DataArray(daily_perc_ma, dims=daily_data.dims, name=out_var_name, attrs=daily_data.attrs,
                                      coords=new_coords)

        daily_perc = daily_perc.where(~daily_perc_ma.mask)

        daily_perc.to_netcdf(str(cache_file))

        return daily_perc



    def __get_daily_aggregates(self, varname_internal=default_varname_mappings.TOTAL_PREC, start_year: int = 1980,
                               end_year: int = 2016, agg_func=np.mean, lons_target=None, lats_target=None,
                               nneighbors=1) -> DataArray:

        """
        Read and interpolate spatially (if required) and get daily aggregates (min, max, avg)
        :param varname_internal:
        :param start_year:
        :param end_year:
        :param agg_func:
        :param lons_target:
        :param lats_target:
        :param nneighbors:
        :return:
        """
        start_date = Pendulum(start_year, 1, 1)
        end_date = Pendulum(end_year + 1, 1, 1).subtract(microseconds=1)

        all_data = []
        for p_start in Period(start_date, end_date).range("years"):
            p_end = Pendulum(p_start.year + 1, 1, 1).subtract(microseconds=1)
            p = Period(p_start, p_end)
            print("reading {} data for {} -- {}".format(varname_internal, p.start, p.end))


            if lons_target is not None:
                data = self.read_data_for_period_and_interpolate(period=p,
                                                                 varname_internal=varname_internal,
                                                                 lons_target=lons_target,
                                                                 lats_target=lats_target,
                                                                 nneighbors=nneighbors)
            else:
                data = self.read_data_for_period(period=p, varname_internal=varname_internal)

            all_data.append(data.resample("D", dim="t", how=agg_func))

        return xarray.DataArray(xarray.concat(all_data, dim="t"))

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


    def read_data_for_period_and_interpolate(self, period: Period, varname_internal: str,
                                             lons_target, lats_target, ktree=None, nneighbors=1) -> DataArray:
        """
        :param ktree: ktree used for interpolation
        :param nneighbors:
        :param period:
        :param varname_internal:
        :param lons_target: longitudes to where the nn interpolation will be done
        :param lats_target:
        """



        data = self.read_data_for_period(period, varname_internal)

        lons_source, lats_source = data.coords["lon"].values, data.coords["lat"].values

        x_t, y_t, z_t = lat_lon.lon_lat_to_cartesian(lons_target.flatten(), lats_target.flatten())

        if ktree is None:
            ktree = self.get_kdtree(lons=lons_source, lats=lats_source, cache=True)

        dists, inds = ktree.query(list(zip(x_t, y_t, z_t)), k=nneighbors)


        if nneighbors > 1:
            data_res = data.values.reshape(data.shape[0], -1)[:, inds]
            data_res = np.nanmean(data_res, axis=-1).reshape((-1,) + lons_target.shape)
        elif nneighbors == 1:
            data_res = data.values.reshape((data.shape[0], -1))[:, inds].reshape((-1,) + lons_target.shape)
        else:
            raise ValueError(f"nneigbours should be >= 1, not {nneighbors}")



        print(lons_target.shape)


        while lons_target.ndim < len(data.coords["lon"].dims):
            lons_target.shape = lons_target.shape + (1, )
            lats_target.shape = lats_target.shape + (1, )
            data_res.shape = data_res.shape + (1, )


        new_coords = {cn: ca for cn, ca in data.coords.items() if cn in ["t",]}
        new_coords["lon"] = (data.coords["lon"].dims, lons_target)
        new_coords["lat"] = (data.coords["lat"].dims, lats_target)

        print(new_coords)
        print(data_res.shape)

        data_res = xarray.DataArray(data_res, dims=data.dims, name=data.name, attrs=data.attrs, coords=new_coords)

        return data_res


    def __update_bmp_info_from_rpnfile_obj(self, r):
        # save projection paarams for a possible re-use in the future
        proj_params = r.get_proj_parameters_for_the_last_read_rec()
        self.lons, self.lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()
        rlons, rlats = r.get_tictacs_for_the_last_read_record()

        rll = RotatedLatLon(**proj_params)
        bmp = rll.get_basemap_object_for_lons_lats(self.lons, self.lats)

        assert isinstance(bmp, Basemap)

        self.basemap_info_of_the_last_imported_field = {
            "rlon": rlons,
            "rlat": rlats,
        }
        self.basemap_info_of_the_last_imported_field.update(bmp.projparams)

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
                    self.__update_bmp_info_from_rpnfile_obj(r)


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
                    print(f"Skipping {year}-{m}")
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

                    data.update(
                        r.get_all_time_records_for_name_and_level(varname=self.varname_mapping[varname_internal],
                                                                  level=level, level_kind=level_kind))

                    if self.lons is None:
                        self.__update_bmp_info_from_rpnfile_obj(r)

                    r.close()

            dates = list(sorted(data))[:-1]  # Ignore the last date because it is from the next month
            data_list = [data[d] for d in dates]

        elif self.data_source_type == data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT_VNAME_IN_FNAME:
            for month_start in period.range("months"):

                year, m = month_start.year, month_start.month

                print(year, m)

                # Skip years or months that are not available
                if (year, m) not in self.yearmonth_to_path:
                    print("Skipping {year}-{m}")
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
                ds = xarray.open_mfdataset(str(base_folder / "*.nc*"), data_vars="minimal")
            else:
                ## In the case of very different netcdf files in the folder
                ## i.e. data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES_OPEN_EACH_FILE_SEPARATELY
                ds = xarray.open_dataset(self.varname_to_file_path[varname_internal])
                print("reading {} from {}".format(varname_internal, self.varname_to_file_path[varname_internal]))

            # select the variable by name and time
            print(period.start, period.end)
            print(ds[self.varname_mapping[varname_internal]])
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
                    raise Exception(f"{var.ndim}-dimensional variables are not supported")

            # close the dataset
            ds.close()

        else:

            raise NotImplementedError(
                "reading of the layout type {} is not implemented yet.".format(self.data_source_type))

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
            raise IOError(
                "Could not find any {} data for the period {}..{} in {}".format(self.varname_mapping[varname_internal],
                                                                                period.start, period.end,
                                                                                self.base_folder))
        # Convert units based on supplied mappings
        return self.multipliers[varname_internal] * DataArray.from_dict(vardict) + self.offsets[varname_internal]

    def get_kdtree(self, lons=None, lats=None, cache=True):
        """

        :param lons:
        :param lats:
        :param cache: if True then reuse the kdtree
        :return:
        """
        if lons is None:
            lons = self.lons
            lats = self.lats


        if lons is None:
            raise Exception(
                "The coordinates (lons and lats) are not yet set for the manager, please read some data first")

        if cache and self.kdtree is not None:
            return self.kdtree

        xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons.flatten(), lats.flatten())
        kdtree = KDTree(list(zip(xs, ys, zs)))

        if cache:
            self.kdtree = kdtree

        return kdtree

    def get_seasonal_means(self, start_year: int, end_year: int, season_to_months: dict, varname_internal: str):

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

    def get_seasonal_maxima(self, start_year: int, end_year: int, season_to_months: dict, varname_internal: str):

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

    def get_min_max_avg_for_short_period(self, start_time: Pendulum, end_time: Pendulum, varname_internal: str):
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
        avg_vals = xarray.DataArray(name="avg_{}".format(varname_internal), data=avg_current, dims=("x", "y"))

        result = {
            min_vals.name: min_vals,
            min_dates.name: min_dates,
            max_vals.name: max_vals,
            max_dates.name: max_dates,
            avg_vals.name: avg_vals
        }

        return result

    def get_min_max_avg_for_period(self, start_year: int, end_year: int, varname_internal: str):
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

        avg_vals = xarray.DataArray(name="avg_{}".format(varname_internal), data=avg_vals, dims=("x", "y"))

        result = {
            min_vals.name: min_vals,
            min_dates.name: min_dates,
            max_vals.name: max_vals,
            max_dates.name: max_dates,
            avg_vals.name: avg_vals
        }

        return result


def _get_dates_for_extremes(extr_vals: xarray.DataArray, current_data_chunk: xarray.DataArray,
                            extr_dates: xarray.DataArray = None):
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


def test_daily_clim_quantile_calculations():
    dm = DataManager(
        store_config={
            DataManager.SP_BASE_FOLDER: "/snow3/huziy/NEI/WC/NEI_WC0.11deg_Crr1/Samples",
            DataManager.SP_DATASOURCE_TYPE: data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT,
            DataManager.SP_INTERNAL_TO_INPUT_VNAME_MAPPING: default_varname_mappings.vname_map_CRCM5,
            DataManager.SP_LEVEL_MAPPING: default_varname_mappings.vname_to_level_map,
            DataManager.SP_VARNAME_TO_FILENAME_PREFIX_MAPPING: default_varname_mappings.vname_to_fname_prefix_CRCM5
        }
    )

    # PR90
    dm.compute_climatological_quantiles(start_year=1980, end_year=1998, daily_agg_func=np.mean,
                                        rolling_mean_window_days=29, q=0.9,
                                        varname_internal=default_varname_mappings.TOTAL_PREC)

    # TX10
    dm.compute_climatological_quantiles(start_year=1980, end_year=1998, daily_agg_func=np.max,
                                        rolling_mean_window_days=5, q=0.1,
                                        varname_internal=default_varname_mappings.T_AIR_2M)

    # TN90
    dm.compute_climatological_quantiles(start_year=1980, end_year=1998, daily_agg_func=np.min,
                                        rolling_mean_window_days=5, q=0.9,
                                        varname_internal=default_varname_mappings.T_AIR_2M)


if __name__ == '__main__':
    test_daily_clim_quantile_calculations()
    pass
