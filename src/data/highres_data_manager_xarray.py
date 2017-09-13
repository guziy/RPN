from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import dask.array as da
import xarray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from memory_profiler import profile

from util.agg_blocks import agg_blocks


class HighResDataManagerXarray(object):
    def __init__(self, path="", vname=""):
        self.ds = xarray.open_mfdataset(path)

        print(self.ds)

        self.data = self.ds[vname]
        self.vname = vname


        # Create the caching directory for a variable
        self.cache_dir = Path("Daymet_cache") / vname
        self.cache_dir.mkdir(parents=True, exist_ok=True)









    @profile
    def calculate_daily_percentile_for_period(self, start_year=-np.Inf, end_year=np.Inf, percentile=0.5,
                                              rolling_window_days=None, data_arr: xarray.DataArray=None,
                                              max_space_slice_size=100):

        """
        (assumes daily data as input)
        :param start_year:
        :param end_year:
        :param percentile:
        :param rolling_window_days: if not None a rolling mean is applied before calculation of the percentile
        :param data_arr: default is None, then the data is used from the default dataset attached to the object, sometimes
        rawer data could be passed to do the percentile calculations

        Saves the data to a netcdf file, because does not fit to memory
        """

        if data_arr is not None:
            data = data_arr
        else:
            data = self.data


        print(data["time.year"].max())


        assert isinstance(data, xarray.DataArray)

        nt, ny, nx = data.shape

        # create output DataArray
        d0 = datetime(2001, 1, 1)
        dt = timedelta(days=1)
        out_time = [d0 + i * dt for i in range(365)]
        out_arr = None


        in_times = data.time[:]


        print(in_times[0], in_times[1], "...", in_times[-1])
        print(data.isel(y=0, x=0).values)


        # split i-axis and j-axis into chunks
        jslice_list = get_chunked_slices_for_axis(ny, max_slice_size=max_space_slice_size)
        islice_list = get_chunked_slices_for_axis(nx, max_slice_size=max_space_slice_size)



        for slice_num, islice in enumerate(islice_list):
            print("islice = {}".format(islice))

            for jslice in jslice_list:

                out_file = self.cache_dir / "j_{}-{}__i_{}-{}__period_{}-{}__percentile_{}__rolling_window_days_{}.nc".format(
                    jslice.start, jslice.stop - 1, islice.start, islice.stop - 1, start_year, end_year, percentile, rolling_window_days)

                # Skip files that are already calculated
                if out_file.exists():
                    continue


                assert isinstance(jslice, slice)

                out_block_shape = (len(out_time), jslice.stop - jslice.start, islice.stop - islice.start)

                # print("out_block_shape={}".format(out_block_shape))

                # reuse the out_array for memory recycling
                if out_arr is None or out_arr.shape != out_block_shape:
                    out_arr = xarray.DataArray(np.zeros(out_block_shape))


                block = data.isel(y=jslice, x=islice).values  # since the data layout is (t, y, x)


                # print("data.shape={}".format(data.shape))
                # print("block.shape={}".format(block.shape))


                if not np.all(np.isnan(block)):

                    print("block range: {} .... {}".format(block[~np.isnan(block)].min(), block[~np.isnan(block)].max()))

                    df = pd.DataFrame(data=block.reshape(len(in_times), -1), index=in_times)
                    df = df.select(lambda d: not (d.month == 2 and d.day == 29) and (start_year <= d.year <= end_year))


                    assert isinstance(df, pd.DataFrame)

                    if rolling_window_days is not None:
                        df = df.rolling(rolling_window_days, min_periods=rolling_window_days).mean().bfill()


                    # calculate the percentiles for each day of year
                    df = df.groupby([df.index.month, df.index.day]).quantile(q=percentile)

                    assert df.values.shape[0] == out_block_shape[0]
                    assert df.values.shape[1] == out_block_shape[1] * out_block_shape[2]

                    out_arr.values = df.values.reshape(out_block_shape)


                else:
                    out_arr.values[:] = np.nan

                out_arr.coords["time"] = out_time
                out_arr.to_netcdf(str(out_file))



    def collect_file_parts(self, start_year=-np.Inf, end_year=np.Inf, percentile=0.5, rolling_window_days=None):
        """
        input file format: self.cache_dir / "j_{}-{}__i_{}-{}__period_{}-{}__percentile_{}__rolling_window_days_{}.nc"
        """


        common_part = "period_{}-{}__percentile_{}__rolling_window_days_{}.nc".format(start_year, end_year, percentile, rolling_window_days)


        out_file = "full__" + common_part
        out_file = self.cache_dir / out_file

        if out_file.exists():
            print("{} already exists, will not redo the collection of tiles".format(out_file))
            return


        jlimits_to_ilimits_to_path = defaultdict(dict)

        def __get_index_limits_from_path(fpath, prefix="i"):
            fields = fpath.name.split("__")

            for field in fields:
                if field.startswith(prefix):
                    limits = tuple(int(s) for s in field.split("_")[-1].split("-"))
                    return limits

            raise Exception("Could not find the prefix={} in the name of {}".format(prefix, fpath))


        for tile_file in self.cache_dir.iterdir():

            if not tile_file.name.endswith(common_part):
                continue

            ilimits = __get_index_limits_from_path(tile_file, prefix="i")
            jlimits = __get_index_limits_from_path(tile_file, prefix="j")

            jlimits_to_ilimits_to_path[jlimits][ilimits] = tile_file


        # join first cloumns into rows and then join the rows to a matrix
        rows = []
        for jlimits in sorted(jlimits_to_ilimits_to_path, key=lambda pair: pair[0]):
            ilimits_to_path = jlimits_to_ilimits_to_path[jlimits]
            cols = []
            for ilimits in sorted(ilimits_to_path, key=lambda pair: pair[0]):
                the_path = ilimits_to_path[ilimits]
                cols.append(xarray.open_dataset(str(the_path)))

            rows.append(xarray.concat(cols, dim="dim_2"))

        ds_out = xarray.concat(rows, dim="dim_1")
        assert isinstance(ds_out, xarray.Dataset)

        # rename the variable
        for old_name in ds_out:

            if "time" in old_name.lower():
                continue

            ds_out.rename({old_name: self.vname}, inplace=True)
            ds_out.rename({"dim_0": "time", "dim_1": "y", "dim_2": "x"}, inplace=True)
            break

        # add spatial coordinates
        for in_name in self.ds:
            if in_name.lower() in ["x", "y", "lon", "lat"]:
                ds_out[in_name] = self.ds[in_name]


        # save data to the disk
        ds_out.to_netcdf(str(out_file))



    def close(self):
        self.ds.close()


def get_chunked_slices_for_axis(size, max_slice_size=100):
    """
    Divide the axis to the chunks of afordable size
    :param size:
    :param max_slice_size:
    """
    jslice_list = []


    for left in range(0, size, max_slice_size):
        if left + max_slice_size > size:
            jslice_list.append(slice(left, size))
        else:
            jslice_list.append(slice(left, left + max_slice_size))

    return jslice_list




def spatial_aggregate_daymet_data(source_dir, dest_dir, block_shape: tuple=(10, 10), vname="prcp",
                                  filename_prefix=None):
    """
    :param dest_dir: destination directory to save the results
    :param block_shape: the shape of blocks to which the input arrays will be subdivided
    Assume that each file contains the following variables: x, y, lons, lats, <var>const_vname</var>, time

    Note:
    """

    out_dir = Path(dest_dir) / "daymet_spatial_agg_{}_{}x{}".format(vname, block_shape[0], block_shape[1])

    out_dir.mkdir(parents=True, exist_ok=True)


    assert out_dir != Path(source_dir), "source directory should be different from the destination directory: {}".format(out_dir)


    if filename_prefix is None:
        filename_prefix = "daymet_v3_{}".format(vname)


    file_list = [f for f in source_dir.iterdir() if f.name.startswith(filename_prefix)]

    # static data
    ds_base = xarray.open_dataset(str(file_list[0]))


    const_varnames = ["x", "y", "lon", "lat", "lambert_conformal_conic"]
    const_vars = {k: ds_base[k] for k in const_varnames}

    from skimage.measure import block_reduce

    # aggregate if required
    for const_vname in const_vars:
        if const_vname == "lambert_conformal_conic":
            continue


        if const_vname == "x":
            data = block_reduce(const_vars[const_vname].values, block_size=(block_shape[1],), func=np.mean)
        elif const_vname == "y":
            data = block_reduce(const_vars[const_vname].values, block_size=(block_shape[0],), func=np.mean)
        else:
            data = block_reduce(const_vars[const_vname].values, block_size=block_shape, func=np.mean)

        darr = xarray.DataArray(data, name=const_vname)
        darr.assign_attrs(const_vars[const_vname].attrs)
        const_vars[const_vname] = darr



    print(ds_base[vname].attrs)


    def __agg_masked(data):
        print(type(data), data.shape)
        if not np.all(data.mask):
            return data[~data.mask].mean()[None, None]
        else:
            return np.ma.masked_all((1,))[None, None]


    for f in file_list:
        ds_in = xarray.open_dataset(str(f))

        f_out = dest_dir / f.name

        print("{} => {}".format(f, f_out))

        if f_out.exists():
            print("{} exists, won't redo!".format(f_out))
            continue

        ds_out = xarray.Dataset()
        ds_out["time"] = ds_in["time"]
        ds_out["time_bnds"] = ds_in["time_bnds"]

        arr_in = ds_in[vname]

        all_data = []

        for ti in range(arr_in.shape[0]):

            assert isinstance(arr_in, xarray.DataArray)

            masked_arr_in = arr_in[ti, :, :].to_masked_array()



            all_data.append(agg_blocks())

        arr_out = xarray.DataArray(all_data, name=vname)

        arr_out.assign_attrs(arr_in.attrs)

        coords = dict({const_vars}).update({"time": ds_in["time"]})
        arr_out.assign_coords(coords)

        ds_out[vname] = arr_out

        # save the output data to disk
        ds_out.assign_attrs(ds_in.attrs)
        ds_out.to_netcdf(str(f), unlimited_dims=("time"))

        ds_in.close()

        if True:
            return

    ds_base.close()








# some tests

def test():
    manager = HighResDataManagerXarray(
        path="/snow3/huziy/Daymet_daily/daymet_v3_prcp_*_na.nc4",
        vname="prcp")

    manager.calculate_daily_percentile_for_period(rolling_window_days=None, start_year=1980, end_year=2016, max_space_slice_size=500)
    manager.close()


def test_collect_file_parts():
    manager = HighResDataManagerXarray(
        path="/snow3/huziy/Daymet_daily/daymet_v3_prcp_*_na.nc4",
        vname="prcp")

    manager.collect_file_parts(rolling_window_days=None, percentile=0.5, start_year=1980, end_year=2016)
    manager.close()


def test_spatial_aggregate():
    spatial_aggregate_daymet_data(
        source_dir=Path("/snow3/huziy/Daymet_daily"), dest_dir=Path("/snow3/huziy/Daymet_daily_derivatives"), block_shape=(10, 10),
        vname="prcp"
    )



if __name__ == '__main__':
    # test()
    # test_collect_file_parts()
    test_spatial_aggregate()