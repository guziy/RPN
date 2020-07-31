import shutil
import sys
from pathlib import Path

import dask
import dask.array as darr
import numpy as np

import xarray
from pyresample.kd_tree import KDTree

from util.geo import lat_lon

# TODO: fix the interpolation so that the value at the end of a period,
# i.e. at 00:00 of the start of the period are not considered as a value for that day.


def main(in_file: Path, target_grid_file: Path, out_dir: Path = None):

    # handle few edge cases
    if in_file.is_dir():
        sys.stderr.write(f"WARNING: {in_file} is a directory, skipping!")
        return

    if in_file.name == target_grid_file.name:
        sys.stderr.write(f"INFO: {in_file} is already on the destination grid, skipping!")
        return

    if out_dir is not None:
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / (in_file.name + "_interpolated")
    else:
        out_file = in_file.parent / (in_file.name + "_interpolated")

    if out_file.exists():
        print(f"Skipping, output already exists ({out_file})")
        return

    # temporary dir
    tmp_dir = out_dir / f"tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir()
    print(f"temporary files will be saved to: {tmp_dir}")

    sys.stdout.write(f"interpolating {in_file} => {out_file}\n")

    with xarray.open_dataset(target_grid_file) as ds_grid:
        lons, lats = ds_grid["lon"][:].values, ds_grid["lat"][:].values

        xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons.flatten(), lats.flatten())

        with xarray.open_dataset(in_file) as ds_in:

            lons_s, lats_s = ds_in["lon"][:].values, ds_in["lat"][:].values
            xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons_s.flatten(), lats_s.flatten())

            ktree = KDTree(np.array(list(zip(xs, ys, zs))))
            dists, inds = ktree.query(np.array(list(zip(xt, yt, zt))), k=1)

            # resample to daily TODO: do it after the interpolation in space
            # ds_in_r = ds_in.resample(t="1D", keep_attrs=True).mean(dim="t")
            # ds_out = ds_in.interp(coords=ds_grid.coords, method="nearest")

            i_mat, j_mat = None, None

            for vname, var in ds_in.items():

                assert isinstance(var, xarray.DataArray)

                var = var.squeeze()

                print(var.shape)

                # only interested in (t, x, y) fields
                if var.ndim != 3:
                    print(f"skipping {vname}")
                    continue

                print(var.attrs)
                target_shape = lons.shape
                print(f"target_shape = {target_shape}")

                # find out i_arr and j_arr, that should speed up the interpolation
                if i_mat is None:
                    i_mat, j_mat = np.indices(var.shape[1:])
                    i_mat = i_mat.flatten()[inds].reshape(target_shape)
                    j_mat = j_mat.flatten()[inds].reshape(target_shape)

                if vname.lower() not in ["t", "time", "lon", "lat"]:
                    print(f"Processing {vname}")
                    # make it in chunks !!

                    t_chunk_size = 1000
                    for ti_left in range(0, var.shape[0], t_chunk_size):

                        ti_right = min(ti_left + t_chunk_size, var.shape[0])

                        var_interpolated = var[ti_left:ti_right].values[:, i_mat, j_mat]

                        tmp_file = tmp_dir / f"{ti_left:08d}_interpolated.nc"

                        dso_tmp = xarray.Dataset()
                        dso_tmp[vname] = xarray.DataArray(
                            data=var_interpolated, dims=("t", "x", "y"),
                            attrs=var.attrs,
                        )
                        dso_tmp["t"] = ds_in["t"][ti_left:ti_right]

                        mode = "a" if tmp_file.exists() else "w"
                        dso_tmp.to_netcdf(tmp_file, mode=mode)

                        print(f"Saving {tmp_file}")

            # resample to daily and save to disk
            # ds_out.resample(t="1D", keep_attrs=True).mean(dim="t").to_netcdf(out_file)
            print(f"Saving {out_file}")

            with xarray.open_mfdataset(str(tmp_dir / f"*_interpolated.nc"),
                                       concat_dim="t", data_vars="minimal") as ds_out:
                ds_out.resample(t="1D", keep_attrs=True).mean(dim="t").to_netcdf(out_file)
                ds_grid["lon"].to_netcdf(out_file, mode="a")
                ds_grid["lat"].to_netcdf(out_file, mode="a")

            # clean up the temp dir
            shutil.rmtree(tmp_dir)


def entry_gl011_canesm2_current():
    out_dir = Path("/scratch/huziy/NEI/GL_samples_only/GL_CC_CanESM2_RCP85/for_hles_analysis")

    grid_file = out_dir / "lon_lat.nc"
    in_dir = out_dir.parent / "coupled-GL-current_CanESM2/Netcdf_exports_CanESM2_GL_1989-2010"

    for f in in_dir.iterdir():
        main(f, grid_file, out_dir)


def entry_gl011_canesm2_future():
    out_dir = Path("/scratch/huziy/NEI/GL_samples_only/GL_CC_CanESM2_RCP85/for_hles_analysis")

    grid_file = out_dir / "lon_lat.nc"
    in_dir = out_dir.parent / "coupled-GL-future_CanESM2/Netcdf_exports_CanESM2_GL_2079-2100"

    for f in in_dir.iterdir():
        main(f, grid_file, out_dir)


def entry_gl011_canesm2_current_fix():
    """
    The output directory shoul contain the grid file with only the target grid
    """
    out_dir = Path("/home/huziy/data/big1/Projects/HLES_GL_NEMO_CRCM5_CC/cc_coupled-GL_CanESM2/cur/hles")

    grid_file = next(out_dir.glob("*.nc"))
    in_dir = Path("/home/huziy/data/big1/Projects/HLES_GL_NEMO_CRCM5_CC/cc_coupled-GL_CanESM2/cur")

    for f in in_dir.glob("*.nc"):
        main(f, grid_file, out_dir)


def entry_gl011_canesm2_future_fix():
    out_dir = Path("/home/huziy/data/big1/Projects/HLES_GL_NEMO_CRCM5_CC/cc_coupled-GL_CanESM2/fut/hles")

    grid_file = next(out_dir.glob("*.nc"))
    in_dir = Path("/home/huziy/data/big1/Projects/HLES_GL_NEMO_CRCM5_CC/cc_coupled-GL_CanESM2/fut")

    for f in in_dir.glob("*.nc"):
        main(f, grid_file, out_dir)


def entry_gl011_canesm2_mbair():
    grid_file = Path("/Users/huziy/Projects/lon_lat.nc")

    files = [
        Path("/Users/huziy/Projects/lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_2079-2100_2079-2100/merged/CanESM2_GL_2079-2100_SN_2079-2100.nc"),
        Path("/Users/huziy/Projects/lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_1989-2010_1989-2010/merged/CanESM2_GL_1989-2010_SN_1989-2010.nc")
    ]

    for f in files:
        main(f, grid_file)


if __name__ == '__main__':
    # entry_gl011_canesm2_current()
    # entry_gl011_canesm2_future()

    entry_gl011_canesm2_current_fix()
    entry_gl011_canesm2_future_fix()

    # entry_gl011_canesm2_mbair()
