from pathlib import Path


import xarray
from scipy.spatial import KDTree

from util.geo import lat_lon


def main(in_file: Path, target_grid_file: Path, out_dir: Path=None):

    with xarray.open_dataset(target_grid_file) as ds_grid:
        lons, lats = ds_grid["lon"][:].values, ds_grid["lat"][:].values

        xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons.flatten(), lats.flatten())

        with xarray.open_dataset(in_file) as ds_in:

            lons_s, lats_s = ds_in["lon"][:].values, ds_in["lat"][:].values
            xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons_s.flatten(), lats_s.flatten())

            ktree = KDTree(list(zip(xs, ys, zs)))
            dists, inds = ktree.query(list(zip(xt, yt, zt)), k=1)


            if out_dir is not None:
                out_dir.mkdir(exist_ok=True)
                out_file = out_dir / (in_file.name + "_interpolated")
            else:
                out_file = in_file.parent / (in_file.name + "_interpolated")

            if out_file.exists():
                print(f"Skipping {in_file}, output already exists ({out_file})")
                return 


            # resample to daily
            ds_in_r = ds_in.resample(t="1D", keep_attrs=True).mean()


            ds_out = xarray.Dataset()
            for vname, var in ds_grid.variables.items():
                ds_out[vname] = var[:]

            ds_out["t"] = ds_in_r["t"][:]


            for vname, var in ds_in_r.variables.items():

                assert isinstance(var, xarray.Variable)

                var = var.squeeze()

                # only interested in (t, x, y) fields
                if var.ndim != 3:
                    print(f"skipping {vname}")
                    continue

                if vname.lower() not in ["t", "time", "lon", "lat"]:
                    print(f"Processing {vname}")
                    var_interpolated = [var[ti].values.flatten()[inds].reshape(lons.shape) for ti in range(var.shape[0])]
                    ds_out[vname] = xarray.DataArray(
                        var_interpolated, dims=("t", "x", "y"),
                        attrs=var.attrs,
                    )

            ds_out.to_netcdf(out_file)



def entry_gl011_canesm2_current():
    out_dir = Path("/scratch/huziy/NEI/GL_samples_only/GL_CC_CanESM2_RCP85/for_hles_analysis")

    grid_file = out_dir / "lon_lat.nc"
    in_dir =  out_dir.parent / "coupled-GL-current_CanESM2/Netcdf_exports_CanESM2_GL_1989-2010"

    for f in in_dir.iterdir():
        main(f, grid_file, out_dir)



def entry_gl011_canesm2_future():
    out_dir = Path("/scratch/huziy/NEI/GL_samples_only/GL_CC_CanESM2_RCP85/for_hles_analysis")

    grid_file = out_dir / "lon_lat.nc"
    in_dir = out_dir.parent / "coupled-GL-future_CanESM2/Netcdf_exports_CanESM2_GL_2079-2100"

    for f in in_dir.iterdir():
        main(f, grid_file, out_dir)


def entry_gl011_canesm2_current_fix():
    out_dir = Path("/scratch/huziy/NEI/HLES/GL_CC_CanESM2_RCP85/for_hles_analysis")

    grid_file = out_dir / "lon_lat.nc"
    in_dir = Path("/scratch/huziy/Output/GL_CC_CanESM2_RCP85/coupled-GL-current_CanESM2/Netcdf_exports_CanESM2_GL_1989-2010")

    for f in in_dir.iterdir():
        main(f, grid_file, out_dir)



def entry_gl011_canesm2_future_fix():
    out_dir = Path("/scratch/huziy/NEI/HLES/GL_CC_CanESM2_RCP85/for_hles_analysis")

    grid_file = out_dir / "lon_lat.nc"
    in_dir = Path("/scratch/huziy/Output/GL_CC_CanESM2_RCP85/coupled-GL-future_CanESM2/Netcdf_exports_CanESM2_GL_2079-2100")

    for f in in_dir.iterdir():
        main(f, grid_file, out_dir)


if __name__ == '__main__':
    # entry_gl011_canesm2_current()
    # entry_gl011_canesm2_future()

    entry_gl011_canesm2_current_fix()
    entry_gl011_canesm2_future_fix()