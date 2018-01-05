from pathlib import Path

import xarray
from rpn.domains import lat_lon
from scipy.spatial import cKDTree as KDTree

from domains import grid_config

import numpy as np

from lake_effect_snow.default_varname_mappings import mps_per_knot


def get_ktree(ds: xarray.Dataset):

    lonv = ds["lon"]

    if lonv.ndim == 3:
        lon, lat = ds["lon"][0].values, ds["lat"][0].values
    else:
        lon, lat = ds["lon"].values, ds["lat"].values

    x, y, z = lat_lon.lon_lat_to_cartesian(lon.flatten(), lat.flatten())

    return KDTree(
        list(zip(x, y, z))
    )


# This is the nearest neighbor interpolation so the grids should be of the same size
def main(convert_mps_to_knots=True):



    # target grid for interpolation
    nml_path = "/HOME/huziy/skynet3_rech1/obs_data_for_HLES/interploated_to_the_same_grid/obs_anuspmaurer_narr/gemclim_settings.nml"
    target_grid_config = grid_config.gridconfig_from_gemclim_settings_file(nml_path)
    print(target_grid_config)

    target_lons, target_lats = target_grid_config.get_lons_and_lats_of_gridpoint_centers()
    xt, yt, zt = lat_lon.lon_lat_to_cartesian(target_lons.flatten(), target_lats.flatten())

    # the output folder
    out_folder = Path(nml_path).parent


    # Source data for wind: NARR
    data_sources = {
        "UU": "/RESCUE/skynet3_rech1/huziy/obs_data_for_HLES/initial_data/narr/uwnd.10m.*.nc",
        "VV": "/RESCUE/skynet3_rech1/huziy/obs_data_for_HLES/initial_data/narr/vwnd.10m.*.nc"
    }

    vname_map = {
        "UU": "uwnd", "VV": "vwnd"
    }

    chunk_size = 1000

    for vname, data_path in data_sources.items():
        with xarray.open_mfdataset(data_path) as ds:
            vname_daymet = vname_map[vname]
            arr = ds[vname_daymet]

            t = ds["time"]

            ktree = get_ktree(ds)

            d, sp_inds = ktree.query(list(zip(xt, yt, zt)), k=1)

            data_out = []

            nt = len(t)
            for start_index in range(0, nt, chunk_size):
                end_index = min(start_index + chunk_size - 1, nt - 1)
                chunk = end_index - start_index + 1

                arr_sel = arr[start_index:end_index + 1, :, :].to_masked_array()
                print(arr_sel.shape)

                data = arr_sel.reshape((chunk, -1))[:, sp_inds].reshape((chunk, ) + target_lons.shape)
                data_out.append(data)

            # ---
            data_out = np.concatenate(data_out, axis=0)

            if convert_mps_to_knots:
                data_out *= mps_per_knot



            ds_out = xarray.Dataset(
                data_vars={
                    vname: (["time", "x", "y"], data_out),
                },
                coords={"time": ("time", t.values),
                        "lon": (["x", "y"], target_lons),
                        "lat": (["x", "y"], target_lats),
                },
            )
            ds_out[vname].attrs.units = "knots"


            ds_out.to_netcdf(str(out_folder / f"{vname}.nc"))


if __name__ == '__main__':
    main()