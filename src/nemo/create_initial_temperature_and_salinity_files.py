from netCDF4 import Dataset
import os

__author__ = 'huziy'

import numpy as np
#


def create_file_with_field(folder="", fname="", var_name="", data=None):
    fpath = os.path.join(folder, fname)
    ds = Dataset(fpath, "w", format="NETCDF3_CLASSIC")
    nz, ny, nx = data.shape
    ds.createDimension("x", nx)
    ds.createDimension("y", ny)
    ds.createDimension("z", nz)

    var = ds.createVariable(var_name, "f4", dimensions=("z", "y", "x"))
    var[:] = data
    ds.close()


def main():
    import nemo.generate_grid.nemo_domain_properties as domain_props
    dest_folder = "/home/huziy/skynet3_rech1/NEMO_fields_to_interpolated_210x130_0.1deg"
    t_file_name = "IC_T.nc"
    t_var_name = "votemper"

    s_file_name = "IC_S.nc"
    s_var_name = "vosaline"

    # the_shape = 35, 455, 355
    the_shape = 35, domain_props.ny, domain_props.nx
    initial_temperature = 4.0 * np.ones(the_shape)
    initial_salinity = 0.0 * np.ones(the_shape)

    create_file_with_field(folder=dest_folder, fname=t_file_name, var_name=t_var_name, data=initial_temperature)
    create_file_with_field(folder=dest_folder, fname=s_file_name, var_name=s_var_name, data=initial_salinity)


def main_orca12():
    t_file_name = "IC_T.nc"
    t_var_name = "votemper"

    s_file_name = "IC_S.nc"
    s_var_name = "vosaline"

    dest_folder = "/home/olh001/data/ppp2-sitestore/prepare_ic_orca12"
    nx, ny, nz = 4322, 3059, 1
    the_shape = nz, ny, nx
    initial_temperature = (4.0 + 273.15) * np.ones(the_shape)
    initial_salinity = 0.0 * np.ones(the_shape)

    create_file_with_field(folder=dest_folder, fname=t_file_name, var_name=t_var_name, data=initial_temperature)
    create_file_with_field(folder=dest_folder, fname=s_file_name, var_name=s_var_name, data=initial_salinity)


if __name__ == "__main__":
    main_orca12()
