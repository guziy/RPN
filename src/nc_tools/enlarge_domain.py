
# add a few gridpoints from either side of the domain in a netcdf file (no coordinate handling whatsoever)
import netCDF4
from netCDF4 import Dataset

from multiprocessing import Pool

from pathlib import Path
import itertools as itt


# Extension params

def process_files(nprocs=10, in_folder="", out_folder="", ill=0, jll=0, ni=-1, nj=-1, i_dim_name="lon", j_dim_name="lat", time_dim_name="time"):
    ppool = Pool(processes=nprocs)

    in_folder_p = Path(in_folder)
    out_folder_p = Path(out_folder)

    in_files = []
    out_files = []


    for in_file in in_folder_p.iterdir():

        if not in_file.name.endswith("nc"):
            continue

        in_files.append(str(in_file))
        out_files.append(str(out_folder_p.joinpath(in_file.name)))


    # create the arguments for each process
    args = zip(itt.repeat(ill), itt.repeat(jll), itt.repeat(ni), itt.repeat(nj), in_files, out_files, itt.repeat(i_dim_name), itt.repeat(j_dim_name), itt.repeat(time_dim_name))

    # Do the processing in parallel
    ppool.map(main_wrapper, args)


def main_wrapper(args):
    main(*args)




def main(ill=0, jll=0, ni=-1, nj=-1, infile=None, outfile=None, i_dim_name="lon", j_dim_name="lat", time_dim_name="time"):
    """
    Note indices are 0-based
    Assuming the variable layout X(time, lon, lat)

    :param time_dim_name:
    :param j_dim_name:
    :param i_dim_name:
    :param nj: Number of gridpoints in the vertical direction
    :param ni: Number of gridpoints in the horizontal direction
    :param ill: i-index of the lower left corner of the new grid with respect to the lower left corner of the initial grid
    :param jll: j-index of the lower left corner of the new grid with respect to the lower left corner of the initial grid
    :param infile:
    :param outfile:
    """

    if outfile is None:
        outfile = infile[:-3] + "_enlarged.nc"

    # Check if the file exists to not overwrite by mistake
    outfile_p = Path(outfile)
    if outfile_p.exists():
        print("Skipping {}: already exists".format(outfile_p))
        return

    dims_of_interest = [i_dim_name, j_dim_name, time_dim_name]

    with Dataset(infile) as ds_in:

        with Dataset(outfile, "w") as ds_out:

            # Take care of the dimensions
            for dimname, the_dim in ds_in.dimensions.items():

                if dimname == i_dim_name:
                    dim_size = ni
                elif dimname == j_dim_name:
                    dim_size = nj
                else:
                    dim_size = len(the_dim)

                ds_out.createDimension(dimname, dim_size)

            ni_in, nj_in = [len(ds_in.dimensions[k]) for k in dims_of_interest[:2]]

            if ni < 0:
                ni = ni_in

            if nj < 0:
                nj = nj_in


            # Get the coordinates of the intersection with respect to the initial and extended
            # domain's lower left corner
            i_ll_wrt_in = max(0, ill)
            j_ll_wrt_in = max(0, jll)
            i_ur_wrt_in = min(ni + ill - 1, ni_in - 1)
            j_ur_wrt_in = min(nj + jll - 1, nj_in - 1)

            i_ll_wrt_out = i_ll_wrt_in - ill
            j_ll_wrt_out = j_ll_wrt_in - jll
            i_ur_wrt_out = i_ur_wrt_in - ill
            j_ur_wrt_out = j_ur_wrt_in - jll


            print([i_ll_wrt_in, j_ll_wrt_in, i_ur_wrt_in, j_ur_wrt_in])
            print([i_ll_wrt_out, j_ll_wrt_out, i_ur_wrt_out, j_ur_wrt_out])

            # Take care of the variables
            for vname, v in ds_in.variables.items():
                assert isinstance(v, netCDF4.Variable)

                if set(v.dimensions) == {i_dim_name, j_dim_name, time_dim_name}:
                    v_out = ds_out.createVariable(vname, v.datatype, (time_dim_name, i_dim_name, j_dim_name), fill_value=0)


                    # read the data in memory
                    data_in = v[:]

                    print("Input shape: {}".format(v.shape))
                    print("Output shape: {}".format(v_out.shape))

                    assert isinstance(v_out, netCDF4.Variable)
                    v_out.setncatts({k: v.getncattr(k) for k in v.ncattrs()})

                    v_out[:, i_ll_wrt_out:i_ur_wrt_out + 1, j_ll_wrt_out:j_ur_wrt_out + 1] = data_in[:, i_ll_wrt_in:i_ur_wrt_in + 1, j_ll_wrt_in:j_ur_wrt_in + 1]

                    # test_arr = np.ma.masked_array(data_in[0, :, :].copy())
                    # test_arr[i_ll_wrt_in:i_ur_wrt_in + 1, j_ll_wrt_in:j_ur_wrt_in + 1] = np.ma.masked
                    # plt.figure()
                    # plt.pcolormesh(test_arr.T)
                    # plt.colorbar()
                    # plt.title(v.name)
                    # plt.xlabel(v.dimensions[1])
                    # plt.ylabel(v.dimensions[2])
                    # plt.show()

                    delattr(v_out, "_FillValue")

                elif i_dim_name in v.dimensions and j_dim_name in v.dimensions:
                    # need for clay, sand and depth to bedrock
                    v_out = ds_out.createVariable(vname, v.datatype, v.dimensions, fill_value=0)
                    v_out.setncatts({k: v.getncattr(k) for k in v.ncattrs()})

                    if list(v.dimensions[1:]) == [i_dim_name, j_dim_name]:
                        v_out[:, i_ll_wrt_out:i_ur_wrt_out + 1, j_ll_wrt_out:j_ur_wrt_out + 1] = v[:, i_ll_wrt_in:i_ur_wrt_in + 1, j_ll_wrt_in:j_ur_wrt_in + 1]
                    elif list(v.dimensions[:-1]) == [i_dim_name, j_dim_name]:
                        v_out[i_ll_wrt_out:i_ur_wrt_out + 1, j_ll_wrt_out:j_ur_wrt_out + 1, :] = v[i_ll_wrt_in:i_ur_wrt_in + 1, j_ll_wrt_in:j_ur_wrt_in + 1, :]
                    elif set(v.dimensions) == {i_dim_name, j_dim_name}:
                        v_out[i_ll_wrt_out:i_ur_wrt_out + 1, j_ll_wrt_out:j_ur_wrt_out + 1] = v[i_ll_wrt_in:i_ur_wrt_in + 1, j_ll_wrt_in:j_ur_wrt_in + 1]
                    else:
                        # Handle the case for fields without time dimension
                        print("Not implemented for {} with the following layout {}".format(v.name, v.dimensions))

                    delattr(v_out, "_FillValue")
                else:
                    # Just copy other variables
                    v_out = ds_out.createVariable(vname, v.datatype, v.dimensions)
                    v_out.setncatts({k: v.getncattr(k) for k in v.ncattrs()})
                    v_out[:] = v[:]
                    continue


if __name__ == '__main__':

    pass
    # Cordex 0.11deg
    # process_files(nprocs=10, in_folder="/RESCUE/skynet3_rech1/huziy/CORDEX_ERAI_0.11deg_NA_RUNOFF_NC",
    #               out_folder="/RESCUE/skynet3_rech1/huziy/CORDEX_ERAI_0.11deg_West_Canada_RUNOFF_NC",
    #               ill=12 - 20,
    #               jll=244 - 20,
    #               ni=404,
    #               nj=380,
    #               i_dim_name="lon", j_dim_name="lat")



    # geophysics file
    # main(ill=12 - 20, jll=244 - 20, ni=404, nj=380, i_dim_name="longitude", j_dim_name="latitude",
    #      infile="/RESCUE/skynet3_rech1/huziy/water_route_mh_bc_011deg_wc/sand_clay_depth_to_bedrock_650x640.nc",
    #      outfile="/RESCUE/skynet3_rech1/huziy/water_route_mh_bc_011deg_wc/sand_clay_depth_to_bedrock_404x380.nc")


    # main(12 - 20, 224 - 20, 404, 308, infile="/RESCUE/skynet3_rech1/huziy/CORDEX_ERAI_0.11deg_West_Canada_RUNOFF_NC/test.nc",
    #      outfile=None, i_dim_name="x", j_dim_name="y")

    # main(12 - 20, 224 - 20, 404, 308, infile="/HOME/huziy/skynet3_rech1/CORDEX_ERAI_0.11deg_West_Canada_RUNOFF_NC/NorthAmerica_0.11deg_ERA40-Int0.75_B1_198001.nc",
    #      outfile=None, i_dim_name="lon", j_dim_name="lat")


