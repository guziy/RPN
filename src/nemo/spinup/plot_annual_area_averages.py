from pathlib import Path

import xarray

import matplotlib.pyplot as plt
import numpy as np


from application_properties import main_decorator


@main_decorator
def main():
    path = "/HOME/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3_spinup/EXP00/spinup_outputs"
    # path = "test_xarr_links"

    path_folder = Path(path)

    vname = "votemper"
    suffix = "grid_T.nc"

    with xarray.open_mfdataset(str(path_folder.joinpath("GLK_1d_*{}".format(suffix)))) as ds:

        gl_mask = ~(ds[vname].max(axis=0) <= 0).squeeze().values

        v_annual_mean = ds[vname].groupby("time_counter.year").mean(axis=0)

        # print(gl_mask)


        annual_and_spatial_avg = (v_annual_mean * gl_mask).sum(axis=range(1, v_annual_mean.ndim)) / gl_mask.sum()

        years_list = list(sorted(annual_and_spatial_avg.coords["year"].values))
        print("years_list = ", years_list)

        # np.reshape is to set the correct shape of the one element array
        annual_means_list = [np.reshape(annual_and_spatial_avg.sel(year=y).values, 1)[0] for y in years_list]
        print("annual_means_list = ", annual_means_list)


    assert isinstance(annual_and_spatial_avg, xarray.DataArray)


    # plot
    plt.figure()
    plt.plot(years_list, annual_means_list, "r-")

    plt.savefig("spinup_{}.png".format(vname))
    plt.show()


if __name__ == '__main__':
    main()