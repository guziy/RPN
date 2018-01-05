from pathlib import Path

import xarray
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np



def test_input_tmax_and_tmin():
    p_min = "/snow3/huziy/Daymet_daily/daymet_v3_tmin_1988_na.nc4"
    p_max = "/snow3/huziy/Daymet_daily/daymet_v3_tmax_1988_na.nc4"


    tmx = xarray.open_dataset(p_max)["tmax"][0, :, :]
    tmn = xarray.open_dataset(p_min)["tmin"][0, :, :]



    tmn_msk = tmn.to_masked_array()
    tmx_msk = tmx.to_masked_array()


    diff = tmx_msk - tmn_msk


    print(diff.shape)

    plt.figure()

    suspect = (diff < 0.) & ~np.isnan(diff) & ~diff.mask
    if np.any(suspect):
        print(diff[suspect], len(suspect), diff[suspect].min(), "...", diff[suspect].max())
        # print(tmx_msk[suspect])
        # print(tavg_msk[suspect])
        # print(tmn_msk[suspect])
        diff[suspect] = -100


    im = plt.pcolormesh(tmn.x, tmn.y, diff, cmap=cm.get_cmap("coolwarm", 20), vmin=-2, vmax=2)
    plt.colorbar(im)
    plt.show()



def test_input_tavg_and_tmin():
    p_min = "/snow3/huziy/Daymet_daily/daymet_v3_tmin_1988_na.nc4"
    p_avg = "/snow3/huziy/Daymet_daily/daymet_v3_tavg_1988_na.nc4"


    tavg = xarray.open_dataset(p_avg)["tavg"][0, :, :]
    tmn = xarray.open_dataset(p_min)["tmin"][0, :, :]



    tmn_msk = tmn.to_masked_array()
    tavg_msk = tavg.to_masked_array()


    diff = tavg_msk - tmn_msk


    print(diff.shape)

    plt.figure()

    suspect = (diff < 0.) & ~np.isnan(diff) & ~diff.mask
    if np.any(suspect):
        print(diff[suspect], len(suspect), diff[suspect].min(), "...", diff[suspect].max())
        # print(tmx_msk[suspect])
        # print(tavg_msk[suspect])
        # print(tmn_msk[suspect])
        diff[suspect] = -100


    im = plt.pcolormesh(tmn.x, tmn.y, diff, cmap=cm.get_cmap("coolwarm", 20), vmin=-2, vmax=2)
    plt.colorbar(im)
    plt.show()



def main():
    p_max = "/snow3/huziy/Daymet_daily_derivatives/daymet_spatial_agg_tmax_10x10/daymet_v3_tmax_1988_na.nc4"
    p_min = "/snow3/huziy/Daymet_daily_derivatives/daymet_spatial_agg_tmin_10x10/daymet_v3_tmin_1988_na.nc4"
    p_avg = "/snow3/huziy/Daymet_daily_derivatives/daymet_spatial_agg_tavg_10x10/daymet_v3_tavg_1988_na.nc4"


    tmx = xarray.open_dataset(p_max)["tmax"]
    tavg = xarray.open_dataset(p_avg)["tavg"]
    tmn = xarray.open_dataset(p_min)["tmin"]

    print(tmx.shape)
    print(tavg.shape)


    tmx_msk = tmx.to_masked_array()
    tmn_msk = tmn.to_masked_array()

    tavg_msk = tavg.to_masked_array()


    diff = tavg_msk[100, :, :] - tmn_msk[100, :, :]


    print(diff.shape)

    plt.figure()

    suspect = (diff < 0.) & ~np.isnan(diff) & ~diff.mask
    if np.any(suspect):
        print(diff[suspect], len(suspect), diff[suspect].min(), "...", diff[suspect].max())
        # print(tmx_msk[suspect])
        # print(tavg_msk[suspect])
        # print(tmn_msk[suspect])
        diff[suspect] = -100


    im = plt.pcolormesh(tmx.x, tmx.y, diff[:, :], cmap=cm.get_cmap("coolwarm", 20), vmin=-2, vmax=2)
    plt.colorbar(im)
    plt.show()




if __name__ == '__main__':
    # main()
    # test_input_tavg_and_tmin()
    test_input_tmax_and_tmin()