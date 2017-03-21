import xarray as xr

import matplotlib.pyplot as plt

def main():
    path_to_glerl_ice_fractions = "/RESCUE/skynet3_rech1/huziy/obs_data_for_HLES/initial_data/glerl_icecov1.nc"

    with xr.open_dataset(path_to_glerl_ice_fractions) as ds:
        ice_cover = ds["ice_cover"][:]
        print(ice_cover)

        for i, t in enumerate(ice_cover.time[:]):
            print(t)
            field = ice_cover.sel(time=t)

            assert isinstance(field, xr.DataArray)





            if i > 20:
                plt.figure()
                im = plt.pcolormesh(field.T)
                plt.colorbar(im)
                plt.show()
                raise Exception





if __name__ == '__main__':
    main()