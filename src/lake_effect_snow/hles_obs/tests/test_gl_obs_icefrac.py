import xarray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    source_nc_path = "/HOME/huziy/skynet3_rech1/obs_data_for_HLES/interploated_to_the_same_grid/GL_0.1_452x260_icefix/cis_nic_glerl_interpolated_lc_fix.nc"

    ice_fr = xarray.open_dataset(source_nc_path)["LC"]

    assert isinstance(ice_fr, xarray.DataArray)
    ice_fr = ice_fr.where((ice_fr >= 0) & (ice_fr <= 1))

    # t, x, y
    source_data = ice_fr.to_masked_array(copy=False)
    source_time = ice_fr.coords["time"]
    source_time = pd.to_datetime(source_time.values.tolist())

    s_source = pd.Series(data=[
        (field[~field.mask].mean() if not np.all(field.mask) else np.nan) for field in source_data
    ], index=source_time)


    ice_max = s_source.groupby(s_source.index.year).max()

    ice_max.plot(marker=".")

    plt.grid(True)

    plt.show()



if __name__ == '__main__':
    main()