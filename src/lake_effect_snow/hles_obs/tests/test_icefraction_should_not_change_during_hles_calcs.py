from datetime import datetime

import xarray
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import date2num
import numpy as np
from matplotlib.gridspec import GridSpec


def test_plot_area_avg(target_nc_folder="", source_nc_path=""):

    # target_nc_folder = "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_daily_Obs_monthly_icefix_1980-2009"
    # target_nc_folder = "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_icefix_Obs_1980-1981_test"

    #target_nc_folder = "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_daily_Obs_monthly_icefix_test2_1980-1981_test1"



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

    ice_fr_lkeff = xarray.open_mfdataset(target_nc_folder + "/*daily.nc")["lake_ice_fraction"]
    lkeff_data = ice_fr_lkeff.to_masked_array(copy=False)
    lkeff_time = pd.to_datetime(ice_fr_lkeff.coords["t"].values.tolist())

    s_lkeff = pd.Series([
        (field[~field.mask].mean() if not np.all(field.mask) else np.nan) for field in lkeff_data
    ], index=lkeff_time)

    s_source = s_source[(s_source.index <= lkeff_time[-1]) & (s_source.index >= lkeff_time[0])]

    assert isinstance(s_source, pd.Series)

    #
    print(f"Source: len={len(s_source)}")
    print(f"Lkeff: len={len(s_lkeff)}")

    # do the plotting
    fig = plt.figure()
    gs = GridSpec(2, 1)
    # plot initial lake fractions
    ax = fig.add_subplot(gs[0, 0])
    s_source.plot(ax=ax, marker=".", linestyle="None", label="original")
    ax.legend()


    # plot lake fractions outputed by hles algorithm
    ax = fig.add_subplot(gs[1, 0], sharex=ax)
    s_lkeff.plot(ax=ax, marker=".", linestyle="None", label="lkeff")

    ax.legend()
    # plt.show()


def __print_field_stats(tfield, field, label):


    good_mask = ~field.mask

    if not np.any(good_mask):
        print(f"{label}: no meaningful data")
        return

    good_data = field[good_mask]
    print(f"{label} {tfield}:\n{good_data.min()}...{good_data.max()}\n"
          f"mean={good_data.mean()}\n"
          f"std={good_data.std()}\n")
    print("-" * 20)


def test_plot_maps(target_nc_folder, source_nc_path=""):

    # target_nc_folder = "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_daily_Obs_monthly_icefix_1980-2009"



    # target_nc_folder = "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_daily_Obs_monthly_icefix_test2_1980-1981_test1"

    ice_fr = xarray.open_dataset(source_nc_path)["LC"]

    assert isinstance(ice_fr, xarray.DataArray)
    ice_fr = ice_fr.where((ice_fr >= 0) & (ice_fr <= 1))


    start_date = datetime(1981, 1, 1)


    # t, x, y
    source_data = ice_fr.to_masked_array(copy=False)
    source_time = ice_fr.coords["time"]
    source_time = pd.to_datetime(source_time.values.tolist())


    ice_fr_lkeff = xarray.open_mfdataset(target_nc_folder + "/*daily.nc")["lake_ice_fraction"]
    lkeff_data = ice_fr_lkeff.to_masked_array(copy=False)
    lkeff_time = pd.to_datetime(ice_fr_lkeff.coords["t"].values.tolist())


    # select from lkeff data
    lkeff_time_sel = []
    lkeff_data_sel = []

    for t, afield in zip(lkeff_time, lkeff_data):
        if t < start_date:
            continue

        lkeff_time_sel.append(t)
        lkeff_data_sel.append(afield)

    lkeff_time = lkeff_time_sel
    lkeff_data = lkeff_data_sel



    # Select from the source time and data
    source_data_sel = []
    source_time_sel = []
    for t, afield in zip(source_time, source_data):

        if lkeff_time[0] <= t <= lkeff_time[-1]:
            source_data_sel.append(afield)
            source_time_sel.append(t)


    gs = GridSpec(1, 2)
    for i in range(len(source_time_sel)):

        ts = source_time_sel[i]
        tl = lkeff_time[i]

        data_s = source_data_sel[i]
        data_l = lkeff_data[i]

        fig = plt.figure(figsize=(20, 10))

        ax = fig.add_subplot(gs[0, 0])
        ax.set_title(f"Source if: {ts}")
        cs = ax.contourf(data_s, np.arange(0, 1.1, 0.1))
        plt.colorbar(cs, ax=ax)

        ax = fig.add_subplot(gs[0, 1])
        ax.set_title(f"Lkeff if: {tl}")
        cs = ax.contourf(data_l, np.arange(0, 1.1, 0.1))
        plt.colorbar(cs, ax=ax)

        print("*" * 20)
        __print_field_stats(ts, data_s, "source")
        __print_field_stats(tl, data_l, "lkeff")
        print("*" * 20)



        ms = data_s[~data_s.mask].mean()
        ml = data_l[~data_l.mask].mean()
        if ms != ml:
            print(f"ms={ms}; ml={ml}")
            plt.show()

        plt.close(fig)






def main():
    target_nc_folder = "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_daily_Obs_monthly_icefix_test2_1proc_1980-1981"
    # source_nc_path = "/HOME/huziy/skynet3_rech1/obs_data_for_HLES/interploated_to_the_same_grid/GL_0.1_452x260/cis_nic_glerl_interpolated_lc.nc"
    source_nc_path = "/HOME/huziy/skynet3_rech1/obs_data_for_HLES/interploated_to_the_same_grid/GL_0.1_452x260_icefix/cis_nic_glerl_interpolated_lc_fix.nc"

    test_plot_area_avg(target_nc_folder=target_nc_folder, source_nc_path=source_nc_path)
    # test_plot_maps(target_nc_folder=target_nc_folder, source_nc_path=source_nc_path)
    plt.show()



if __name__ == '__main__':
    main()