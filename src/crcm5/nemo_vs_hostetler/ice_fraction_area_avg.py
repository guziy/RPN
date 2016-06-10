from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

from matplotlib.dates import date2num, DateFormatter, YearLocator
from pathlib import Path
from rpn import level_kinds
from rpn.rpn_multi import MultiRPN
import os

from application_properties import main_decorator
from crcm5.nemo_vs_hostetler import commons

import pandas as pd

from crcm5.nemo_vs_hostetler.time_height_plots_area_avg import get_nemo_lakes_mask
from data import GL_obs_timeseries
from netCDF4 import Dataset, MFDataset, num2date

from util import plot_utils

img_folder = "nemo_vs_hostetler"

def get_area_avg_timeseries(samples_dir, start_year=-np.Inf, end_year=np.Inf, filename_prefix="pm",
                            level=-1, level_kind=level_kinds.ARBITRARY,
                            varname="", mask=None) -> pd.Series:
    """
    get the timeseries of area averaged ice fraction
    :rtype : pd.Series
    """

    yearly_ts = []

    for y in range(start_year, end_year + 1):
        files_for_year = []

        mfolders = [os.path.join(samples_dir, f) for f in os.listdir(samples_dir) if f[:-2].endswith(str(y))]

        for mfolder in mfolders:
            files_for_year += [os.path.join(mfolder, fn) for fn in os.listdir(mfolder) if
                               fn.startswith(filename_prefix) and fn[-9:-1] != "0" * 8]


        if len(files_for_year) == 0:
            continue

        mrpn = MultiRPN(files_for_year)
        data = mrpn.get_all_time_records_for_name_and_level(varname=varname, level=level, level_kind=level_kind)

        for t, field in data.items():
            data[t] = field[mask].mean()

        tlist = [t for t in data.keys()]
        ser = pd.Series(index=tlist, data=[data[t] for t in tlist])
        ser.sort_index(inplace=True)
        yearly_ts.append(ser)

    return pd.concat(yearly_ts)

@main_decorator
def main():
    start_year = 1979
    end_year = 2000

    HL_LABEL = "CRCM5_HL"
    NEMO_LABEL = "CRCM5_NEMO"

    sim_label_to_path = OrderedDict(
        [(HL_LABEL, "/RESCUE/skynet3_rech1/huziy/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl_oneway/Samples"),
         (NEMO_LABEL, "/HOME/huziy/skynet3_rech1/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl/Samples")]
    )

    var_name_list = ["TT", "PR", "LC", "HR", "HU", "AV", "I5", "AL"]

    season_to_months = commons.season_to_months

    vname_to_level = {
        "TT": 1, "PR": -1, "SN": -1, "LC": -1, "HR": 1, "HU": 1, "AV": -1, "I5": -1, "AL": -1
    }

    vname_to_level_kind = {
        "TT": level_kinds.HYBRID, "PR": level_kinds.ARBITRARY, "SN": level_kinds.ARBITRARY,
        "LC": level_kinds.ARBITRARY, "HR": level_kinds.HYBRID, "HU": level_kinds.HYBRID, "AV": level_kinds.ARBITRARY,
        "I5": level_kinds.ARBITRARY, "AL": level_kinds.ARBITRARY
    }

    vname_to_file_prefix = {
        "TT": "dm",
        "PR": "pm",
        "SN": "pm",
        "LC": "pm",
        "HR": "dm",
        "HU": "dm",
        "AV": "pm",
        "I5": "pm",
        "AL": "pm"
    }

    # ---> ---->
    avg_mask = get_nemo_lakes_mask(samples_dir=sim_label_to_path[NEMO_LABEL])

    vname = "LC"

    common_params = dict(start_year=start_year, end_year=end_year,
                         filename_prefix=vname_to_file_prefix[vname], level=vname_to_level[vname],
                         level_kind=vname_to_level_kind[vname], varname=vname, mask=avg_mask)

    hl_icefrac = get_area_avg_timeseries(sim_label_to_path[HL_LABEL], **common_params)
    nemo_icefrac = get_area_avg_timeseries(sim_label_to_path[NEMO_LABEL], **common_params)

    obs_icefrac = GL_obs_timeseries.get_ts_with_real_dates_from_file(
        path="/RESCUE/skynet3_rech1/huziy/obs_data/Lake_ice_concentration_Great_lakes_timeseries/GLK-30x.TXT",
        start_year=start_year - 1, end_year=end_year - 1)

    obs_icefrac /= 100.0


    plot_utils.apply_plot_params(font_size=14)

    # nemo ice fraction from 1-way coupled
    # with MFDataset("/HOME/huziy/skynet3_rech1/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl_oneway/NEMO/*grid_T.nc") as ds:
    #     timevar = ds.variables["time_counter"]
    #     print(timevar.units)
    #     times = num2date(timevar[:], timevar.units)
    #
    #     print(times[0], times[-1], type(times[0]))
    #
    #     print(times[:10])
    #
    #     print("===== avg_mask ====")
    #     print(avg_mask.min(), avg_mask.max(), avg_mask.shape)
    #
    #     print(ds.variables["soicecov"][0, :, :].shape)
    #     vals = [field.transpose()[20:-20, 20:-20][avg_mask].mean() for field in ds.variables["soicecov"][:]]
    #
    #     assert len(vals) == len(times)
    #
    #     ts_nemo_1way = pd.Series(index=[d for d in times], data=vals)
    #     ts_nemo_1way.sort_index(inplace=True)


    # nemo offline
    with MFDataset("/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/outputs_nemo_offline/GLK_1d_*_grid_T.nc") as ds:
        timevar = ds.variables["time_counter"]
        print(timevar.units)
        times = num2date(timevar[:], timevar.units)

        print(times[0], times[-1], type(times[0]))

        print(times[:10])

        print("===== avg_mask ====")
        print(avg_mask.min(), avg_mask.max(), avg_mask.shape)

        print(ds.variables["soicecov"][0, :, :].shape)
        vals = [field.transpose()[avg_mask].mean() for field in ds.variables["soicecov"][:]]

        assert len(vals) == len(times)

        ts_nemo_offline = pd.Series(index=[d for d in times], data=vals)
        ts_nemo_offline.sort_index(inplace=True)


    # nemo offline
    with MFDataset("/HOME/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/outputs_nemo_offline/dt_5min/GLK_1d_*_grid_T.nc") as ds:
        timevar = ds.variables["time_counter"]
        print(timevar.units)
        times = num2date(timevar[:], timevar.units)

        print(times[0], times[-1], type(times[0]))

        print(times[:10])

        print("===== avg_mask ====")
        print(avg_mask.min(), avg_mask.max(), avg_mask.shape)

        print(ds.variables["soicecov"][0, :, :].shape)
        vals = [field.transpose()[avg_mask].mean() for field in ds.variables["soicecov"][:]]

        assert len(vals) == len(times)

        ts_nemo_offline_5min = pd.Series(index=[d for d in times], data=vals)
        ts_nemo_offline_5min.sort_index(inplace=True)

    # nemo offline
    with MFDataset("/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/outputs_nemo_offline/doubled_snowfall/GLK_1d_*_grid_T.nc") as ds:
        timevar = ds.variables["time_counter"]
        print(timevar.units)
        times = num2date(timevar[:], timevar.units)

        print(times[0], times[-1], type(times[0]))

        print(times[:10])

        print("===== avg_mask ====")
        print(avg_mask.min(), avg_mask.max(), avg_mask.shape)

        print(ds.variables["soicecov"][0, :, :].shape)
        vals = [field.transpose()[avg_mask].mean() for field in ds.variables["soicecov"][:]]

        assert len(vals) == len(times)

        ts_nemo_offline_double_snow = pd.Series(index=[d for d in times], data=vals)
        ts_nemo_offline_double_snow.sort_index(inplace=True)

    # nemo offline
    with MFDataset("/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/outputs_nemo_offline/dt_5min_eddy_tra10_dyn30/GLK_1d_*_grid_T.nc") as ds:
        timevar = ds.variables["time_counter"]
        print(timevar.units)
        times = num2date(timevar[:], timevar.units)

        print(times[0], times[-1], type(times[0]))

        print(times[:10])

        print("===== avg_mask ====")
        print(avg_mask.min(), avg_mask.max(), avg_mask.shape)

        print(ds.variables["soicecov"][0, :, :].shape)
        vals = [field.transpose()[avg_mask].mean() for field in ds.variables["soicecov"][:]]

        assert len(vals) == len(times)

        ts_nemo_offline_dt5min_small_ldf = pd.Series(index=[d for d in times], data=vals)
        ts_nemo_offline_dt5min_small_ldf.sort_index(inplace=True)


   # nemo offline
    with MFDataset("/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/outputs_nemo_offline/dt_30min_eddy_tra10_dyn30_traldf3d/GLK_1d_*_grid_T.nc") as ds:
        timevar = ds.variables["time_counter"]
        print(timevar.units)
        times = num2date(timevar[:], timevar.units)

        print(times[0], times[-1], type(times[0]))

        print(times[:10])

        print("===== avg_mask ====")
        print(avg_mask.min(), avg_mask.max(), avg_mask.shape)

        print(ds.variables["soicecov"][0, :, :].shape)
        vals = [field.transpose()[avg_mask].mean() for field in ds.variables["soicecov"][:]]

        assert len(vals) == len(times)

        ts_nemo_offline_dt30min_small_ldf = pd.Series(index=[d for d in times], data=vals)
        ts_nemo_offline_dt30min_small_ldf.sort_index(inplace=True)

    # nemo offline
    with MFDataset("/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/outputs_nemo_offline/dt_5min_eddy_tra10_dyn30_sbc_forcing_3hourly/GLK_1d_*_grid_T.nc") as ds:
        timevar = ds.variables["time_counter"]
        print(timevar.units)
        times = num2date(timevar[:], timevar.units)

        print(times[0], times[-1], type(times[0]))

        print(times[:10])

        print("===== avg_mask ====")
        print(avg_mask.min(), avg_mask.max(), avg_mask.shape)

        print(ds.variables["soicecov"][0, :, :].shape)
        vals = [field.transpose()[avg_mask].mean() for field in ds.variables["soicecov"][:]]

        assert len(vals) == len(times)

        ts_nemo_offline_dt5min_small_ldf_sbc3h = pd.Series(index=[d for d in times], data=vals)
        ts_nemo_offline_dt5min_small_ldf_sbc3h.sort_index(inplace=True)


    # nemo offline
    with MFDataset("/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/outputs_nemo_offline/dt_5min_eddy_tra10_dyn30_sbc_forcing_6hourly_hicrit_0.05m/GLK_1d_*_grid_T.nc") as ds:
        timevar = ds.variables["time_counter"]
        print(timevar.units)
        times = num2date(timevar[:], timevar.units)

        print(times[0], times[-1], type(times[0]))

        print(times[:10])

        print("===== avg_mask ====")
        print(avg_mask.min(), avg_mask.max(), avg_mask.shape)

        print(ds.variables["soicecov"][0, :, :].shape)
        vals = [field.transpose()[avg_mask].mean() for field in ds.variables["soicecov"][:]]

        assert len(vals) == len(times)

        ts_nemo_offline_dt5min_small_ldf_sbc6h = pd.Series(index=[d for d in times], data=vals)
        ts_nemo_offline_dt5min_small_ldf_sbc6h.sort_index(inplace=True)



    # nemo offline
    with MFDataset("/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/outputs_nemo_offline/dt_5min_eddy_tra10_dyn30_sbc_forcing_6hourly_hicrit_0.05m_noicedyn/GLK_1d_*_grid_T.nc") as ds:
        timevar = ds.variables["time_counter"]
        print(timevar.units)
        times = num2date(timevar[:], timevar.units)

        print(times[0], times[-1], type(times[0]))

        print(times[:10])

        print("===== avg_mask ====")
        print(avg_mask.min(), avg_mask.max(), avg_mask.shape)

        print(ds.variables["soicecov"][0, :, :].shape)
        vals = [field.transpose()[avg_mask].mean() for field in ds.variables["soicecov"][:]]

        assert len(vals) == len(times)

        ts_nemo_offline_dt5min_small_ldf_sbc6h_nodyn = pd.Series(index=[d for d in times], data=vals)
        ts_nemo_offline_dt5min_small_ldf_sbc6h_nodyn.sort_index(inplace=True)


   # nemo offline
    with MFDataset("/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/outputs_nemo_offline/dt_5min_eddy_tra10_dyn30_sbc_forcing_5min_noicedyn/GLK_1d_*_grid_T.nc") as ds:
        timevar = ds.variables["time_counter"]
        print(timevar.units)
        times = num2date(timevar[:], timevar.units)

        print(times[0], times[-1], type(times[0]))

        print(times[:10])

        print("===== avg_mask ====")
        print(avg_mask.min(), avg_mask.max(), avg_mask.shape)

        print(ds.variables["soicecov"][0, :, :].shape)
        vals = [field.transpose()[avg_mask].mean() for field in ds.variables["soicecov"][:]]

        assert len(vals) == len(times)

        ts_nemo_offline_dt5min_small_ldf_sbc5min_nodyn = pd.Series(index=[d for d in times], data=vals)
        ts_nemo_offline_dt5min_small_ldf_sbc5min_nodyn.sort_index(inplace=True)



   # nemo offline
    with MFDataset("/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/outputs_nemo_offline/dt_5min_eddy_tra10_dyn30_sbc_forcing_5min_noicedyn_notime_interpolation_of_forcing/GLK_1d_*_grid_T.nc") as ds:
        timevar = ds.variables["time_counter"]
        print(timevar.units)
        times = num2date(timevar[:], timevar.units)

        print(times[0], times[-1], type(times[0]))

        print(times[:10])

        print("===== avg_mask ====")
        print(avg_mask.min(), avg_mask.max(), avg_mask.shape)

        print(ds.variables["soicecov"][0, :, :].shape)
        vals = [field.transpose()[avg_mask].mean() for field in ds.variables["soicecov"][:]]

        assert len(vals) == len(times)

        ts_nemo_offline_dt5min_small_ldf_sbc5min_nodyn_noforc_tint = pd.Series(index=[d for d in times], data=vals)
        ts_nemo_offline_dt5min_small_ldf_sbc5min_nodyn_noforc_tint.sort_index(inplace=True)


   # nemo offline
    with MFDataset("/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/outputs_nemo_offline/dt_5min_eddy_tra10_dyn30_sbc5min_lim3_6h/GLK_1d_*_grid_T.nc") as ds:
        timevar = ds.variables["time_counter"]
        print(timevar.units)
        times = num2date(timevar[:], timevar.units)

        print(times[0], times[-1], type(times[0]))

        print(times[:10])

        print("===== avg_mask ====")
        print(avg_mask.min(), avg_mask.max(), avg_mask.shape)

        print(ds.variables["soicecov"][0, :, :].shape)
        vals = [field.transpose()[avg_mask].mean() for field in ds.variables["soicecov"][:]]

        assert len(vals) == len(times)

        ts_nemo_offline_dt5min_small_ldf_sbc5min_lim6h = pd.Series(index=[d for d in times], data=vals)
        ts_nemo_offline_dt5min_small_ldf_sbc5min_lim6h.sort_index(inplace=True)


    #  nemo offline
    with MFDataset("/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/outputs_nemo_offline/dt_5min_eddy_tra10_dyn30_tke_penetr10m/GLK_1d_*_grid_T.nc") as ds:
        timevar = ds.variables["time_counter"]
        print(timevar.units)
        times = num2date(timevar[:], timevar.units)

        print(times[0], times[-1], type(times[0]))

        print(times[:10])

        print("===== avg_mask ====")
        print(avg_mask.min(), avg_mask.max(), avg_mask.shape)

        print(ds.variables["soicecov"][0, :, :].shape)
        vals = [field.transpose()[avg_mask].mean() for field in ds.variables["soicecov"][:]]

        assert len(vals) == len(times)

        ts_nemo_offline_dt5min_small_ldf_sbc5min_tkepenet_10m = pd.Series(index=[d for d in times], data=vals)
        ts_nemo_offline_dt5min_small_ldf_sbc5min_tkepenet_10m.sort_index(inplace=True)

   #  nemo offline
    with MFDataset("/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/outputs_nemo_offline/dt_5min_eddy_tra10_dyn30_tke0.01_penetr10m/GLK_1d_*_grid_T.nc") as ds:
        timevar = ds.variables["time_counter"]
        print(timevar.units)
        times = num2date(timevar[:], timevar.units)

        print(times[0], times[-1], type(times[0]))

        print(times[:10])

        print("===== avg_mask ====")
        print(avg_mask.min(), avg_mask.max(), avg_mask.shape)

        print(ds.variables["soicecov"][0, :, :].shape)
        vals = [field.transpose()[avg_mask].mean() for field in ds.variables["soicecov"][:]]

        assert len(vals) == len(times)

        ts_nemo_offline_dt5min_small_ldf_sbc5min_tke0001penet_10m = pd.Series(index=[d for d in times], data=vals)
        ts_nemo_offline_dt5min_small_ldf_sbc5min_tke0001penet_10m.sort_index(inplace=True)


   #  nemo offline
    with MFDataset("/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/outputs_nemo_offline/dt_5min_eddy_tra10_dyn30_sbc_forcing_5min_gls1/GLK_1d_*_grid_T.nc") as ds:
        timevar = ds.variables["time_counter"]
        print(timevar.units)
        times = num2date(timevar[:], timevar.units)

        print(times[0], times[-1], type(times[0]))

        print(times[:10])

        print("===== avg_mask ====")
        print(avg_mask.min(), avg_mask.max(), avg_mask.shape)

        print(ds.variables["soicecov"][0, :, :].shape)
        vals = [field.transpose()[avg_mask].mean() for field in ds.variables["soicecov"][:]]

        assert len(vals) == len(times)

        ts_nemo_offline_dt5min_small_ldf_sbc5min_gls1 = pd.Series(index=[d for d in times], data=vals)
        ts_nemo_offline_dt5min_small_ldf_sbc5min_gls1.sort_index(inplace=True)


   #  nemo offline
    data_folder = Path("/HOME/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/outputs_nemo_offline/zdf_gls_dt_and_sbc_5min/")
    with MFDataset([str(f) for f in sorted(data_folder.iterdir(), key=lambda p: p.name) if f.name.endswith("_grid_T.nc")]) as ds:
        timevar = ds.variables["time_counter"]
        print(timevar.units)
        times = num2date(timevar[:], timevar.units)

        print(times[0], times[-1], type(times[0]))
        print(times[:10])

        print("===== avg_mask ====")
        print(avg_mask.min(), avg_mask.max(), avg_mask.shape)

        print(ds.variables["soicecov"][0, :, :].shape)
        vals = [field.transpose()[avg_mask].mean() for field in ds.variables["soicecov"][:]]

        assert len(vals) == len(times)

        ts_nemo_offline_dt5min_small_ldf_sbc5min_gls1_5year = pd.Series(index=[d for d in times], data=vals)
        ts_nemo_offline_dt5min_small_ldf_sbc5min_gls1_5year.sort_index(inplace=True)



    fig = plt.figure()

    print("hl_icefrac :", hl_icefrac.index[0], hl_icefrac.index[-1], type(hl_icefrac))
    print("nemo_icefrac :", nemo_icefrac.index[0], nemo_icefrac.index[-1])

    ax = plt.gca()

    # ax = hl_icefrac.plot(lw=2, color="k", label=HL_LABEL)
    # nemo_icefrac.plot(lw=2, color="r", ax=ax, label=NEMO_LABEL)
    # ts_nemo_1way.plot(lw=2, color="g", label="CRCM5_NEMO_oneway", ax=ax, zorder=5)
    # obs_icefrac.plot(lw=2, color="b", ax=ax, label="Obs.")

    ax.plot(hl_icefrac.index, hl_icefrac.values, lw=2, color="b", label=HL_LABEL)
    ax.plot(nemo_icefrac.index, nemo_icefrac.values, lw=2, color="r", label=NEMO_LABEL)
    # ax.plot(ts_nemo_1way.index, ts_nemo_1way.values, lw=2, color="g", label="CRCM5_NEMO_oneway")
    ax.plot(obs_icefrac.index, obs_icefrac.values, lw=2, color="k", label="Obs.")
    #ax.plot(ts_nemo_offline.index, ts_nemo_offline.values, lw=2, color="#FFA500", label="NEMO-offline")
    #ax.plot(ts_nemo_offline_5min.index, ts_nemo_offline_5min.values, lw=2, color="g", label="NEMO-offline-5min")
    #ax.plot(ts_nemo_offline_double_snow.index, ts_nemo_offline_double_snow.values, lw=2, color="m", label="NEMO-offline-doublesnow")

    ax.plot(ts_nemo_offline_dt5min_small_ldf.index, ts_nemo_offline_dt5min_small_ldf.values, lw=2, color="g", label="NEMO-offline dt=5min, small ldf")
    # ax.plot(ts_nemo_offline_dt30min_small_ldf.index, ts_nemo_offline_dt30min_small_ldf.values, lw=2, color="y", label="NEMO-offline dt=30min,traldf3d, small ldf")
    # ax.plot(ts_nemo_offline_dt5min_small_ldf_sbc3h.index, ts_nemo_offline_dt5min_small_ldf_sbc3h.values, lw=2, color="c", label="NEMO-offline dt=5min,traldf3d, small ldf, sbc3h")
    ax.plot(ts_nemo_offline_dt5min_small_ldf_sbc6h.index, ts_nemo_offline_dt5min_small_ldf_sbc6h.values, lw=2, color="#6ACC65", label="NEMO-offline dt=5min,traldf3d, small ldf, sbc6h, hicr5cm")
    # ax.plot(ts_nemo_offline_dt5min_small_ldf_sbc6h_nodyn.index, ts_nemo_offline_dt5min_small_ldf_sbc6h_nodyn.values, lw=2, color="#6ACEEE", label="NEMO-offline dt=5min,traldf3d, small ldf, sbc6h, noicedyn")
    # ax.plot(ts_nemo_offline_dt5min_small_ldf_sbc5min_nodyn.index, ts_nemo_offline_dt5min_small_ldf_sbc5min_nodyn.values, lw=2, color="#9ABF12", label="NEMO-offline dt=5min,traldf3d, small ldf, sbc5min, noicedyn")
    # ax.plot(ts_nemo_offline_dt5min_small_ldf_sbc5min_nodyn_noforc_tint.index, ts_nemo_offline_dt5min_small_ldf_sbc5min_nodyn_noforc_tint.values, lw=2, color="#111111", label="NEMO-offline dt=5min,traldf3d,small ldf,sbc5min,noicedyn,noforctint")


    # ax.plot(ts_nemo_offline_dt5min_small_ldf_sbc5min_lim6h.index, ts_nemo_offline_dt5min_small_ldf_sbc5min_lim6h.values, lw=2,
    #         color="m", label="NEMO-offline dt=5min,traldf3d, small ldf, sbc5min, lim 6h")
    #
    # ax.plot(ts_nemo_offline_dt5min_small_ldf_sbc5min_tkepenet_10m.index, ts_nemo_offline_dt5min_small_ldf_sbc5min_tkepenet_10m.values, lw=2,
    #         color="y", label="NEMO-offline dt=5min,traldf3d,small ldf,sbc5min,tke penet 10m")
    #
    # ax.plot(ts_nemo_offline_dt5min_small_ldf_sbc5min_tke0001penet_10m.index, ts_nemo_offline_dt5min_small_ldf_sbc5min_tke0001penet_10m.values, lw=2,
    #         color="#ccceee", label="NEMO-offline dt=5min,traldf3d,small ldf,sbc5min,tke penet0.01 10m")
    #
    # ax.plot(ts_nemo_offline_dt5min_small_ldf_sbc5min_gls1.index, ts_nemo_offline_dt5min_small_ldf_sbc5min_gls1.values, lw=2,
    #         color="#aaa111", label="NEMO-offline dt=5min,traldf3d,small ldf,sbc5min,GLS1m")

    ax.plot(ts_nemo_offline_dt5min_small_ldf_sbc5min_gls1_5year.index, ts_nemo_offline_dt5min_small_ldf_sbc5min_gls1_5year.values, lw=2,
            color="#aaa111", label="NEMO-offline dt=5min,traldf3d,small ldf,sbc5min,GLS1m,5year")

    # ax.set_xlim([date2num(datetime(1979, 10, 1)), date2num(datetime(1980, 12, 31))])

    ax.xaxis.set_major_locator(YearLocator())

    fig.autofmt_xdate()

    ax.legend()
    ax.grid()


    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    fig.tight_layout()
    fig.savefig(os.path.join(img_folder, "lake_icefr_ts_{}-{}.png".format(start_year, end_year)), dpi=commons.dpi, transparent=True, bbox_inches="tight")




if __name__ == '__main__':
    main()
