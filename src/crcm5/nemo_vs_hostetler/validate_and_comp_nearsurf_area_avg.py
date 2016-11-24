from collections import OrderedDict
from pathlib import Path

from matplotlib.gridspec import GridSpec
from netCDF4 import MFDataset, num2date, Dataset
from rpn import level_kinds
from scipy.spatial import cKDTree as KDTree

from application_properties import main_decorator


# Read ERA-Interim data from netcdf
# Read CRCM5 outputs from CRCM5_NEMO and CRCM5_HL simulations

# get the lake mask and compare the area-mean timeseries
from crcm5.nemo_vs_hostetler.ice_fraction_area_avg import get_area_avg_timeseries
from crcm5.nemo_vs_hostetler.time_height_plots_area_avg import get_nemo_lakes_mask
from util import plot_utils
import matplotlib.pyplot as plt

import numpy as np
import os
import pandas as pd
from datetime import datetime

from util.geo import lat_lon

img_folder = "nemo_vs_hostetler"


def get_area_avg_from_erai_data(start_year=-np.Inf, end_year=np.Inf, var_folder="", varname="", mask=None, mask_lons=None, mask_lats=None):

    """

    Interpolate the mask to the ERA-Interim grid using nearest neighbour approach

    :param start_year:
    :param end_year:
    :param var_folder:
    :param varname:
    :param mask:
    :return:
    """

    def _get_year(fn):
        return int(fn.split(".")[0].split("_")[1])

    flist = [os.path.join(var_folder, fn) for fn in os.listdir(var_folder) if fn.startswith(varname) and (start_year <= _get_year(fn)) and (_get_year(fn) <= end_year)]
    print(flist)


    ktree = None
    mask_interpolated = None
    lons_target, lats_target = None, None

    ser_list = []
    for fp in flist:

        with Dataset(fp) as ds:
            time_var = ds.variables["time"]

            times = num2date(time_var[:], time_var.units)

            print(times[0], times[-1])

            # Determine nearest neighbours for interpolation (do it only once)
            if ktree is None:

                # get lons and lats from the bathymetry file
                data_folder_p = Path(var_folder).parent

                for f in data_folder_p.iterdir():
                    if f.name.lower().startswith("bathy_meter"):
                        with Dataset(str(f)) as ds_bathy:
                            lons_target, lats_target = [ds_bathy.variables[k][:] for k in ["nav_lon", "nav_lat"]]
                            break


                x, y, z = lat_lon.lon_lat_to_cartesian(mask_lons.flatten(), mask_lats.flatten())
                xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons_target.flatten(), lats_target.flatten())
                ktree = KDTree(list(zip(x, y, z)))

                dists, inds = ktree.query(list(zip(xt, yt, zt)), k=1)

                mask_interpolated = mask.flatten()[inds]
                mask_interpolated = mask_interpolated.reshape(lons_target.shape)


            vals = [field[mask_interpolated].mean() for field in ds.variables[varname][:]]
            ser = pd.Series(index=times, data=vals)

            if varname == "TT":
                ser -= 273.15

            ser.sort_index(inplace=True)

            ser_list.append(ser)

    return pd.concat(ser_list)



@main_decorator
def main():
    start_year = 1979
    end_year = 1981

    HL_LABEL = "CRCM5_HL"
    NEMO_LABEL = "CRCM5_NEMO"



    # Older, shorter [1971 - 1981], smaller domain simulations
    sim_label_to_path = OrderedDict(
        [(HL_LABEL, "/RESCUE/skynet3_rech1/huziy/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl_oneway/Samples"),
         (NEMO_LABEL, "/HOME/huziy/skynet3_rech1/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl/Samples")]
    )

    var_name_list = ["TT", "HU"]

    vname_to_level = {
        "TT": 1, "PR": -1, "SN": -1, "LC": -1, "HR": 1, "HU": 1, "AV": -1, "I5": -1, "AL": -1
    }

    vname_to_level_kind = {
        "TT": level_kinds.HYBRID, "PR": level_kinds.ARBITRARY, "SN": level_kinds.ARBITRARY,
        "LC": level_kinds.ARBITRARY, "HR": level_kinds.HYBRID, "HU": level_kinds.HYBRID, "AV": level_kinds.ARBITRARY,
        "I5": level_kinds.ARBITRARY, "AL": level_kinds.ARBITRARY, "J8": level_kinds.ARBITRARY
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
        "AL": "pm",
        "J8": "pm"
    }

    # ---> ---->
    avg_mask = get_nemo_lakes_mask(samples_dir=sim_label_to_path[NEMO_LABEL])

    # extend the mask
    dn = 40
    marginx = dn // 2
    marginy = dn // 2
    nx_ext, ny_ext = [(n + dn) for n in avg_mask.shape]
    print([(n + dn) for n in avg_mask.shape])

    avg_mask_ext = np.zeros((nx_ext, ny_ext), dtype=np.bool)
    avg_mask_ext[marginx:-marginx, marginy:-marginy] = avg_mask


    path_to_erai_data = "/RESCUE/skynet3_rech1/huziy/ERA-Interim_0.75_NEMO_pilot/"



    vname_to_ts_hl = OrderedDict()
    vname_to_ts_erai = OrderedDict()
    vname_to_ts_nemo = OrderedDict()


    for vname in var_name_list:


        common_params = dict(start_year=start_year, end_year=end_year,
                             filename_prefix=vname_to_file_prefix[vname], level=vname_to_level[vname],
                             level_kind=vname_to_level_kind[vname], varname=vname, mask=avg_mask)

        current_label = HL_LABEL
        vname_to_ts_hl[vname] = get_area_avg_timeseries(sim_label_to_path[current_label], **common_params)

        current_label = NEMO_LABEL
        vname_to_ts_nemo[vname] = get_area_avg_timeseries(sim_label_to_path[current_label], **common_params)


        vname_to_ts_erai[vname] = get_area_avg_from_erai_data(start_year=start_year, end_year=end_year,
                                                              var_folder=os.path.join(os.path.join(path_to_erai_data, vname)),
                                                              varname=vname, mask=avg_mask_ext)


    plot_utils.apply_plot_params(font_size=12, width_cm=20, height_cm=20)
    fig = plt.figure()
    gs = GridSpec(len(vname_to_ts_hl), 1)

    for row, vname in enumerate(vname_to_ts_hl):
        ax = fig.add_subplot(gs[row, 0])

        ax.set_ylabel(vname)


        coef = 1
        coef_erai = 1

        # plot monthly means for precipitations
        if vname == "PR":
            coef = 24 * 3600 * 1000
            coef_erai = coef / 1000.0 # Already in mm/s


            ts = vname_to_ts_hl[vname].groupby(lambda d: datetime(d.year, d.month, 15)).mean() * coef
            ax.plot(ts.index, ts.values, lw=2, color="b", label=HL_LABEL)

            ts = vname_to_ts_nemo[vname].groupby(lambda d: datetime(d.year, d.month, 15)).mean() * coef
            ax.plot(ts.index, ts.values, lw=2, color="r", label=NEMO_LABEL)

        else:

            ts = vname_to_ts_hl[vname].groupby(lambda d: datetime(d.year, d.month, d.day)).mean() * coef
            ax.plot(ts.index, ts.values, lw=2, color="b", label=HL_LABEL)

            ts = vname_to_ts_nemo[vname].groupby(lambda d: datetime(d.year, d.month, d.day)).mean() * coef
            ax.plot(ts.index, ts.values, lw=2, color="r", label=NEMO_LABEL)


        ts = vname_to_ts_erai[vname].groupby(lambda d: datetime(d.year, d.month, 15)).mean() * coef_erai
        ax.plot(ts.index, ts.values, lw=2, color="k", label="ERA-Interim")

        if row == 0:
            ax.legend()

        ax.grid()


    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(os.path.join(img_folder, "{}-{}_validate_{}_{}_ts_with_erai.png".format(start_year, end_year, HL_LABEL, NEMO_LABEL)), transparent=True, dpi=400)




if __name__ == '__main__':
    main()