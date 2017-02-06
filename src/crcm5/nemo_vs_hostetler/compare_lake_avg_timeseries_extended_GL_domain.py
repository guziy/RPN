import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from matplotlib.gridspec import GridSpec
from rpn import level_kinds

from application_properties import main_decorator
from crcm5.nemo_vs_hostetler.ice_fraction_area_avg import get_area_avg_timeseries
from crcm5.nemo_vs_hostetler.time_height_plots_area_avg import get_nemo_lakes_mask
from crcm5.nemo_vs_hostetler.validate_and_comp_nearsurf_area_avg import get_area_avg_from_erai_data
from util import plot_utils
import matplotlib.pyplot as plt


#  Note: This version, for the extended domain is better (more general), because it does not assume
# the grids from ERA-interim or different simulations to be the same

@main_decorator
def main():
    start_year = 1979
    end_year = 1988


    img_folder = "nemo_vs_hostetler_GL_extended_domain"

    # create the image folder if necessary
    img_folder_p = Path(img_folder)
    if not img_folder_p.is_dir():
        img_folder_p.mkdir()



    HL_LABEL = "CRCM5_HL"
    NEMO_LABEL = "CRCM5_NEMO"

    # Older, shorter [1971 - 1981], smaller domain simulations
    # sim_label_to_path = OrderedDict(
    #     [(HL_LABEL, "/RESCUE/skynet3_rech1/huziy/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl_oneway/Samples"),
    #      (NEMO_LABEL, "/HOME/huziy/skynet3_rech1/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl/Samples")]
    # )

    sim_label_to_path = OrderedDict(
        [(HL_LABEL, "/RECH2/huziy/coupling/GL_440x260_0.1deg_GL_with_Hostetler/Samples_selected"),
         (NEMO_LABEL, "/RECH2/huziy/coupling/coupled-GL-NEMO1h_30min/Samples")]
    )

    var_name_list = ["TT",
                     "PR",
                     ]

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
    nemo_lake_mask, mask_lons, mask_lats = get_nemo_lakes_mask(samples_dir=sim_label_to_path[NEMO_LABEL])


    path_to_erai_data = "/RESCUE/skynet3_rech1/huziy/ERA-Interim_0.75_NEMO_pilot/"

    vname_to_ts_hl = OrderedDict()
    vname_to_ts_erai = OrderedDict()
    vname_to_ts_nemo = OrderedDict()

    for vname in var_name_list:
        common_params = dict(start_year=start_year, end_year=end_year,
                             filename_prefix=vname_to_file_prefix[vname], level=vname_to_level[vname],
                             level_kind=vname_to_level_kind[vname], varname=vname,
                             mask=nemo_lake_mask, mask_lons2d=mask_lons, mask_lats2d=mask_lats)

        current_label = HL_LABEL
        vname_to_ts_hl[vname], _, _ = get_area_avg_timeseries(sim_label_to_path[current_label],
                                                        file_per_var=True, **common_params)

        current_label = NEMO_LABEL
        vname_to_ts_nemo[vname], lons2d, lats2d = get_area_avg_timeseries(sim_label_to_path[current_label],
                                                                          **common_params)

        vname_to_ts_erai[vname] = get_area_avg_from_erai_data(start_year=start_year, end_year=end_year,
                                                              var_folder=os.path.join(path_to_erai_data, vname),
                                                              varname=vname, mask=nemo_lake_mask,
                                                              mask_lons=lons2d, mask_lats=lats2d)

    plot_utils.apply_plot_params(font_size=12, width_cm=20, height_cm=20)
    fig = plt.figure()
    gs = GridSpec(len(vname_to_ts_hl), 1)


    plot_monthly_model_outputs = False
    for row, vname in enumerate(vname_to_ts_hl):
        ax = fig.add_subplot(gs[row, 0])

        ax.set_ylabel(vname)

        coef = 1
        coef_erai = 1

        if vname == "PR":
            coef = 24 * 3600 * 1000
            coef_erai = coef / 1000.0  # Already in mm/s
            plot_monthly_model_outputs = True


        ts = vname_to_ts_hl[vname].groupby(lambda d: datetime(d.year, d.month, 15 if plot_monthly_model_outputs else d.day)).mean() * coef
        ax.plot(ts.index, ts.values, lw=2, color="b", label=HL_LABEL)

        ts = vname_to_ts_nemo[vname].groupby(lambda d: datetime(d.year, d.month, 15 if plot_monthly_model_outputs else d.day)).mean() * coef
        ax.plot(ts.index, ts.values, lw=2, color="r", label=NEMO_LABEL)

        ts = vname_to_ts_erai[vname].groupby(lambda d: datetime(d.year, d.month, 15)).mean() * coef_erai
        ax.plot(ts.index, ts.values, lw=2, color="k", label="ERA-Interim")

        if row == 0:
            ax.legend(loc="lower right")

        ax.grid()

    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(os.path.join(img_folder,
                             "{}-{}_validate_{}_{}_ts_with_erai.png".format(start_year, end_year, HL_LABEL, NEMO_LABEL)),
                transparent=True, dpi=400)

if __name__ == '__main__':
    main()