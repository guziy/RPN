from collections import OrderedDict

from matplotlib.gridspec import GridSpec
from rpn import level_kinds

from application_properties import main_decorator
from crcm5.nemo_vs_hostetler import commons
from crcm5.nemo_vs_hostetler.ice_fraction_area_avg import get_area_avg_timeseries
from crcm5.nemo_vs_hostetler.time_height_plots_area_avg import get_nemo_lakes_mask
import os
import matplotlib.pyplot as plt

from util import plot_utils

img_folder = "nemo_vs_hostetler"


@main_decorator
def main():
    start_year = 1979
    end_year = 1981

    HL_LABEL = "CRCM5_HL"
    NEMO_LABEL = "CRCM5_NEMO"

    sim_label_to_path = OrderedDict(
        [(HL_LABEL, "/RESCUE/skynet3_rech1/huziy/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl_oneway/Samples"),
         (NEMO_LABEL, "/HOME/huziy/skynet3_rech1/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl/Samples")]
    )

    var_name_list = ["TT", "LC", "HU", "AV", "AL"]


    vname_to_level = {
        "TT": 1, "PR": -1, "SN": -1, "LC": -1, "HR": 1, "HU": 1, "AV": -1, "I5": -1, "AL": -1
    }

    vname_to_level_kind = {
        "TT": level_kinds.HYBRID, "PR": level_kinds.ARBITRARY, "SN": level_kinds.ARBITRARY,
        "LC": level_kinds.ARBITRARY, "HR": level_kinds.HYBRID, "HU": level_kinds.HYBRID, "AV": level_kinds.ARBITRARY,
        "I5": level_kinds.ARBITRARY, "AL": level_kinds.ARBITRARY, "J8":level_kinds.ARBITRARY
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

    current_label = HL_LABEL

    vname_to_ts = OrderedDict()
    for vname in var_name_list:

        common_params = dict(start_year=start_year, end_year=end_year,
                             filename_prefix=vname_to_file_prefix[vname], level=vname_to_level[vname],
                             level_kind=vname_to_level_kind[vname], varname=vname, mask=avg_mask)

        ts = get_area_avg_timeseries(sim_label_to_path[current_label], **common_params)

        vname_to_ts[vname] = ts


    plot_utils.apply_plot_params(font_size=10, width_cm=20, height_cm=20)
    fig = plt.figure()
    gs = GridSpec(len(vname_to_ts), 1)

    row = 0
    for vname, ts in vname_to_ts.items():
        ax = fig.add_subplot(gs[row, 0])
        ts.plot(lw=2, color="k", ax=ax)
        ax.set_ylabel(vname)
        row += 1



    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    fig.tight_layout()
    fig.savefig(os.path.join(img_folder, "{}_lakeand_air_props_ts_over_GL_{}-{}.png".format(current_label, start_year, end_year)), dpi=commons.dpi, transparent=True)





if __name__ == '__main__':
    main()
