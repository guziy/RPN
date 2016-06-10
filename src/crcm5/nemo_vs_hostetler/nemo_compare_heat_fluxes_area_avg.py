from collections import OrderedDict

from rpn import level_kinds

from application_properties import main_decorator
from crcm5.nemo_vs_hostetler import commons
from crcm5.nemo_vs_hostetler.ice_fraction_area_avg import get_area_avg_timeseries
from crcm5.nemo_vs_hostetler.time_height_plots_area_avg import get_nemo_lakes_mask


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
