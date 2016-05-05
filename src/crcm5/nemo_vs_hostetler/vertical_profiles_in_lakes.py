from collections import OrderedDict
import numpy as np
from rpn import level_kinds





def get_temperature_profile_HL(start_year=-np.Inf, end_year=np.Inf, samples_dir=""):
    """
    get water temperature profile from Hostetler outputs
    TT(time, z)
    :param start_year:
    :param end_year:
    :param samples_dir:
    """
    pass

def get_temperature_profile_NEMO(start_year=-np.Inf, end_year=np.Inf, samples_dir=""):
    """
    get water temperature profile from NEMO outputs
    TT(time, z)
    :param start_year:
    :param end_year:
    :param samples_dir:
    """
    pass



def main():
    start_year = 1979
    end_year = 1981

    HL_LABEL = "CRCM5_HL"
    NEMO_LABEL = "CRCM5_NEMO"

    dx = 0.1
    dy = 0.1

    file_prefix = "pm"
    PR_level = -1
    PR_level_type = level_kinds.ARBITRARY

    tprecip_vname = "PR"
    sprecip_vname = "SN"

    TT_level = 1
    TT_level_type = level_kinds.HYBRID

    sim_label_to_path = OrderedDict(
        [(HL_LABEL, "/RESCUE/skynet3_rech1/huziy/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl_oneway/Samples"),
         (NEMO_LABEL, "/HOME/huziy/skynet3_rech1/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl/Samples")]
    )










if __name__ == '__main__':
    main()