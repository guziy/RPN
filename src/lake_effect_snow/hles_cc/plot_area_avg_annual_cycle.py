from collections import OrderedDict

from application_properties import main_decorator
from lake_effect_snow.hles_cc import common_params

@main_decorator
def main():
    data_root = common_params.data_root

    label_to_datapath = OrderedDict([
        (common_params.crcm_nemo_cur_label, data_root / "lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_1989-2010_1989-2010"),
        (common_params.crcm_nemo_fut_label, data_root / "lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_2079-2100_2079-2100"),
    ])


    # Mask can be extracted from hles_snow (i.e. smth like ~np.isnan(hles))

if __name__ == '__main__':
    main()