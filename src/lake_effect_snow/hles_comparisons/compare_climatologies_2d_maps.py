# Compare seasonal mean maps of HLES derived from different sources
from collections import OrderedDict


def compare_anusplin_erai075_narr_daymetv3():

    label_to_data_dir = OrderedDict()

    label_to_data_dir["HLES_AE"] = "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_icefix_obs_anuspmaurer_erai_1980-2009"
    label_to_data_dir["HLES_AN"] = "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_icefix_obs_anuspmaurer_narr_1980-2009"
    label_to_data_dir["HLES_DE"] = "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_icefix_obs_daymet_erai_1980-2009"
    label_to_data_dir["HLES_DN"] = "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_icefix_obs_daymet_narr_1980-2009"

    main(label_to_data_dir)



def main(label_to_data_dir):
    """
    main script called with different configuration sets
    :param label_to_data_dir:
    """
    for label, data_dir in label_to_data_dir.items():

        # TODO: implement
        pass




if __name__ == '__main__':
    compare_anusplin_erai075_narr_daymetv3()