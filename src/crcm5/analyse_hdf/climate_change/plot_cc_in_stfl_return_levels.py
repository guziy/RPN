from collections import OrderedDict
from pathlib import Path
from rpn.rpn import RPN
from crcm5.analyse_hdf.run_config import RunConfig
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis

__author__ = 'huziy'

img_folder = Path("cc_paper")


class ExtremeProperties(object):
    low = "low"
    high = "high"
    extreme_types = [high, low]

    extreme_type_to_return_periods = OrderedDict([
        ("high", [10, 50]),
        ("low", [2, 5]),
    ])

    extreme_type_to_month_of_interest = OrderedDict([
        ("high", range(3, 7)),
        ("low", range(1, 6)),
    ])

    extreme_type_to_n_agv_days = OrderedDict([
        ("high", 1),
        ("low", 15),
    ])

    def __init__(self, ret_lev_dict=None, std_dict=None):
        self.return_lev_dict = ret_lev_dict
        self.std_dict = std_dict

    def get_low_rl_for_period(self, return_period=2):
        return self.return_lev_dict[self.low][return_period]

    def get_high_rl_for_period(self, return_period=10):
        return self.return_lev_dict[self.high][return_period]

    def get_rl_and_std(self, ex_type=high, return_period=10):
        """
        Return level along with the standard deviation calculated using bootstrap
        :param ex_type:
        :param return_period:
        :return:
        """
        return [z[ex_type][return_period] for z in (self.return_lev_dict, self.std_dict)]


def get_return_levels_and_unc_using_bootstrap(rconfig, varname="STFL"):
    """
    return the extreme properties object
    :param rconfig:
    :param varname:
    """
    result = ExtremeProperties()
    for extr_type, months in ExtremeProperties.extreme_type_to_month_of_interest.items():

        # 3D array of annual extremes for each grid point
        ext_values = analysis.get_annual_extrema(rconfig=rconfig, varname=varname,
                                                 months_of_interest=months,
                                                 n_avg_days=ExtremeProperties.extreme_type_to_n_agv_days[extr_type],
                                                 high_flow=ExtremeProperties.high == extr_type)


        nyears = ext_values.shape[0]


        for ret_period in ExtremeProperties.extreme_type_to_return_periods[extr_type]:
            print(extr_type, months, ret_period)




def main():
    import application_properties
    application_properties.set_current_directory()


    # Create folder for output images
    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)


    rea_driven_path = "/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5"
    rea_driven_label = "CRCM5-L-ERAI"

    gcm_driven_path_c = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"
    gcm_driven_label_c = "CRCM5-L"

    start_year_c = 1980
    end_year_c = 2010

    varname = "STFL"

    future_shift_years = 75


    params = dict(
        data_path=rea_driven_path, start_year=start_year_c, end_year=end_year_c, label=rea_driven_label)

    geo_data_file = "/skynet3_rech1/huziy/hdf_store/pm1979010100_00000000p"

    rea_driven_config = RunConfig(**params)
    params.update(dict(data_path=gcm_driven_path_c, label=gcm_driven_label_c))

    gcm_driven_config_c = RunConfig(**params)
    gcm_driven_config_f = gcm_driven_config_c.get_shifted_config(shift_years=future_shift_years)

    r_obj = RPN(geo_data_file)
    facc = r_obj.get_first_record_for_name("FAA")
    fldr = r_obj.get_first_record_for_name("FLDR")

    # get basemap information
    bmp_info = analysis.get_basemap_info_from_hdf(file_path=rea_driven_path)



    get_return_levels_and_unc_using_bootstrap(gcm_driven_config_c)



if __name__ == '__main__':

    main()