from collections import OrderedDict
from pathlib import Path
from rpn.rpn import RPN
from crcm5.analyse_hdf.run_config import RunConfig
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis

from gev_dist import gevfit
import numpy as np

import pickle

__author__ = 'huziy'

img_folder = Path("cc_paper")


class ExtremeProperties(object):
    seed = 10
    nbootstrap = 100

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
        Return level along with the standard deviation calculated
        using bootstrap
        :param ex_type:
        :param return_period:
        :return:
        """
        return [z[ex_type][return_period] for z in (self.return_lev_dict, self.std_dict)]

    def __str__(self):
        s = ""
        for et in self.extreme_types:
            s += et + ", periods:\n\t{}\n".format(",".join([str(t) for t in self.return_lev_dict.keys()]))

        return s


def do_gevfit_for_a_point(data, extreme_type=ExtremeProperties.high, return_periods=None):
    """
    returns 2 dicts (ret_period_to_levels, ret_period_to_std)
    with the layout {return_period: value}
    """
    # to have the same result for different launches and extreme types
    np.random.seed(seed=ExtremeProperties.seed)

    is_high_flow = extreme_type == ExtremeProperties.high

    if return_periods is None:
        return_periods = ExtremeProperties.extreme_type_to_return_periods[extreme_type]

    nyears = len(data)

    ret_period_to_level = {k: -1 for k in return_periods}
    ret_period_to_std = {k: -1 for k in return_periods}

    # return -1 if all the data is 0
    if all(data <= 0):
        return ret_period_to_level, ret_period_to_std

    params = gevfit.optimize_stationary_for_period(
        data, high_flow=is_high_flow
    )

    # Calculate return levels for all return periods
    for t in return_periods:
        ret_period_to_level[t] = gevfit.get_return_level_for_type_and_period(
            params, t, extreme_type=extreme_type)

    ret_period_to_level_list = {k: [] for k in return_periods}

    for b_index in range(ExtremeProperties.nbootstrap):

        indices = np.random.random_integers(0, high=nyears - 1, size=nyears)

        params = gevfit.optimize_stationary_for_period(
            data[indices], high_flow=is_high_flow
        )


        for ret_period in return_periods:
            print(extr_type, months, ret_period)

            ret_period_to_level_list[ret_period].append(
                gevfit.get_return_level_for_type_and_period(
                    params, ret_period, extreme_type=extreme_type
                )
            )

    # Calculate standard deviation of the bootstrapped return levels
    ret_period_to_std = {t: np.std(v) for t, v in ret_period_to_level_list.items()}

    return ret_period_to_level, ret_period_to_std



def get_cache_file_name(rconfig, months=None, ret_period=2, extreme_type="high"):
    months_str = "-".join([str(m) for m in months])

    return "{}_{}-{}_{}_{}.bin".format(
        extreme_type, rconfig.start_year, rconfig.end_year, rconfig.label, months_str)

def get_return_levels_and_unc_using_bootstrap(rconfig, varname="STFL"):
    """
    return the extreme properties object
    :param rconfig:
    :param varname:
    """
    result = ExtremeProperties()
    for extr_type, months in ExtremeProperties.extreme_type_to_month_of_interest.items():

        result.return_lev_dict[extr_type] = {}
        result.std_dict[extr_type] = {}

        return_periods = ExtremeProperties.extreme_type_to_return_periods[extr_type]

        # Do not do the calculations for the cached return periods
        cached_periods = []
        for return_period in list(return_periods):
            # Construct the name of the cache file
            cache_file = get_cache_file_name(rconfig, months=months,
                ret_period=return_period, extreme_type=extr_type)

            p = Path(cache_file)

            if p.is_file():
                cached_periods.append(return_period)
                return_periods.remove(return_period)

                cache_levs, cache_stds = pickle.load(p.open("rb"))

                result.ret_lev_dict[extr_type][ret_period] = cache_levs
                result.std_dict[extr_type][return_period] = cache_stds

        # Do not do anything if the return levels for all periods are cached
        # for this type of extreme events
        if len(return_periods) == 0:
            continue


        # 3D array of annual extremes for each grid point
        ext_values = analysis.get_annual_extrema(rconfig=rconfig, varname=varname,
                                                 months_of_interest=months,
                                                 n_avg_days=ExtremeProperties.extreme_type_to_n_agv_days[extr_type],
                                                 high_flow=ExtremeProperties.high == extr_type)


        nyears = ext_values.shape[0]
        nx, ny = ext_values.shape[1:]

        result.return_lev_dict[extr_type].update({k: -np.ones((nx, ny)) for k in return_periods})
        result.std_dict[extr_type].update({k: -np.ones((nx, ny)) for k in return_periods})

        # Probably needs to be optimized ...
        for i in range(nx):
            for j in range(ny):
                ret_period_to_level, ret_period_to_std = do_gevfit_for_a_point(ext_values[:, i, j],
                    extreme_type=extr_type, return_periods=return_periods)

                for ret_period in return_periods:
                    result.return_lev_dict[extr_type][ret_period][i, j] = ret_period_to_level[ret_period]
                    result.std_dict[extr_type][ret_period][i, j] = ret_period_to_std[ret_period]

        # Save the computed return levels and standard deviations to the cache file
        for return_period in return_periods:
            # Construct the name of the cache file
            cache_file = get_cache_file_name(rconfig, months=months,
                ret_period=return_period, extreme_type=extr_type)

            p = Path(cache_file)

            to_save = [
                result.return_lev_dict[extr_type][ret_period],
                result.std_dict[extr_type][ret_period]
            ]

            pickle.dump(to_save, p.open("wb"))




    return result


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



    rs_gcm_c = get_return_levels_and_unc_using_bootstrap(gcm_driven_config_c)

    print(rs_gcm_c)

if __name__ == '__main__':
    import time
    t0 = time.clock()
    main()
    print("Execution time: {}s".format(time.clock() - t0))
