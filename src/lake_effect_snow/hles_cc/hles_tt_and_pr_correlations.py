from collections import OrderedDict
from datetime import datetime

from pendulum import Period

from application_properties import main_decorator
from data.robust import data_source_types
from data.robust.data_manager import DataManager
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.hles_cc import common_params
from lake_effect_snow.hles_cc.cc_period import CcPeriodsInfo
from lake_effect_snow.hles_cc.plot_cc_2d_all_variables_for_all_periods import get_gl_mask
from util import plot_utils


@main_decorator
def entry_for_cc_canesm2_gl():
    """
    for CanESM2 driven CRCM5_NEMO simulation
    """
    data_root = common_params.data_root
    label_to_datapath = OrderedDict([
        (common_params.crcm_nemo_cur_label, data_root / "lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_1989-2010_1989-2010/merged/"),
        (common_params.crcm_nemo_fut_label, data_root / "lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_2079-2100_2079-2100/merged/"),
    ])

    cur_st_date = datetime(1989, 1, 1)
    cur_en_date = datetime(2011, 1, 1)  # end date not inclusive

    fut_st_date = datetime(2079, 1, 1)
    fut_en_date = datetime(2101, 1, 1)  # end date not inclusive

    cur_period = Period(cur_st_date, cur_en_date)
    fut_period = Period(fut_st_date, fut_en_date)

    periods_info = CcPeriodsInfo(cur_period=cur_period, fut_period=fut_period)


    season_to_months = OrderedDict([
        ("ND", [11, 12]),
        ("JF", [1, 2]),
        ("MA", [3, 4])
    ])

    var_pairs = [("hles_snow", "TT"), ("hles_snow", "PR")]

    var_display_names = {
        "hles_snow": "HLES",
        "lake_ice_fraction": "Lake ice fraction",
        "TT": "2m air\n temperature",
        "PR": "total\nprecipitation"
    }

    plot_utils.apply_plot_params(width_cm=18, height_cm=20, font_size=8)

    the_mask = get_gl_mask(label_to_datapath[common_params.crcm_nemo_cur_label])


def main(label_to_data_path: dict, var_pairs: list,
         periods_info: CcPeriodsInfo,
         vname_display_names=None,
         season_to_months: dict=None,
         cur_label=common_params.crcm_nemo_cur_label,
         fut_label=common_params.crcm_nemo_fut_label):


    # get a flat list of all the required variable names (unique)
    varnames = []
    for vpair in var_pairs:
        for v in vpair:
            if v not in varnames:
                varnames.append(v)


    if vname_display_names is None:
        vname_display_names = {}

    varname_mapping = {v: v for v in varnames}
    level_mapping = {v: VerticalLevel(0) for v in varnames} # Does not really make a difference, since all variables are 2d

    comon_store_config = {
        DataManager.SP_DATASOURCE_TYPE: data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES,
        DataManager.SP_INTERNAL_TO_INPUT_VNAME_MAPPING: varname_mapping,
        DataManager.SP_LEVEL_MAPPING: level_mapping
    }

    cur_dm = DataManager(
        store_config=dict({DataManager.SP_BASE_FOLDER: label_to_data_path[cur_label]}, **comon_store_config)
    )

    fut_dm = DataManager(
        store_config=dict({DataManager.SP_BASE_FOLDER: label_to_data_path[fut_label]}, **comon_store_config)
    )

    # get the data and do calculations
    label_to_var_to_season_to_data = {}

    cur_start_yr, cur_end_year = periods_info.get_cur_year_limits()
    fut_start_yr, fut_end_year = periods_info.get_fut_year_limits()

    for vname in varnames:
        cur_means = cur_dm.get_seasonal_means(start_year=cur_start_yr, end_year=cur_end_year,
                                              season_to_months=season_to_months, varname_internal=vname)

        fut_means = fut_dm.get_seasonal_means(start_year=fut_start_yr, end_year=fut_end_year,
                                              season_to_months=season_to_months, varname_internal=vname)


        label_to_data_path[cur_label] = {
            vname: cur_means
        }

        label_to_data_path[fut_label] = {
            vname: fut_means
        }




if __name__ == '__main__':
    entry_for_cc_canesm2_gl()