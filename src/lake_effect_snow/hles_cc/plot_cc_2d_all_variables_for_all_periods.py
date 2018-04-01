

"""
Plot the panel of cc for selected periods to all variables

for now: HLES, ice cover

"""
from collections import OrderedDict, defaultdict

from matplotlib.gridspec import GridSpec

from application_properties import main_decorator
from data.robust.data_manager import DataManager
from data.robust import data_source_types
from lake_effect_snow.hles_cc import common_params
from util import plot_utils
import matplotlib.pyplot as plt
from pendulum import Period
from datetime import datetime




def get_mask_from_hles(path):
    # TODO: implement
    pass


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

    season_to_months = OrderedDict([
        ("ND", [11, 12]),
        ("JF", [1, 2]),
        ("MA", [3, 4])
    ])

    varnames = ["hles_snow", "lake_ice_fraction",]

    var_display_names = {
        "hles_snow": "HLES",
        "lake_ice_fraction": "Lake ice fraction"
    }

    plot_utils.apply_plot_params(width_cm=18, height_cm=5.5, font_size=8)

    main(label_to_datapath, varnames=varnames,
         cur_label=common_params.crcm_nemo_cur_label,
         fut_label=common_params.crcm_nemo_fut_label,
         season_to_months=season_to_months,
         vname_display_names=var_display_names)



@main_decorator
def main(label_to_data_path: dict, varnames=None, season_to_months: dict=None,
         cur_label="", fut_label="",
         vname_to_mask: dict=None, vname_display_names:dict=None,
         pval_crit=0.1):

    """

    :param label_to_data_path:
    :param varnames:
    :param season_to_months:
    :param cur_label:
    :param fut_label:
    :param vname_to_mask: - to mask everything except the region of interest
    """

    if vname_display_names is None:
        vname_display_names = {}



    varname_mapping = {v: v for v in varnames}
    level_mapping = {v: 0 for v in varnames} # Does not really make a difference, since all variables are 2d

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
    var_to_season_to_data = {}
    for vname in varnames:
        cur_means = cur_dm.get_seasonal_means()


    # Plotting
    # panel grid dimensions
    ncols = len(season_to_months)
    nrows = len(varnames)

    gs = GridSpec(nrows, ncols)
    fig = plt.figure()

    for col, seas_name in enumerate(season_to_months):
        for row, vname in enumerate(varnames):

            ax = fig.add_subplot(gs[row, col])

            # set season titles
            if row == 0:
                ax.set_title(seas_name)

            # identify variable names
            if col == 0:
                ax.set_ylabel(vname_display_names.get(vname, vname))



    # Save the figure in file
    img_folder = common_params.img_folder
    img_folder.mkdir(exist_ok=True)

    img_file = img_folder / f"cc_{fut_label}-{cur_label}.png"

    fig.savefig(str(img_file), **common_params.image_file_options)



if __name__ == '__main__':
    entry_for_cc_canesm2_gl()