from collections import namedtuple, OrderedDict
from pathlib import Path
from matplotlib.gridspec import GridSpec
from util import plot_utils

__author__ = 'huziy'

from crcm5.analyse_hdf import do_analysis_using_pytables as analysis

import matplotlib.pyplot as plt

FieldInfo = namedtuple("FieldInfo", "varname start_year end_year basemap lons lats")
img_folder = "cc_paper/bfe"

def plot_bfe_row_for_var(season_to_diff=None, field_info=None, ax_list=None):

    assert hasattr(field_info, "varname")
    assert hasattr(field_info, "start_year")
    assert hasattr(field_info, "end_year")
    assert hasattr(field_info, "end_year")


# Plot boundary forcing errors (compare with ERAI-driven simulation)
def draw_bfe_in_seasonal_mean(varnames, level_index_list=None, season_to_months=None,
                              path_to_era_driven="", path_to_gcm_driven="",
                              start_year=None, end_year=None):

    lons, lats, bmp = analysis.get_basemap_from_hdf(file_path=path_to_era_driven)

    if level_index_list is None:
        level_index_list = [0, ] * len(varnames)



    for vname, level_index in zip(varnames, level_index_list):
        season_to_diff = {}

        finf = FieldInfo(vname, start_year, end_year, bmp, lons, lats)
        for season, months in season_to_months.items():
            era = analysis.get_seasonal_climatology(hdf_path=path_to_era_driven,
                                                    start_year=start_year, end_year=end_year,
                                                    var_name=vname, level=level_index, months=months)

            gcm = analysis.get_seasonal_climatology(hdf_path=path_to_gcm_driven,
                                                    start_year=start_year, end_year=end_year,
                                                    var_name=vname, level=level_index, months=months)

            season_to_diff[season] = gcm - era

        # Plot the differences for each season ...
        plot_bfe_row_for_var(season_to_diff=season_to_diff, field_info=finf)





def draw_bfe_in_seasonal_max(varnames, level_index=0, season_to_months=None,
                             path_to_era_driven="", path_to_gcm_driven="",
                             start_year=None, end_year=None):
    pass


def main():
    path_to_era_driven = "/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5"
    path_to_gcm_driven = \
        "/RESCUE/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"

    # Year range ...
    start_year = 1980
    end_year = 2010


    seasons_for_mean = OrderedDict(
        ("Winter", (12, 1, 2)),
        ("Spring", range(3, 6)),
        ("Summer", range(6, 9)),
        ("Fall", range(9, 12))
    )

    seasons_for_max = OrderedDict(
        ("Annual maximum", range(1, 13))
    )

    variables_mean_bfe = ["TT", "PR", ]

    variables_annual_max_bfe = ["I5", ]

    all_vars = variables_mean_bfe + variables_annual_max_bfe

    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=20)
    fig = plt.figure()

    nrows = len(all_vars)
    ncols = max([len(seasons_for_mean), len(seasons_for_max)]) + 1
    gs = GridSpec(nrows, ncols, width_ratios=[1., ] * (ncols - 1) + [0.05, ])





    # Save the figure to a file
    p = Path(img_folder)
    if not p.is_dir():
        p.mkdir(parents=True)

    img_name = "{}_{}-{}_{}.png".format()

    img_path = str(p.joinpath(img_name))
    fig.tight_layout()
    fig.savefig(img_path)


if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()
    main()