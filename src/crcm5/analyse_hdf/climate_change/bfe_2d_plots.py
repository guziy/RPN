from collections import namedtuple, OrderedDict
from pathlib import Path
import collections
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from crcm5 import infovar
from util import plot_utils
from util.geo import quebec_info
from util.geo.basemap_info import BasemapInfo

__author__ = 'huziy'

from crcm5.analyse_hdf import do_analysis_using_pytables as analysis

import matplotlib.pyplot as plt

FieldInfo = namedtuple("FieldInfo", "varname start_year end_year")

img_folder = "cc_paper/bfe"

import numpy as np

BASIN_BOUNDARIES_SHP = quebec_info.BASIN_BOUNDARIES_FILE_LATLON


def get_diff_levels(key_to_data, ncolors=20, varname=""):
    """
    get nice levels for the contourf plot
    :param field:
    :type key_to_data: dict
    """

    locator = MaxNLocator(nbins=ncolors, symmetric=True)
    if varname in ["TT", ]:
        return locator.tick_values(-7, 7)
    elif varname in ["PR", ]:
        return locator.tick_values(-3, 3)

    delta = -1
    for k, field in key_to_data.items():
        if hasattr(field, "mask"):
            good_data = field[~field.mask]
        else:
            good_data = field.flatten()

        delta = max(np.round(np.percentile(np.abs(good_data), 95)), delta)

    return locator.tick_values(-delta, delta)


def plot_bfe_row_for_var(finfo_to_season_to_diff=None, ax_list=None, season_titles=False,
                         varname="", basemap_info=None):
    cmap = cm.get_cmap("RdBu_r", 20)

    assert isinstance(basemap_info, BasemapInfo)

    xx, yy = None, None
    cs = None
    for finfo, season_to_diff in finfo_to_season_to_diff.items():
        assert isinstance(finfo, FieldInfo)

        if finfo.varname != varname:
            continue

        for season in season_to_diff:
            season_to_diff[season] = infovar.get_to_plot(varname, season_to_diff[season], difference=True,
                                                         lons=basemap_info.lons, lats=basemap_info.lats)

        clevs = get_diff_levels(season_to_diff, ncolors=cmap.N, varname=varname)
        for i, (season, diff) in enumerate(season_to_diff.items()):
            ax = ax_list[i]

            if xx is None or yy is None:
                xx, yy = basemap_info.get_proj_xy()

            print(diff.shape)




            cs = basemap_info.basemap.contourf(xx, yy, diff[:], cmap=cmap,
                                               levels=clevs,
                                               extend="both", ax=ax)
            basemap_info.basemap.drawcoastlines(ax=ax)
            # ax.set_aspect("auto")
            basemap_info.basemap.readshapefile(BASIN_BOUNDARIES_SHP[:-4], "basin", ax=ax)

            if season_titles:
                ax.set_title(season)

            if i == 0:
                ax.set_ylabel(infovar.get_display_label_for_var(finfo.varname))

            if finfo.varname in ["I5", ] and season.lower() in ["summer"]:
                ax.set_visible(False)

    ax = ax_list[-1]
    # ax.set_aspect(30)
    ax.set_title(infovar.get_units(varname))
    plt.colorbar(cs, cax=ax_list[-1])


# Plot boundary forcing errors (compare with ERAI-driven simulation)

def get_bfe_in_seasonal_mean(varnames, level_index_dict=None, season_to_months=None,
                             path_to_era_driven="", path_to_gcm_driven="",
                             start_year=None, end_year=None):
    if level_index_dict is None:
        level_index_dict = collections.defaultdict(lambda: 0)

    finfo_to_season_to_diff = {}
    for vname in varnames:
        level_index = level_index_dict[vname]
        season_to_diff = OrderedDict()

        finf = FieldInfo(vname, start_year, end_year)

        for season, months in season_to_months.items():
            era = analysis.get_seasonal_climatology(hdf_path=path_to_era_driven,
                                                    start_year=start_year, end_year=end_year,
                                                    var_name=vname, level=level_index, months=months)

            gcm = analysis.get_seasonal_climatology(hdf_path=path_to_gcm_driven,
                                                    start_year=start_year, end_year=end_year,
                                                    var_name=vname, level=level_index, months=months)

            season_to_diff[season] = gcm - era

        finfo_to_season_to_diff[finf] = season_to_diff

    return finfo_to_season_to_diff


def get_bfe_in_annual_max(varnames, level_index_dict=None,
                          path_to_era_driven="", path_to_gcm_driven="",
                          start_year=None, end_year=None):
    if level_index_dict is None:
        level_index_dict = collections.defaultdict(lambda: 0)

    finfo_to_season_to_diff = OrderedDict()
    for vname in varnames:
        finf = FieldInfo(vname, start_year, end_year)

        era = analysis.get_annual_maxima(path_to_hdf_file=path_to_era_driven,
                                         start_year=start_year, end_year=end_year,
                                         var_name=vname, level=level_index_dict[vname])

        gcm = analysis.get_annual_maxima(path_to_hdf_file=path_to_gcm_driven,
                                         start_year=start_year, end_year=end_year,
                                         var_name=vname, level=level_index_dict[vname])

        finfo_to_season_to_diff[finf] = {
            "Annual maximum": np.mean(list(gcm.values()), axis=0) - np.mean(list(era.values()), axis=0)
        }

    return finfo_to_season_to_diff


def main():
    path_to_era_driven = "/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5"
    path_to_gcm_driven = \
        "/RESCUE/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"

    # Year range ...
    start_year = 1980
    end_year = 2010

    lons, lats, bmp = analysis.get_basemap_from_hdf(file_path=path_to_era_driven)
    bmp_info = BasemapInfo(lons=lons, lats=lats, bmp=bmp)

    seasons_for_mean = OrderedDict([
        ("Winter", (12, 1, 2)),
        ("Spring", range(3, 6)),
        ("Summer", range(6, 9)),
        ("Fall", range(9, 12))
    ])

    seasons_for_max = OrderedDict(
        [("Annual maximum", range(1, 13)), ]
    )

    variables_mean_bfe = ["TT", "PR", "I5", "STFL"]

    variables_annual_max_bfe = ["I5", ]
    variables_annual_max_bfe.pop()

    level_index_dict = collections.defaultdict(lambda: 0)
    level_index_dict.update({
        "I5": 0,
        "TT": 0,
        "PR": 0
    })

    all_vars = variables_mean_bfe + variables_annual_max_bfe

    plot_utils.apply_plot_params(font_size=12, width_pt=None, width_cm=25, height_cm=25)
    fig = plt.figure()

    nrows = len(all_vars)

    ncols = max(len(seasons_for_mean) + 1, len(seasons_for_max) + 1)
    gs = GridSpec(nrows, ncols,
                  width_ratios=[1., ] * (ncols - 1) + [0.05, ])

    # Plot seasonal mean precipitation and temperature
    finfo_to_season_to_diff = get_bfe_in_seasonal_mean(variables_mean_bfe, level_index_dict=level_index_dict,
                                                       season_to_months=seasons_for_mean,
                                                       path_to_era_driven=path_to_era_driven,
                                                       path_to_gcm_driven=path_to_gcm_driven,
                                                       start_year=start_year, end_year=end_year)

    row = 0
    for vname in variables_mean_bfe:
        row_axes = [fig.add_subplot(gs[row, col]) for col in range(len(seasons_for_mean) + 1)]

        plot_bfe_row_for_var(finfo_to_season_to_diff=finfo_to_season_to_diff,
                             ax_list=row_axes, season_titles=row == 0, varname=vname,
                             basemap_info=bmp_info)

        row += 1

    # plot annual maximum swe
    finfo_to_season_to_diff = get_bfe_in_annual_max(variables_annual_max_bfe, level_index_dict=level_index_dict,
                                                    path_to_era_driven=path_to_era_driven,
                                                    path_to_gcm_driven=path_to_gcm_driven,
                                                    start_year=start_year, end_year=end_year)

    add_season_titles = True
    for vname in variables_annual_max_bfe:
        row_axes = [fig.add_subplot(gs[row, col]) for col in range(ncols)]

        for i, the_ax in enumerate(row_axes[:-1]):
            if i < ncols - len(seasons_for_max) - 1:
                the_ax.set_visible(False)

        row_axes = list(reversed(row_axes[:-1])) + [row_axes[-1], ]

        plot_bfe_row_for_var(finfo_to_season_to_diff=finfo_to_season_to_diff,
                             ax_list=row_axes, season_titles=add_season_titles, varname=vname,
                             basemap_info=bmp_info)

        add_season_titles = False
        row += 1


    # Save the figure to a file
    p = Path(img_folder)
    if not p.is_dir():
        p.mkdir(parents=True)

    img_name = "{}_{}-{}_{}.png".format("-".join(variables_annual_max_bfe + variables_mean_bfe),
                                        start_year, end_year, "-".join(seasons_for_mean.keys()))

    img_path = str(p.joinpath(img_name))
    # fig.tight_layout()
    fig.savefig(img_path, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()
    main()
