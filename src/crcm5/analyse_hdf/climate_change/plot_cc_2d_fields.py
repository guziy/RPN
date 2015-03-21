from collections import OrderedDict
import os
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.basemap import maskoceans
from crcm5 import infovar
from crcm5.analyse_hdf.run_config import RunConfig
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis

import numpy as np
import matplotlib.pyplot as plt
from util import plot_utils

__author__ = 'huziy'

# Plot seasonal mean fields


def get_default_season_to_months_dict():
    return OrderedDict([
        ("Winter", (1, 2, 12)),
        ("Spring", range(3, 6)),
        ("Summer", range(6, 9)),
        ("Fall", range(9, 12)),
    ])


def compute_seasonal_means_for_each_year(sim_config, season_to_months=None, var_name="", level=0):
    """
    For each year so the significance test could be applied
    :param sim_config:
    :param season_to_months:
    :return OrderedDict(season name: mean field for the season)
    """

    assert isinstance(sim_config, RunConfig)

    if season_to_months is None:
        season_to_months = get_default_season_to_months_dict()

    season_to_field = OrderedDict()
    for season, months in season_to_months.iteritems():
        season_to_field[season] = analysis.get_mean_2d_fields_for_months(
            path=sim_config.data_path, var_name=var_name, level=level, months=months,
            start_year=sim_config.start_year, end_year=sim_config.end_year
        )

    return season_to_field


def _plot_row(vname="", level=0, config_dict=None):
    fig = config_dict.fig
    gs = config_dict.gs
    label_base = config_dict.label_base
    label_modif = config_dict.label_modif

    the_row = config_dict.the_row
    season_to_months = config_dict.season_to_months

    current_base = compute_seasonal_means_for_each_year(config_dict["Current"][label_base], var_name=vname, level=level,
                                                        season_to_months=season_to_months)
    future_base = compute_seasonal_means_for_each_year(config_dict["Future"][label_base], var_name=vname, level=level,
                                                       season_to_months=season_to_months)

    current_modif = compute_seasonal_means_for_each_year(config_dict["Current"][label_modif], var_name=vname,
                                                         level=level,
                                                         season_to_months=season_to_months)
    future_modif = compute_seasonal_means_for_each_year(config_dict["Future"][label_modif], var_name=vname, level=level,
                                                        season_to_months=season_to_months)

    # Calculate the differences in cc signal
    season_to_diff = OrderedDict()

    diff_max = 0
    print current_base.keys()
    for season in current_base.keys():
        season_to_diff[season] = (future_modif[season] - current_modif[season]) - \
                                 (future_base[season] - current_base[season])
        # Convert units if required
        if vname in config_dict.multipliers:
            season_to_diff[season] *= config_dict.multipliers[vname]

        diff_max = max(np.percentile(np.abs(season_to_diff[season].mean(axis=0)), 95), diff_max)

    bmp, lons, lats = config_dict.basemap, config_dict.lons, config_dict.lats
    xx, yy = bmp(lons, lats)

    img = None
    locator = MaxNLocator(nbins=10, symmetric=True)
    clevels = locator.tick_values(-diff_max, diff_max)

    bn = BoundaryNorm(clevels, len(clevels) - 1)
    cmap = cm.get_cmap("RdBu_r", len(clevels) - 1)
    for col, season in enumerate(current_base.keys()):
        ax = fig.add_subplot(gs[the_row, col])

        if not col:
            ax.set_ylabel(vname)

        if not the_row:
            ax.set_title(season)

        to_plot = season_to_diff[season].mean(axis=0)

        # Mask oceans
        lons[lons > 180] -= 360
        to_plot = maskoceans(lons, lats, to_plot)

        img = bmp.pcolormesh(xx, yy, to_plot[:, :], vmin=-diff_max, vmax=diff_max,
                             cmap=cmap, norm=bn)

        bmp.drawcoastlines(ax=ax, linewidth=0.4)

    plt.colorbar(img, cax=fig.add_subplot(gs[the_row, len(current_base)]))


def main_interflow():
    # season_to_months = _get_default_season_to_months_dict()
    season_to_months = OrderedDict([("April", [4, ]), ("May", [5, ]), ("June", [6, ])])

    var_names = ["TT", "HU", "PR", "AV", "STFA", "TRAF", "I1"]

    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=20 * len(season_to_months) / 4.0, height_cm=20 * len(var_names) / 5.0)

    levels = [0, 0, 0, 0, 0, 0, 0]
    multipliers = {
        "PR": 1.0e3 * 24.0 * 3600.,
        "TRAF": 24.0 * 3600.,
        "I1": 1000 * infovar.soil_layer_widths_26_to_60[0]
    }

    base_current_path = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/" \
                        "quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"
    base_label = "CRCM5-L2"

    modif_current_path = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/" \
                         "quebec_0.1_crcm5-hcd-rl-intfl-cc-canesm2-1980-2010.hdf5"
    modif_label = "CRCM5-L2I"

    start_year_c = 1980
    end_year_c = 2010

    future_shift_years = 90

    params = dict(
        data_path=base_current_path, start_year=start_year_c, end_year=end_year_c, label=base_label
    )

    base_config_c = RunConfig(**params)
    base_config_f = base_config_c.get_shifted_config(future_shift_years)

    params.update(
        dict(data_path=modif_current_path, label=modif_label)
    )

    modif_config_c = RunConfig(**params)
    modif_config_f = modif_config_c.get_shifted_config(future_shift_years)

    config_dict = OrderedDict([
        ("Current", OrderedDict([(base_label, base_config_c), (modif_label, modif_config_c)])),
        ("Future", OrderedDict([(base_label, base_config_f), (modif_label, modif_config_f)]))
    ])




    # Plot the differences
    fig = plt.figure()
    gs = GridSpec(len(var_names), len(season_to_months) + 1, width_ratios=[1., ] * len(season_to_months) + [0.05, ])

    config_dict.fig = fig
    config_dict.gs = gs
    config_dict.label_modif = modif_config_c.label
    config_dict.label_base = base_config_c.label
    config_dict.season_to_months = season_to_months
    config_dict.multipliers = multipliers

    lons, lats, bmp = analysis.get_basemap_from_hdf(base_current_path)
    config_dict.lons = lons
    config_dict.lats = lats
    config_dict.basemap = bmp

    # Calculate and plot seasonal means
    for vname, level, the_row in zip(var_names, levels, range(len(levels))):
        config_dict.the_row = the_row

        _plot_row(vname=vname, level=level, config_dict=config_dict)

    # Save the image to the file
    img_folder = os.path.join("cc_paper", "{}_vs_{}".format(modif_label, base_label))
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)

    img_name = "{}_{}-{}_{}-{}.png".format(base_config_f.start_year, base_config_f.end_year,
                                           base_config_c.start_year, base_config_c.end_year,
                                           "-".join(season_to_months.keys()))

    img_path = os.path.join(img_folder, img_name)
    fig.savefig(img_path, bbox_inches="tight")


def main():
    season_to_months = get_default_season_to_months_dict()

    var_names = ["TT", "HU", "PR", "AV", "STFA"]
    levels = [0, 0, 0, 0, 0]
    multipliers = {
        "PR": 1.0e3 * 24.0 * 3600.
    }

    base_current_path = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/" \
                        "quebec_0.1_crcm5-r-cc-canesm2-1980-2010.hdf5"
    base_label = "CRCM5-NL"

    modif_current_path = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/" \
                         "quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"
    modif_label = "CRCM5-L2"

    start_year_c = 1980
    end_year_c = 2010

    future_shift_years = 90

    params = dict(
        data_path=base_current_path, start_year=start_year_c, end_year=end_year_c, label=base_label
    )

    base_config_c = RunConfig(**params)
    base_config_f = base_config_c.get_shifted_config(future_shift_years)

    params.update(
        dict(data_path=modif_current_path, label=modif_label)
    )

    modif_config_c = RunConfig(**params)
    modif_config_f = modif_config_c.get_shifted_config(future_shift_years)

    config_dict = OrderedDict([
        ("Current", OrderedDict([(base_label, base_config_c), (modif_label, modif_config_c)])),
        ("Future", OrderedDict([(base_label, base_config_f), (modif_label, modif_config_f)]))
    ])




    # Plot the differences
    fig = plt.figure()
    gs = GridSpec(len(var_names), len(season_to_months) + 1, width_ratios=[1., ] * len(season_to_months) + [0.05, ])

    config_dict.fig = fig
    config_dict.gs = gs
    config_dict.label_modif = modif_config_c.label
    config_dict.label_base = base_config_c.label
    config_dict.season_to_months = season_to_months
    config_dict.multipliers = multipliers

    lons, lats, bmp = analysis.get_basemap_from_hdf(base_current_path)
    config_dict.lons = lons
    config_dict.lats = lats
    config_dict.basemap = bmp

    # Calculate and plot seasonal means
    for vname, level, the_row in zip(var_names, levels, range(len(levels))):
        config_dict.the_row = the_row

        _plot_row(vname=vname, level=level, config_dict=config_dict)

    # Save the image to the file
    img_folder = os.path.join("cc_paper", "{}_vs_{}".format(modif_label, base_label))
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)

    img_name = "{}_{}-{}_{}-{}.png".format(base_config_f.start_year, base_config_f.end_year,
                                           base_config_c.start_year, base_config_c.end_year,
                                           "-".join(season_to_months.keys()))

    img_path = os.path.join(img_folder, img_name)
    fig.savefig(img_path, bbox_inches="tight")


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()

    # main()
    main_interflow()