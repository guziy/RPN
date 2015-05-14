from collections import OrderedDict
import os
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.basemap import maskoceans
from pathlib import Path
from scipy.stats import ttest_ind
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
        ("Spring", list(range(3, 6))),
        ("Summer", list(range(6, 9))),
        ("Fall", list(range(9, 12))),
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
    for season, months in season_to_months.items():
        season_to_field[season] = analysis.get_mean_2d_fields_for_months(
            path=sim_config.data_path, var_name=var_name, level=level, months=months,
            start_year=sim_config.start_year, end_year=sim_config.end_year)

    return season_to_field


def _plot_row(vname="", level=0, config_dict=None):

    bmp, lons, lats = config_dict.basemap, config_dict.lons, config_dict.lats
    xx, yy = bmp(lons, lats)
    lons[lons > 180] -= 360


    fig = config_dict.fig
    gs = config_dict.gs
    label_base = config_dict.label_base
    label_modif = config_dict.label_modif

    the_row = config_dict.the_row
    season_to_months = config_dict.season_to_months


    if "+" in vname or "-" in vname:
        op = "+" if "+" in vname else "-"
        vname1, vname2 = vname.split(op)
        
        vname1 = vname1.strip()
        vname2 = vname2.strip()
               
        current_base = {}
        future_base = {}
        
        current_modif = {}
        future_modif = {}

        # vname1 
        current_base1 = compute_seasonal_means_for_each_year(config_dict["Current"][label_base], var_name=vname1, level=level,
                                                            season_to_months=season_to_months)
        future_base1 = compute_seasonal_means_for_each_year(config_dict["Future"][label_base], var_name=vname1, level=level,
                                                           season_to_months=season_to_months)

        current_modif1 = compute_seasonal_means_for_each_year(config_dict["Current"][label_modif], var_name=vname1,
                                                             level=level,
                                                             season_to_months=season_to_months)
        future_modif1 = compute_seasonal_means_for_each_year(config_dict["Future"][label_modif], var_name=vname1, level=level,
                                                            season_to_months=season_to_months)

       
        # vname2
        current_base2 = compute_seasonal_means_for_each_year(config_dict["Current"][label_base], var_name=vname2, level=level,
                                                            season_to_months=season_to_months)
        future_base2 = compute_seasonal_means_for_each_year(config_dict["Future"][label_base], var_name=vname2, level=level,
                                                           season_to_months=season_to_months)

        current_modif2 = compute_seasonal_means_for_each_year(config_dict["Current"][label_modif], var_name=vname2,
                                                             level=level,
                                                             season_to_months=season_to_months)
        future_modif2 = compute_seasonal_means_for_each_year(config_dict["Future"][label_modif], var_name=vname2, level=level,
                                                            season_to_months=season_to_months)

        
        for season in current_base1:
            current_base[season] = eval("current_base2[season]{}current_base1[season]".format(op))
            future_base[season] = eval("future_base2[season]{}future_base1[season]".format(op))
            current_modif[season] = eval("current_modif2[season]{}current_modif1[season]".format(op))
            future_modif[season] = eval("future_modif2[season]{}future_modif1[season]".format(op))
 

    else:
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

    season_to_plot_diff = OrderedDict()

    diff_max = 0
    print(list(current_base.keys()))
    # Get the ranges for colorbar and calculate p-values
    season_to_pvalue = OrderedDict()
    for season in list(current_base.keys()):
        season_to_diff[season] = (future_modif[season] - current_modif[season]) - \
                                 (future_base[season] - current_base[season])


        _, pvalue_current = ttest_ind(current_modif[season], current_base[season], axis=0, equal_var=False)
        _, pvalue_future = ttest_ind(future_modif[season], current_base[season], axis=0, equal_var=False)


        season_to_pvalue[season] = np.minimum(pvalue_current, pvalue_future)

        # Convert units if required
        if vname in config_dict.multipliers:
            season_to_diff[season] *= config_dict.multipliers[vname]

        field_to_plot = infovar.get_to_plot(vname, season_to_diff[season].mean(axis=0), lons=lons, lats=lats)
        season_to_plot_diff[season] = field_to_plot

        if hasattr(field_to_plot, "mask"):
            diff_max = max(np.percentile(np.abs(field_to_plot[~field_to_plot.mask]), 95), diff_max)
        else:
            diff_max = max(np.percentile(np.abs(field_to_plot), 95), diff_max)


    img = None
    locator = MaxNLocator(nbins=10, symmetric=True)
    clevels = locator.tick_values(-diff_max, diff_max)

    bn = BoundaryNorm(clevels, len(clevels) - 1)
    cmap = cm.get_cmap("RdBu_r", len(clevels) - 1)
    for col, season in enumerate(current_base.keys()):
        ax = fig.add_subplot(gs[the_row, col])

        if not col:
            ax.set_ylabel(infovar.get_display_label_for_var(vname))

        if not the_row:
            ax.set_title(season)

        img = bmp.pcolormesh(xx, yy, season_to_plot_diff[season].copy(), vmin=-diff_max, vmax=diff_max,
                             cmap=cmap, norm=bn)

        p = season_to_pvalue[season]
        if hasattr(season_to_plot_diff[season], "mask"):
            p = np.ma.masked_where(season_to_plot_diff[season].mask, p)


        cs = bmp.contourf(xx, yy, p, hatches=["//"], levels=[0.1, 1], colors='none')
        # create a legend for the contour set
        # artists, labels = cs.legend_elements()
        # ax.legend(artists, labels, handleheight=2)


        bmp.drawcoastlines(ax=ax, linewidth=0.4)

    cb = plt.colorbar(img, cax=fig.add_subplot(gs[the_row, len(current_base)]))

    if hasattr(config_dict, "name_to_units") and vname in config_dict.name_to_units:
        cb.ax.set_title(config_dict.name_to_units[vname])
    else:
        cb.ax.set_title(infovar.get_units(vname))


def main_interflow():

    # Changes global plot properties mainly figure size and font size
    plot_utils.apply_plot_params(font_size=12, width_cm=20)

    # season_to_months = get_default_season_to_months_dict()
    season_to_months = OrderedDict([("January", [1, ]), ("February", [2, ]), ("March", [3, ]), ])

    var_names = ["TT", "HU", "PR", "AV", "TRAF", "I1", "STFL"]

    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=20 * len(season_to_months) / 4.0,
                                 height_cm=20 * len(var_names) / 5.0)

    levels = [0, 0, 0, 0, 0, 0, 0]
    multipliers = {
        "PR": 1.,
        "TRAF": 1.,
        "I1": infovar.soil_layer_widths_26_to_60[0] * 1000.0,
        "TRAF+TDRA": 24 * 60 * 60 
    }

    name_to_units = {
        "TRAF": "mm/day", "I1": "mm", "PR": "mm/day", "TRAF+TDRA": "mm/day"
    }

    base_current_path = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/" \
                        "quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"
    base_label = "CRCM5-L"

    modif_current_path = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/" \
                         "quebec_0.1_crcm5-hcd-rl-intfl-cc-canesm2-1980-2010.hdf5"
    modif_label = "CRCM5-LI"

    start_year_c = 1980
    end_year_c = 2010

    future_shift_years = 75

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
    config_dict.name_to_units = name_to_units

    # Calculate and plot seasonal means
    for vname, level, the_row in zip(var_names, levels, list(range(len(levels)))):
        config_dict.the_row = the_row

        _plot_row(vname=vname, level=level, config_dict=config_dict)

    # Save the image to the file
    img_path = get_image_path(base_config_c, base_config_f, modif_config_c, season_to_months=season_to_months)
    fig.savefig(img_path, bbox_inches="tight")


def get_image_path(base_config_c, base_config_f, modif_config_c, season_to_months=None):
    img_folder = Path("cc_paper").joinpath("{}_vs_{}".format(modif_config_c.label, base_config_c.label))
    if not img_folder.exists():
        img_folder.mkdir(parents=True)

    img_name = "{}_{}-{}_{}-{}.png".format(base_config_f.start_year, base_config_f.end_year,
                                           base_config_c.start_year, base_config_c.end_year,
                                           "-".join(list(season_to_months.keys())))
    return str(img_folder.joinpath(img_name))


def main():
    season_to_months = get_default_season_to_months_dict()

    # var_names = ["TT", "HU", "PR", "AV", "STFL"]
    # var_names = ["TRAF", "STFL", "TRAF+TDRA"]
    # var_names = ["TT", "PR", "STFL"]
    var_names = ["TT", "PR", "STFL", ]
    levels = [0, 0, 0, 0, 0]
    multipliers = {
        "PR": 1.0, 
        "TRAF+TDRA": 24 * 60 * 60 
    }
    name_to_units = {
        "TRAF": "mm/day", "I1": "mm", "PR": "mm/day", "TRAF+TDRA": "mm/day"
    }


    base_current_path = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/" \
                        "quebec_0.1_crcm5-r-cc-canesm2-1980-2010.hdf5"
    base_label = "CRCM5-NL"

    modif_current_path = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/" \
                         "quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"
    modif_label = "CRCM5-L"

    start_year_c = 1980
    end_year_c = 2010

    future_shift_years = 75

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

    # Changes global plot properties mainly figure size and font size
    plot_utils.apply_plot_params(font_size=12, width_cm=25, height_cm=15)


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

    config_dict.name_to_units = name_to_units

    # Calculate and plot seasonal means
    for vname, level, the_row in zip(var_names, levels, list(range(len(levels)))):
        config_dict.the_row = the_row

        _plot_row(vname=vname, level=level, config_dict=config_dict)

    # Save the image to the file
    img_path = get_image_path(base_config_c, base_config_f, modif_config_c, season_to_months=season_to_months)
    fig.savefig(img_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()

    main()
    # main_interflow()
