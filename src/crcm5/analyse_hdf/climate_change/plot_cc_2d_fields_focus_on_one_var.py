from collections import OrderedDict
import os
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, LogLocator
from mpl_toolkits.basemap import maskoceans
from scipy.stats import ttest_ind
from crcm5 import infovar
from crcm5.analyse_hdf.run_config import RunConfig
import matplotlib.pyplot as plt
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis
from crcm5.analyse_hdf.climate_change.plot_cc_2d_fields import compute_seasonal_means_for_each_year, \
    get_default_season_to_months_dict
from util import plot_utils
import numpy as np


__author__ = 'huziy'

"""
A figure for a variable

            (current)   (future)    (future-current)

(sim1)          c1           f1          f1-c1

(sim2)          c2           f2          f2-c2

(sim2-sim1)     c2-c1       f2-f1        (f2-f1) - (c2-c1) ==  (f2-c2) - (f1-c1)

"""

multiplier_dict = {
    "PR": 24 * 3600 * 1000
}


def get_data(vname="", level=0, config_dict=None):
    modif_label = config_dict.label_modif
    base_label = config_dict.label_base

    base_config_c, modif_config_c = [config_dict["Current"][the_label] for the_label in [base_label, modif_label]]
    base_config_f, modif_config_f = [config_dict["Future"][the_label] for the_label in [base_label, modif_label]]

    current_base = compute_seasonal_means_for_each_year(base_config_c, var_name=vname, level=level,
                                                        season_to_months=config_dict.season_to_months)

    future_base = compute_seasonal_means_for_each_year(base_config_f, var_name=vname, level=level,
                                                       season_to_months=config_dict.season_to_months)

    current_modif = compute_seasonal_means_for_each_year(modif_config_c, var_name=vname,
                                                         level=level,
                                                         season_to_months=config_dict.season_to_months)

    future_modif = compute_seasonal_means_for_each_year(modif_config_f, var_name=vname, level=level,
                                                        season_to_months=config_dict.season_to_months)

    return OrderedDict([(base_config_c, current_base), (base_config_f, future_base),
                        (modif_config_c, current_modif), (modif_config_f, future_modif)])


def _plot_var(vname="", level=0, config_dict=None, data_dict=None):
    modif_label = config_dict.label_modif
    base_label = config_dict.label_base

    base_config_c, modif_config_c = [config_dict["Current"][the_label] for the_label in [base_label, modif_label]]
    base_config_f, modif_config_f = [config_dict["Future"][the_label] for the_label in [base_label, modif_label]]

    bmp = config_dict.basemap
    lons = config_dict.lons
    lons[lons > 180] -= 360

    lats = config_dict.lats
    xx, yy = bmp(lons, lats)

    i_to_label = {
        0: base_label,
        1: modif_label,
        2: "{} - {}".format(modif_label, base_label)
    }

    j_to_title = {
        0: "Current ({}-{})".format(base_config_c.start_year, base_config_c.end_year),
        1: "Future ({}-{})".format(base_config_f.start_year, base_config_f.end_year),
        2: "Future - Current"
    }

    # Create a folder for images
    img_folder = os.path.join("cc_paper", "{}_vs_{}".format(modif_label, base_label))
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)

    nrows = ncols = 3
    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=27, height_cm=20)

    field_cmap = cm.get_cmap("jet", 10)
    diff_cmap = cm.get_cmap("RdBu_r", 10)

    for season in list(data_dict[base_config_c].keys()):

        fig = plt.figure()
        fig.suptitle(
            "{} ({})".format(infovar.get_long_display_label_for_var(vname), season),
            font_properties=FontProperties(weight="bold"))

        gs = GridSpec(nrows=nrows, ncols=ncols)

        mean_c_base = data_dict[base_config_c][season].mean(axis=0)
        mean_f_base = data_dict[base_config_f][season].mean(axis=0)


        mean_c_modif = data_dict[modif_config_c][season].mean(axis=0)
        mean_f_modif = data_dict[modif_config_f][season].mean(axis=0)


        print("------------ {}, {} --------------".format(vname, season))
        print("data_dict[base_config_c][season].shape = {}".format(data_dict[base_config_c][season].shape))
        _, p_signif_base_cc = ttest_ind(data_dict[base_config_c][season], data_dict[base_config_f][season], axis=0, equal_var=False)
        _, p_signif_current = ttest_ind(data_dict[base_config_c][season], data_dict[modif_config_c][season], axis=0, equal_var=False)

        print("Base p-value ranges: {} to {}".format(p_signif_base_cc.min(), p_signif_base_cc.max()))
        _, p_signif_modif_cc = ttest_ind(data_dict[modif_config_c][season], data_dict[modif_config_f][season], axis=0, equal_var=False)
        _, p_signif_future = ttest_ind(data_dict[base_config_f][season], data_dict[modif_config_f][season], axis=0, equal_var=False)

        ij_to_pvalue = {
            (0, 2): p_signif_base_cc,
            (1, 2): p_signif_modif_cc,
            (2, 0): p_signif_current,
            (2, 1): p_signif_future,
            (2, 2): np.minimum(p_signif_current, p_signif_future)
        }

        ij_to_data = {
            (0, 0): mean_c_base, (1, 0): mean_c_modif,
            (0, 1): mean_f_base, (1, 1): mean_f_modif
        }

        # Apply multipliers for unit conversion
        for k in list(ij_to_data.keys()):
            ij_to_data[k] *= multiplier_dict.get(vname, 1)

            # mask oceans
            mask_inland = vname in ["STFL", "STFA"] or vname.startswith("I")
            ij_to_data[k] = maskoceans(lonsin=lons, latsin=lats, datain=ij_to_data[k], inlands=mask_inland)

        # get all means to calculate the ranges of mean fields
        all_means = []
        for vals in ij_to_data.values():
            all_means.extend(vals[~vals.mask])

        if vname == "PR":
            # mm/day
            minval = 0
            maxval = 5
            extend_val = "max"
        else:
            minval = np.percentile(all_means, 1)
            maxval = np.percentile(all_means, 95)
            extend_val = "neither"


        d1 = ij_to_data[0, 1] - ij_to_data[0, 0]
        d2 = ij_to_data[1, 1] - ij_to_data[1, 0]
        all_cc_diffs = np.ma.asarray([d1, d2])
        mindiff_cc = np.percentile(all_cc_diffs[~all_cc_diffs.mask], 1)
        maxdiff_cc = np.percentile(all_cc_diffs[~all_cc_diffs.mask], 99)

        maxdiff_cc = max(np.ma.abs(maxdiff_cc), np.ma.abs(mindiff_cc))
        mindiff_cc = -maxdiff_cc


        # Add all differences due to changed processes in a list
        all_proc_diffs = []

        d1 = ij_to_data[1, 0] - ij_to_data[0, 0]
        d2 = ij_to_data[1, 1] - ij_to_data[0, 1]
        all_proc_diffs.extend([d1, d2])
        all_proc_diffs.append(d2 - d1)
        all_proc_diffs = np.ma.asarray(all_proc_diffs)


        mindiff_proc = np.percentile(all_proc_diffs[~all_proc_diffs.mask], 1)
        maxdiff_proc = np.percentile(all_proc_diffs[~all_proc_diffs.mask], 99)

        maxdiff_proc = max(np.abs(maxdiff_proc), np.abs(mindiff_proc))
        mindiff_proc = -maxdiff_proc

        plot_data = []
        for row in range(nrows - 1):
            row_data = []
            for col in range(ncols - 1):
                row_data.append(ij_to_data[row, col])
            plot_data.append(row_data + [0, ])

        # calculate differences between different model configurations
        row = 2
        plot_data.append([0, 0, 0])
        for col in range(2):
            plot_data[row][col] = plot_data[1][col] - plot_data[0][col]

        # calculate CC
        col = 2
        for row in range(3):
            plot_data[row][col] = plot_data[row][1] - plot_data[row][0]


        # Do the plotting
        for i in range(nrows):
            for j in range(ncols):

                ax = fig.add_subplot(gs[i, j])
                plotting_values = (i < 2) and (j < 2)

                if plotting_values:
                    the_min, the_max = minval, maxval
                elif i == nrows - 1:
                    the_min, the_max = mindiff_proc, maxdiff_proc
                else:
                    the_min, the_max = mindiff_cc, maxdiff_cc

                the_cmap = field_cmap if plotting_values else diff_cmap

                locator = MaxNLocator(nbins=the_cmap.N, symmetric=not plotting_values)
                bounds = locator.tick_values(the_min, the_max)

                if vname in ["STFA", "STFL"] and plotting_values:
                    the_min = 1.0e-5
                    the_max = 1.0e4

                    locator = LogLocator(numticks=11)
                    bounds = locator.tick_values(the_min, the_max)

                    print(bounds)

                    norm = LogNorm(vmin=bounds[0], vmax=bounds[-1])

                    plot_data[i][j] = np.ma.masked_where(plot_data[i][j] <= the_min, plot_data[i][j])
                    print("the_max = {}".format(the_max))

                    the_cmap = cm.get_cmap("jet", len(bounds) - 1)

                else:
                    norm = BoundaryNorm(bounds, the_cmap.N)

                extend = "both" if not plotting_values else extend_val

                if plotting_values:
                    print(i, j, the_min, the_max)

                im = bmp.pcolormesh(xx, yy, plot_data[i][j][:, :], cmap=the_cmap, norm=norm)

                if (i, j) in ij_to_pvalue:
                    p = ij_to_pvalue[i, j]
                    # not_signif = (p > 0.2)
                    # not_signif = np.ma.masked_where(~not_signif, not_signif) * 0.75
                    # not_signif = np.ma.masked_where(plot_data[i][j].mask, not_signif)
                    # bmp.pcolormesh(xx, yy, not_signif, vmin=0, vmax=1, cmap=cm.get_cmap("gray", 20))
                    p = np.ma.masked_where(plot_data[i][j].mask, p)
                    cs = bmp.contourf(xx, yy, p, hatches=["//"], levels=[0.1, 1], colors='none')
                    # create a legend for the contour set
                    # artists, labels = cs.legend_elements()
                    # ax.legend(artists, labels, handleheight=2)


                bmp.colorbar(im, ticks=bounds, extend=extend)
                bmp.drawcoastlines(ax=ax, linewidth=0.5)

                ax.set_title(j_to_title.get(j, "") if i == 0 else "")
                ax.set_ylabel(i_to_label.get(i, "") if j == 0 else "")

        # Save the image to the file
        img_name = "{}_{}_{}-{}_{}-{}.png".format(
            vname + str(level), season,
            base_config_f.start_year, base_config_f.end_year,
            base_config_c.start_year, base_config_c.end_year)

        img_path = os.path.join(img_folder, img_name)
        fig.savefig(img_path, bbox_inches="tight")
        plt.close(fig)


def main():
    """

    """

    season_to_months = get_default_season_to_months_dict()

    # season_to_months = OrderedDict([("April", [4, ]), ("May", [5, ]), ("June", [6, ]), ("July", [7, ])])

    var_names = ["TT", "HU", "PR", "AV", "AH", "STFL", "TRAF", "I5", "I0"]

    # var_names = ["TT", "PR"]

    levels = [0, ] * len(var_names)
    multipliers = {
        "PR": 1.0e3 * 24.0 * 3600.,
        "TRAF": 24 * 3600.0
    }

    base_current_path = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/" \
                        "quebec_0.1_crcm5-r-cc-canesm2-1980-2010.hdf5"
    base_label = "CRCM5-NL"

    modif_current_path = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/" \
                         "quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"
    modif_label = "CRCM5-L"

    # base_current_path = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/" \
    # "quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"
    # base_label = "CRCM5-L"
    #
    # modif_current_path = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/" \
    # "quebec_0.1_crcm5-hcd-rl-intfl-cc-canesm2-1980-2010.hdf5"
    # modif_label = "CRCM5-LI"

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
    config_dict.label_modif = modif_config_c.label
    config_dict.label_base = base_config_c.label
    config_dict.season_to_months = season_to_months
    config_dict.multipliers = multipliers

    lons, lats, bmp = analysis.get_basemap_from_hdf(base_current_path)
    config_dict.lons = lons
    config_dict.lats = lats
    config_dict.basemap = bmp

    # Calculate and plot seasonal means
    for vname, level in zip(var_names, levels):
        data = get_data(vname=vname, level=level, config_dict=config_dict)
        _plot_var(vname=vname, level=level, config_dict=config_dict, data_dict=data)


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()

    main()


