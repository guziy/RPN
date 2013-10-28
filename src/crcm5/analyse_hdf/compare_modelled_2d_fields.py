from collections import OrderedDict
from datetime import datetime
import os

from mpl_toolkits.basemap import maskoceans
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from crcm5 import infovar
import common_plot_params as cpp


__author__ = 'huziy'

import do_analysis_using_pytables as analysis
import matplotlib.pyplot as plt
import numpy as np
import brewer2mpl
import common_plot_params


images_folder = "/home/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/images_for_lake-river_paper"
cache_folder = os.path.join(images_folder, "cache")


def compare_annual_mean_fields(paths=None, labels=None, varnames=None):
    compare(paths=paths, varnames=varnames, labels=labels, months_of_interest=range(1, 13))


def _offset_multiplier(colorbar):
    ax = colorbar.ax

    title = ax.get_title()
    ax.set_title("{0}\n\n\n\n".format(title))


def compare(paths=None, path_to_control_data=None, control_label="",
            labels=None, varnames=None, levels=None, months_of_interest=None,
            start_year=None, end_year=None):
    """
    Comparing 2D fields
    :param paths: paths to the simulation results
    :param varnames:
    :param labels: Display name for each simulation (number of labels should
     be equal to the number of paths)
    :param path_to_control_data: the path with which the comparison done i.e. a in the following
     formula
            delta = (x - a)/a * 100%

     generates one image file per variable (in the folder images_for_lake-river_paper):
        compare_varname_<control_label>_<label1>_..._<labeln>_startyear_endyear.png

    """
    #get coordinate data  (assumes that all the variables and runs have the same coordinates)
    lons2d, lats2d, basemap = analysis.get_basemap_from_hdf(file_path=path_to_control_data)
    x, y = basemap(lons2d, lats2d)

    lake_fraction = analysis.get_array_from_file(path=path_to_control_data, var_name="lake_fraction")

    if lake_fraction is None:
        lake_fraction = np.zeros(lons2d.shape)

    ncolors = 10
    # +1 to include white
    diff_cmap = brewer2mpl.get_map("RdBu", "diverging", ncolors + 1, reverse=True).get_mpl_colormap(N=ncolors + 1)

    for var_name, level in zip(varnames, levels):
        sfmt = infovar.get_colorbar_formatter(var_name)
        control_means = analysis.get_mean_2d_fields_for_months(path=path_to_control_data, var_name=var_name,
                                                               months=months_of_interest,
                                                               start_year=start_year, end_year=end_year,
                                                               level=level)

        control_mean = np.mean(control_means, axis=0)
        fig = plt.figure()
        assert isinstance(fig, Figure)
        gs = gridspec.GridSpec(2, len(paths) + 1, wspace=0.5)

        #plot the control
        ax = fig.add_subplot(gs[0, 0])
        assert isinstance(ax, Axes)
        ax.set_title("{0}".format(control_label))
        ax.set_ylabel("Mean: $X_{0}$")
        to_plot = infovar.get_to_plot(var_name, control_mean,
                                      lake_fraction=lake_fraction, mask_oceans=True, lons=lons2d, lats=lats2d)
        #determine colorabr extent and spacing
        field_cmap, field_norm = infovar.get_colormap_and_norm_for(var_name, to_plot, ncolors=ncolors)

        basemap.pcolormesh(x, y, to_plot, cmap=field_cmap, norm=field_norm)
        cb = basemap.colorbar(format=sfmt)

        assert isinstance(cb, Colorbar)
        #cb.ax.set_ylabel(infovar.get_units(var_name))
        units = infovar.get_units(var_name)

        info = "Variable:" \
               "\n{0}" \
               "\nPeriod: {1}-{2}" \
               "\nMonths: {3}" \
               "\nUnits: {4}"

        info = info.format(infovar.get_long_name(var_name), start_year, end_year,
                           ",".join([datetime(2001, m, 1).strftime("%b") for m in months_of_interest]), units)

        ax.annotate(info, xy=(0.1, 0.3), xycoords="figure fraction")

        sel_axes = [ax]

        for the_path, the_label, column in zip(paths, labels, range(1, len(paths) + 1)):

            means_for_years = analysis.get_mean_2d_fields_for_months(path=the_path, var_name=var_name,
                                                                     months=months_of_interest,
                                                                     start_year=start_year, end_year=end_year)
            the_mean = np.mean(means_for_years, axis=0)

            #plot the mean value
            ax = fig.add_subplot(gs[0, column])
            sel_axes.append(ax)
            ax.set_title("{0}".format(the_label))
            to_plot = infovar.get_to_plot(var_name, the_mean, lake_fraction=lake_fraction,
                                          mask_oceans=True, lons=lons2d, lats=lats2d)

            basemap.pcolormesh(x, y, to_plot, cmap=field_cmap, norm=field_norm)
            ax.set_ylabel("Mean: $X_{0}$".format(column))
            cb = basemap.colorbar(format=sfmt)
            #cb.ax.set_ylabel(infovar.get_units(var_name))

            #plot the difference
            ax = fig.add_subplot(gs[1, column])
            sel_axes.append(ax)
            ax.set_ylabel("$X_{0} - X_0$".format(column))

            ##Mask only if the previous plot (means) is masked
            thediff = the_mean - control_mean

            if hasattr(to_plot, "mask"):
                to_plot = np.ma.masked_where(to_plot.mask, thediff)
            else:
                to_plot = thediff

            if var_name == "PR":   # convert to mm/day
                to_plot = infovar.get_to_plot(var_name, to_plot, mask_oceans=False)

            vmin = np.ma.min(to_plot)
            vmax = np.ma.max(to_plot)

            d = max(abs(vmin), abs(vmax))
            vmin = -d
            vmax = d

            field_norm, bounds, vmn_nice, vmx_nice = infovar.get_boundary_norm(vmin, vmax, diff_cmap.N,
                                                                               exclude_zero=True)
            basemap.pcolormesh(x, y, to_plot, cmap=diff_cmap, norm=field_norm, vmin=vmn_nice, vmax=vmx_nice)

            cb = basemap.colorbar(format=sfmt)
            #cb.ax.set_ylabel(infovar.get_units(var_name))


        #plot coastlines
        for the_ax in sel_axes:
            basemap.drawcoastlines(ax=the_ax, linewidth=common_plot_params.COASTLINE_WIDTH)



        #depends on the compared simulations and the months of interest
        figFileName = "compare_{0}_{1}_{2}_months-{3}.jpeg".format(var_name, control_label,
                                                                   "_".join(labels),
                                                                   "-".join([str(m) for m in months_of_interest]))
        figPath = os.path.join(images_folder, figFileName)
        fig.savefig(figPath, dpi=cpp.FIG_SAVE_DPI, bbox_inches="tight")
        plt.close(fig)


def plot_all_seasons_for_a_var_as_panel():
    #TODO:
    pass


def plot_differences_for_a_var_as_panel():
    #TODO:
    pass


class DomainProperties(object):
    #The class for holding properties of the simulation domain
    def __init__(self):
        self.lons2d = None
        self.lats2d = None
        self.basemap = None
        self.lake_fraction = None
        self.x = None
        self.y = None


    def get_lon_lat_and_basemap(self):
        return self.lons2d, self.lats2d, self.basemap


def _plot_row(axes, data, sim_label, var_name, increments=False,
              domain_props=None, season_list=None):
    #data is a dict of season -> field
    #the field is a control mean in the case of the control mean
    #and the difference between the modified simulation and the control mean in the case of the modified simulation

    assert isinstance(domain_props, DomainProperties)

    lons2d, lats2d, basemap = domain_props.get_lon_lat_and_basemap()
    x, y = domain_props.x, domain_props.y

    vmin = None
    vmax = None
    #determine vmin and vmax for the row
    for season, field in data.iteritems():
        #field = _get_to_plot(var_name, field, lake_fraction=domain_props.lake_fraction, lons=lons2d, lats = lats2d)
        min_current, max_current = np.percentile(field[~field.mask], 1), np.percentile(field[~field.mask], 99)
        if vmin is None or min_current < vmin:
            vmin = min_current

        if vmax is None or max_current > vmax:
            vmax = max_current

    ncolors = 10

    if increments:
        # +1 to include white
        if vmin * vmax >= 0:
            if vmin >= 0:
                field_cmap = brewer2mpl.get_map("YlOrBr", "sequential", 9, reverse=False).get_mpl_colormap(N=ncolors)
            else:
                field_cmap = brewer2mpl.get_map("YlGnBu", "sequential", 9, reverse=True).get_mpl_colormap(N=ncolors)
            field_norm, bounds, bounds_min, bounds_max = infovar.get_boundary_norm(vmin, vmax, ncolors,
                                                                                   exclude_zero=False)
        else:
            field_cmap = brewer2mpl.get_map("RdBu", "diverging", 9, reverse=True).get_mpl_colormap(N=ncolors)
            d = np.ma.max(np.abs([vmin, vmax]))
            field_norm, bounds, bounds_min, bounds_max = infovar.get_boundary_norm(-d, d, ncolors, exclude_zero=False)
    else:
        #determine colorabr extent and spacing
        field_cmap, field_norm = infovar.get_colormap_and_norm_for(var_name, vmin=vmin, vmax=vmax, ncolors=ncolors)
    print "vmin = {0}; vmax = {1}".format(vmin, vmax)

    col = 0
    axes[0].set_ylabel(sim_label)
    im = None
    for season in season_list:
        field = data[season]
        ax = axes[col]
        if not increments:
            #since the increments go below
            ax.set_title(season)

        im = basemap.pcolormesh(x, y, field, norm=field_norm, cmap=field_cmap, ax=ax)
        basemap.drawcoastlines(ax=ax, linewidth=cpp.COASTLINE_WIDTH)
        col += 1

    #plot the common colorbar
    if isinstance(field_norm, LogNorm):
        plt.colorbar(im, cax=axes[-1])
    else:
        plt.colorbar(im, cax=axes[-1], extend="both")


def plot_control_and_differences_in_one_panel_for_all_seasons():
    #Used to plot control and differences
    season_to_months = OrderedDict([
        ("Winter", [12, 1, 2]),
        ("Spring", range(3, 6)),
        ("Summer", range(6, 9)),
        ("Fall", range(9, 12))
    ])

    #season_to_months = OrderedDict([
    #    #("March", [3, ]),
    #    ("April", [4, ]),
    #    ("May", [5, ]),
    #    ("June", [6, ]),
    #    ("July", [7, ])
    #])


    season_list = season_to_months.keys()

    # crcm5-r vs crcm5-hcd-r
    #control_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-r_spinup.hdf"
    #control_label = "CRCM5-R"
    #paths = ["/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r_spinup2.hdf", ]
    #labels = ["CRCM5-HCD-R"]

    # crcm5-hcd-rl vs crcm5-hcd-r
    control_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r_spinup2.hdf"
    control_label = "CRCM5-HCD-R"
    paths = ["/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl_spinup.hdf", ]
    labels = ["CRCM5-HCD-RL"]




    #compare simulations with and without interflow
    #control_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl_spinup.hdf"
    #control_label = "CRCM5-HCD-RL"
    #
    #paths = ["/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_do_not_discard_small.hdf", ]
    #labels = ["CRCM5-HCD-RL-INTFL"]


    row_labels = [
        r"$\Delta_{\rm " + s + "}$" for s in labels
    ]
    print labels


    varnames = ["STFL", "TT", "PR", "AV", "AH", "TRAF",
                "TDRA", "I5", "AS"]
    levels = [None, None, None, None, None, 1, 1, None, None]


    #varnames = ["TRAF", "TDRA", "I5"]
    #levels = [1, 1, None]


    #varnames = ["AS", ]
    #levels = [None, ]

    assert len(levels) == len(varnames)

    start_year = 1979
    end_year = 1985

    lons2d, lats2d, basemap = analysis.get_basemap_from_hdf(file_path=control_path)
    x, y = basemap(lons2d, lats2d)
    #save the domain properties for reuse
    domain_props = DomainProperties()
    domain_props.basemap = basemap
    domain_props.lons2d = lons2d
    domain_props.lats2d = lats2d
    domain_props.x = x
    domain_props.y = y

    lake_fraction = analysis.get_array_from_file(path=control_path, var_name="lake_fraction")

    if lake_fraction is None:
        lake_fraction = np.zeros(lons2d.shape)

    ncolors = 10
    # +1 to include white
    diff_cmap = brewer2mpl.get_map("RdBu", "diverging", ncolors + 1, reverse=True).get_mpl_colormap(N=ncolors + 1)

    #plot the plots one file per variable
    for var_name, level in zip(varnames, levels):
        sfmt = infovar.get_colorbar_formatter(var_name)
        season_to_control_mean = {}
        label_to_season_to_difference = {}



        #Calculate the difference for each season, and save the results to dictionaries
        #to access later when plotting
        for season, months_of_interest in season_to_months.iteritems():
            control_means = analysis.get_mean_2d_fields_for_months(path=control_path, var_name=var_name,
                                                                   months=months_of_interest,
                                                                   start_year=start_year, end_year=end_year,
                                                                   level=level)

            control_mean = np.mean(control_means, axis=0)

            control_mean = infovar.get_to_plot(var_name, control_mean,
                                               lake_fraction=domain_props.lake_fraction,
                                               lons=lons2d, lats=lats2d)
            season_to_control_mean[season] = control_mean

            #calculate the difference for each simulation
            for the_path, the_label in zip(paths, row_labels):
                modified_means = analysis.get_mean_2d_fields_for_months(path=the_path, var_name=var_name,
                                                                        months=months_of_interest,
                                                                        start_year=start_year, end_year=end_year,
                                                                        level=level)

                modified_mean = np.mean(modified_means, axis=0)
                if the_label not in label_to_season_to_difference:
                    label_to_season_to_difference[the_label] = OrderedDict()

                modified_mean = infovar.get_to_plot(var_name, modified_mean,
                                                    lake_fraction=domain_props.lake_fraction, lons=lons2d, lats=lats2d)

                label_to_season_to_difference[the_label][season] = modified_mean - control_mean

        #Do the plotting for each variable
        fig = plt.figure()
        assert isinstance(fig, Figure)

        #plot the control data
        ncols = len(season_to_control_mean) + 1  # +1 is for the colorbar
        gs = gridspec.GridSpec(len(paths) + 1, ncols, width_ratios=[1.0, ] * (ncols - 1) + [0.07])
        axes = []
        for col in range(ncols):
            axes.append(fig.add_subplot(gs[0, col]))
        _plot_row(axes, season_to_control_mean, control_label, var_name, domain_props=domain_props,
                  season_list=season_list)

        the_row = 1
        for the_label, data in label_to_season_to_difference.iteritems():
            axes = []
            for col in range(ncols):
                axes.append(fig.add_subplot(gs[the_row, col]))

            _plot_row(axes, data, the_label, var_name, increments=True, domain_props=domain_props,
                      season_list=season_list)
            the_row += 1

        folderPath = os.path.join(images_folder, "seasonal_mean_maps/HCD-R_vs_R")
        if not os.path.isdir(folderPath):
            os.mkdir(folderPath)

        imName = "{0}_{1}.jpeg".format(var_name, "_".join(labels + [control_label]))
        imPath = os.path.join(folderPath, imName)
        fig.savefig(imPath, bbox_inches="tight", dpi=cpp.FIG_SAVE_DPI)


def study_lake_effect_on_atmosphere():
    path_to_control_data = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-r_spinup.hdf"
    control_label = "CRCM5-R"

    paths = ["/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r_spinup2.hdf", ]
    labels = ["CRCM5-HCD-R", ]


    #get control means
    months = [1, ]
    varnames = ["STFL", "TT", "PR", "AV", "AH"]
    levels = [None, None, None, None, None]

    assert len(levels) == len(varnames)

    start_year = 1979
    end_year = 1988

    season_list = [
        [12, 1, 2], range(3, 6), range(6, 9), range(9, 12)
    ]

    for months in season_list:
        compare(paths, path_to_control_data=path_to_control_data,
                control_label=control_label, labels=labels,
                varnames=varnames, start_year=start_year, end_year=end_year,
                months_of_interest=months, levels=levels)


def study_interflow_effect():
    #Compare runs with and without interflow
    paths = ["/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup.hdf", ]
    labels = ["CRCM5-HCD-RL-INTFL", ]

    path_to_control_data = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl_spinup.hdf"
    control_label = "CRCM5-HCD-RL"

    paths = ["/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_do_not_discard_small.hdf", ]
    labels = ["CRCM5-HCD-RL-INTFL", ]


    #get control means
    months = [1, ]
    varnames = ["STFL", "TT", "PR", "AV", "AH", "I0", "I1", "I2"]
    levels = [None, None, None, None, None, 1, 1, 1]

    assert len(levels) == len(varnames)

    start_year = 1979
    end_year = 1985

    season_list = [
        [12, 1, 2], range(3, 6), range(6, 9), range(9, 12)
    ]

    for months in season_list:
        compare(paths, path_to_control_data=path_to_control_data,
                control_label=control_label, labels=labels,
                varnames=varnames, start_year=start_year, end_year=end_year,
                months_of_interest=months, levels=levels)


def main():
    import application_properties

    application_properties.set_current_directory()
    import time

    t0 = time.clock()
    study_interflow_effect()
    #study_lake_effect_on_atmosphere()
    print "Execution time: {0} seconds".format(time.clock() - t0)


if __name__ == "__main__":
    main()
