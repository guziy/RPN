import brewer2mpl
from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import os
from mpl_toolkits.basemap import maskoceans
from crcm5 import infovar
from data.anusplin import AnuSplinManager
from swe import SweDataManager
from matplotlib import cm

__author__ = 'huziy'


# Validate modelled precipitation data with Anusplin as well as daily min and max temperature

import matplotlib.pyplot as plt
import numpy as np

from . import do_analysis_using_pytables as analysis
from . import common_plot_params as cpp


images_folder = "/home/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/images_for_lake-river_paper"


def validate_precip(model_file="", simlabel="", obs_manager=None, season_to_months=None,
                    start_year=None, end_year=None, season_to_plot_indices=None):
    """
    :param model_file:
    :param obs_manager: should implement the method
        getMeanFieldForMonthsInterpolatedTo(self, months = None, lonsTarget = None, latsTarget = None)
        anusplin data is in mm/day
        model data is in m/s
    """

    model_var_name = "PR"
    model_level = 0
    reasonable_error_mm_per_day = 1

    assert isinstance(obs_manager, AnuSplinManager)
    fig = plt.figure()
    assert isinstance(fig, Figure)

    fig.suptitle("({0}) - ({1})".format(simlabel, "Obs."))

    lon, lat, basemap = analysis.get_basemap_from_hdf(file_path=model_file)

    # do calculations and only after that do the plotting
    season_to_field = {}

    # calculate global min and max for plotting
    vmin = None
    vmax = None

    for season, months in season_to_months.items():
        model_field = analysis.get_seasonal_climatology(start_year=start_year, end_year=end_year,
                                                        months=months,
                                                        level=model_level,
                                                        var_name=model_var_name, hdf_path=model_file)

        # convert m/s to mm/day for comparison with anusplin data
        model_field *= 1000.0 * 60 * 60 * 24

        obs_field = obs_manager.getMeanFieldForMonthsInterpolatedTo(months=months, lonstarget=lon, latstarget=lat,
                                                                    start_year=start_year, end_year=end_year)

        # calculate the difference between the modelled and observed fields
        the_diff = model_field - obs_field
        current_min = np.min(the_diff)
        current_max = np.max(the_diff)

        if vmin is not None:
            vmin = current_min if current_min < vmin else vmin
            vmax = current_max if current_max > vmax else vmax
        else:
            vmin = current_min
            vmax = current_max

        season_to_field[season] = the_diff

    ncolors = 12
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])

    cmap = cm.get_cmap("RdBu_r", ncolors)
    x, y = basemap(lon, lat)
    im = None

    d = min(abs(vmin), abs(vmax))
    vmin = -d
    vmax = d
    bn, bounds, _, _ = infovar.get_boundary_norm(vmin, vmax, ncolors, exclude_zero=False)

    print("bounds: ", bounds)

    cs = None
    for season, field in season_to_field.items():
        row, col = season_to_plot_indices[season]
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(season)
        basemap.drawmapboundary(fill_color="gray", ax=ax)
        im = basemap.pcolormesh(x, y, season_to_field[season], vmin=vmin, vmax=vmax, cmap=cmap, norm=bn)
        basemap.drawcoastlines(ax=ax, linewidth=cpp.COASTLINE_WIDTH)

        # small_error = (np.abs(season_to_field[season]) < reasonable_error_mm_per_day).astype(int)
        # nlevs = 1
        # ax.contour(x, y, small_error, nlevs, colors = "black", linestyle = "-")
        # cs = ax.contourf(x, y, small_error, nlevs, colors="none", hatches=["/", None], extend="lower", linewidth=2)


    # artists, labels = cs.legend_elements()
    # plt.legend(artists, labels, handleheight=2)

    cax = fig.add_subplot(gs[:, 2])
    cax.set_title("mm/day\n")
    plt.colorbar(im, cax=cax, extend="both")
    seasons_str = "_".join(sorted([str(s) for s in list(season_to_field.keys())]))
    atm_val_folder = os.path.join(images_folder, "validate_atm")
    if not os.path.isdir(atm_val_folder):
        os.mkdir(atm_val_folder)

    out_filename = "{3}/validate_2d_{0}_{1}_{2}.jpeg".format(model_var_name, simlabel, seasons_str, atm_val_folder)
    fig.savefig(os.path.join(images_folder, out_filename), dpi=cpp.FIG_SAVE_DPI, bbox_inches="tight")


def validate_temperature(
        model_file="", simlabel="", obs_manager=None, season_to_months=None,
        start_year=None, end_year=None, season_to_plot_indices=None, model_var_name="TT_max"):
    """
    :param model_file:
    :param obs_manager: should implement the method
        getMeanFieldForMonthsInterpolatedTo(self, months = None, lonsTarget = None, latsTarget = None)
        anusplin data is in degrees Celsium
        model data is in deg C
    """

    model_level = 0
    reasonable_error_deg = 2

    assert isinstance(obs_manager, AnuSplinManager)
    fig = plt.figure()
    assert isinstance(fig, Figure)

    fig.suptitle("({0}) - ({1})".format(simlabel, obs_manager.name))

    lon, lat, basemap = analysis.get_basemap_from_hdf(file_path=model_file)

    # do calculations and only after that do the plotting
    season_to_field = {}

    # calculate global min and max for plotting
    vmin = None
    vmax = None

    for season, months in season_to_months.items():
        model_field = analysis.get_seasonal_climatology(start_year=start_year, end_year=end_year,
                                                        months=months,
                                                        level=model_level,
                                                        var_name=model_var_name, hdf_path=model_file)

        obs_field = obs_manager.getMeanFieldForMonthsInterpolatedTo(months=months, lonstarget=lon, latstarget=lat,
                                                                    start_year=start_year, end_year=end_year)

        # calculate the difference between the modelled and observed fields
        the_diff = model_field - obs_field
        current_min = np.min(the_diff)
        current_max = np.max(the_diff)

        if vmin is not None:
            vmin = current_min if current_min < vmin else vmin
            vmax = current_max if current_max > vmax else vmax
        else:
            vmin = current_min
            vmax = current_max

        season_to_field[season] = the_diff

    ncolors = 10
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])

    cmap = cm.get_cmap("RdBu_r", ncolors)
    x, y = basemap(lon, lat)
    im = None

    d = min(abs(vmin), abs(vmax))
    vmin = -d
    vmax = d
    bn, bounds, _, _ = infovar.get_boundary_norm(vmin, vmax, ncolors)

    print("bounds: ", bounds)

    cs = None
    for season, field in season_to_field.items():
        row, col = season_to_plot_indices[season]
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(season)
        im = basemap.pcolormesh(x, y, season_to_field[season], vmin=vmin, vmax=vmax, cmap=cmap, norm=bn)
        basemap.drawcoastlines(ax=ax, linewidth=cpp.COASTLINE_WIDTH)

        small_error = (np.abs(season_to_field[season]) < reasonable_error_deg).astype(int)
        nlevs = 1
        # ax.contour(x, y, small_error, nlevs, colors = "black", linestyle = "-")
        cs = ax.contourf(x, y, small_error, nlevs, colors="none", hatches=["/", None], extend="lower", linewidth=2)


    # artists, labels = cs.legend_elements()
    # plt.legend(artists, labels, handleheight=2)

    cax = fig.add_subplot(gs[:, 2])

    units_str = r"${\rm ^\circ}$"
    var_str = r"$T_{\max}$" if model_var_name.endswith("_max") else r"$T_{\min}$"
    cax.set_title("{0}, {1}".format(var_str, units_str))
    plt.colorbar(im, cax=cax, extend = "both")
    seasons_str = "_".join(sorted([str(s) for s in list(season_to_field.keys())]))
    atm_val_folder = os.path.join(images_folder, "validate_atm")
    if not os.path.isdir(atm_val_folder):
        os.mkdir(atm_val_folder)

    out_filename = "{3}/validate_2d_{0}_{1}_{2}.jpeg".format(model_var_name, simlabel, seasons_str, atm_val_folder)
    fig.savefig(os.path.join(images_folder, out_filename), dpi=cpp.FIG_SAVE_DPI, bbox_inches="tight")


def validate_swe(model_file, obs_manager, season_to_months, simlabel, season_to_plot_indices, start_year, end_year,
                 lake_fraction = None):
    model_var_name = "I5"
    model_level = None
    reasonable_error_mm = 100.0
    assert isinstance(obs_manager, SweDataManager)

    print("lake fraction ranges: {0}, {1}".format(lake_fraction.min(), lake_fraction.max()))

    fig = plt.figure()
    obs_manager.name = "Obs."
    fig.suptitle("({0}) - ({1})".format(simlabel, obs_manager.name))


    # 1. read model results
    # 2. plot the differences (model - obs)

    lon, lat, basemap = analysis.get_basemap_from_hdf(file_path=model_file)

    # do calculations and only after that do the plotting
    season_to_field = {}

    # calculate global min and max for plotting
    vmin = None
    vmax = None

    season_to_obs_field = {}
    for season, months in season_to_months.items():
        model_field = analysis.get_seasonal_climatology(start_year=start_year, end_year=end_year,
                                                        months=months,
                                                        level=model_level,
                                                        var_name=model_var_name, hdf_path=model_file)

        obs_field = obs_manager.getMeanFieldForMonthsInterpolatedTo(months=months, lons_target=lon, lats_target=lat,
                                                                    start_year=start_year, end_year=end_year)

        season_to_obs_field[season] = obs_field
        # calculate the difference between the modelled and observed fields
        the_diff = model_field - obs_field
        current_min = np.min(the_diff)
        current_max = np.max(the_diff)

        if vmin is not None:
            vmin = current_min if current_min < vmin else vmin
            vmax = current_max if current_max > vmax else vmax
        else:
            vmin = current_min
            vmax = current_max

        season_to_field[season] = the_diff


    ncolors = 11
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])

    x, y = basemap(lon, lat)
    im = None

    d = min(abs(vmin), abs(vmax))

    d = 100  # limit module of the difference to 200 mm

    vmin = -d
    vmax = d
    # bn, bounds, _, _ = infovar.get_boundary_norm(vmin, vmax, ncolors, exclude_zero=True)

    bounds = [-100, -80, -50, -20, -10, -5]
    bounds += [-b for b in reversed(bounds)]
    bn = BoundaryNorm(bounds, ncolors=len(bounds) - 1)
    cmap = cm.get_cmap("RdBu_r", len(bounds) - 1)

    print("bounds: ", bounds)

    cs = None
    for season, field in season_to_field.items():

        if season.lower() == "summer":
            print("Warning: skipping summer season for SWE")
            continue

        row, col = season_to_plot_indices[season]
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(season)

        negligible_snow = season_to_obs_field[season] < 1

        basemap.drawmapboundary(fill_color="gray")
        to_plot = np.ma.masked_where((lake_fraction > 0.9), season_to_field[season])
        to_plot = maskoceans(lon, lat, to_plot)
        im = basemap.pcolormesh(x, y, to_plot, vmin=vmin, vmax=vmax, cmap=cmap, norm=bn)
        basemap.drawcoastlines(ax=ax, linewidth=cpp.COASTLINE_WIDTH)

        small_error = ((np.abs(season_to_field[season]) < reasonable_error_mm) | to_plot.mask).astype(int)
        nlevs = 1
        # ax.contour(x, y, small_error, nlevs, colors = "black", linestyle = "-")
        cs = ax.contourf(x, y, small_error, nlevs, colors="none", hatches=["/", None], extend="lower", linewidth=2)


    # artists, labels = cs.legend_elements()
    # plt.legend(artists, labels, handleheight=2)

    cax = fig.add_subplot(gs[:, 2])

    units_str = r"${\rm mm}$"
    var_str = r"SWE"
    cax.set_title("{0}, {1}\n".format(var_str, units_str))
    plt.colorbar(im, cax=cax, ticks = bounds, extend = "both")


    seasons_str = "_".join(sorted([str(s) for s in list(season_to_months.keys())]))
    atm_val_folder = os.path.join(images_folder, "validate_atm")
    if not os.path.isdir(atm_val_folder):
        os.mkdir(atm_val_folder)

    out_filename = "{3}/validate_2d_{0}_{1}_{2}.jpeg".format(model_var_name, simlabel, seasons_str, atm_val_folder)
    fig.savefig(os.path.join(images_folder, out_filename), dpi=cpp.FIG_SAVE_DPI, bbox_inches="tight")


def do_4_seasons(start_year=1980, end_year=2010):
    # Creates one file per simulation containing biases for 4 seasons
    season_to_months = {
        "Winter": [12, 1, 2],
        "Spring": list(range(3, 6)),
        "Summer": list(range(6, 9)),
        "Fall": list(range(9, 11))
    }

    season_to_plot_indices = {
        "Winter": (0, 0),
        "Spring": (0, 1),
        "Summer": (1, 0),
        "Fall": (1, 1)
    }

    simlabel_to_path = {
#        "CRCM5-R-CanESM2-current": "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-r-cc-canesm2-1980-2010.hdf5",
#        "CRCM5-R": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-r.hdf5",
        "CRCM5-HCD-R": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r.hdf5",
#        "CRCM5-HCD-RL-INTFL-ECOCLIMAP": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup_ecoclimap.hdf",
#        "CRCM5-HCD-RL-INTFL-ECOCLIMAP-ERA075": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup_ecoclimap_era075.hdf"
    }

    print("Period of interest: {0}-{1}".format(start_year, end_year))

    lake_fraction = analysis.get_array_from_file(list(simlabel_to_path.items())[-1][1],
                                                 var_name=infovar.HDF_LAKE_FRACTION_NAME)

    pcp_obs_manager = AnuSplinManager(variable="pcp")
    tmax_obs_manager = AnuSplinManager(variable="stmx")
    tmin_obs_manager = AnuSplinManager(variable="stmn")

    swe_obs_manager = SweDataManager(var_name="SWE")

    for simlabel, path in simlabel_to_path.items():
        # Validate precipitations
        validate_precip(model_file=path, obs_manager=pcp_obs_manager,
                        season_to_months=season_to_months, simlabel=simlabel,
                        season_to_plot_indices=season_to_plot_indices,
                        start_year=start_year, end_year=end_year)
        #
        # # Validate daily maximum temperature
        validate_temperature(model_file=path, obs_manager=tmax_obs_manager,
                             season_to_months=season_to_months, simlabel=simlabel,
                             season_to_plot_indices=season_to_plot_indices,
                             start_year=start_year, end_year=end_year, model_var_name="TT_max")

        validate_temperature(model_file=path, obs_manager=tmin_obs_manager,
                             season_to_months=season_to_months, simlabel=simlabel,
                             season_to_plot_indices=season_to_plot_indices,
                             start_year=start_year, end_year=end_year, model_var_name="TT_min")


        # validate swe
        validate_swe(model_file=path, obs_manager=swe_obs_manager,
                     season_to_months=season_to_months, simlabel=simlabel,
                     season_to_plot_indices=season_to_plot_indices,
                     start_year=start_year, end_year=end_year, lake_fraction = lake_fraction)


def main():
    obs_varname = "pcp"
    #anusplin = AnuSplinManager(variable=obs_varname)
    #validate_precip(obs_manager=anusplin)


if __name__ == "__main__":
    do_4_seasons(start_year=1980, end_year=1985)

