import brewer2mpl
from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import os
from crcm5 import infovar
from data.anusplin import AnuSplinManager

__author__ = 'huziy'


#Validate modelled precipitation data with Anusplin as well as daily min and max temperature

import matplotlib.pyplot as plt
import numpy as np

import do_analysis_using_pytables as analysis
import common_plot_params as cpp


images_folder = "/home/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/images_for_lake-river_paper"

def validate_precip(model_file = "", simlabel = "", obs_manager = None, season_to_months = None,
                    start_year = None, end_year = None, season_to_plot_indices = None
                    ):
    """
    :param model_file:
    :param obs_manager: should implement the method
        getMeanFieldForMonthsInterpolatedTo(self, months = None, lonsTarget = None, latsTarget = None)
        anusplin data is in mm/day
        model data is in m/s
    """

    model_var_name = "PR"
    model_level = None
    reasonable_error_mm_per_day = 1

    assert isinstance(obs_manager, AnuSplinManager)
    fig = plt.figure()
    assert isinstance(fig, Figure)



    fig.suptitle("({0}) - ({1})".format(simlabel, obs_manager.name))

    lon, lat, basemap = analysis.get_basemap_from_hdf(file_path=model_file)

    #do calculations and only after that do the plotting
    season_to_field = {}

    #calculate global min and max for plotting
    vmin = None
    vmax = None

    for season, months in season_to_months.iteritems():
        model_field = analysis.get_seasonal_climatology(start_year = start_year, end_year = end_year,
            months=months,
            level=model_level,
            var_name=model_var_name, hdf_path=model_file)

        #convert m/s to mm/day for comparison with anusplin data
        model_field *= 1000.0 * 60 * 60 * 24


        obs_field = obs_manager.getMeanFieldForMonthsInterpolatedTo(months=months, lonsTarget=lon, latsTarget=lat,
            start_year=start_year, end_year=end_year
        )
        #calculate the difference between the modelled and observed fields
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
    gs = gridspec.GridSpec(2,3, width_ratios=[1,1,0.05])

    cmap = brewer2mpl.get_map("RdBu", "diverging", 10, reverse=True).get_mpl_colormap(N = ncolors)
    x, y = basemap(lon, lat)
    im = None

    d = min(abs(vmin), abs(vmax))
    vmin = -d
    vmax = d
    bn, bounds, _, _ = infovar.get_boundary_norm(vmin, vmax, ncolors)

    print "bounds: ", bounds

    cs = None
    for season, field in season_to_field.iteritems():
        row, col = season_to_plot_indices[season]
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(season)
        im = basemap.pcolormesh(x, y, season_to_field[season], vmin = vmin, vmax = vmax, cmap = cmap, norm = bn)
        basemap.drawcoastlines(ax=ax, linewidth=cpp.COASTLINE_WIDTH)


        small_error = (np.abs(season_to_field[season]) < reasonable_error_mm_per_day).astype(int)
        nlevs = 1
        #ax.contour(x, y, small_error, nlevs, colors = "black", linestyle = "-")
        cs = ax.contourf(x, y, small_error, nlevs, colors = "none", hatches=["/", None], extend="lower", linewidth = 2)


    #artists, labels = cs.legend_elements()
    #plt.legend(artists, labels, handleheight=2)

    cax = fig.add_subplot(gs[:,2])
    cax.set_title("mm/day")
    plt.colorbar(im, cax=cax)
    seasons_str = "_".join(sorted( [str(s) for s in season_to_field.keys()]))
    out_filename = "validate_2d_{0}_{1}_{2}.jpeg".format(model_var_name, simlabel, seasons_str)
    fig.savefig(os.path.join(images_folder, out_filename), dpi = cpp.FIG_SAVE_DPI, bbox_inches = "tight")


def do_4_seasons(start_year = 1979, end_year = 1988):
    #Creates one file per simulation containing biases for 4 seasons
    season_to_months = {
        "Winter" : [12, 1, 2],
        "Spring" : range(3, 6),
        "Summer" : range(6, 9),
        "Fall"   : range(9, 11)
    }

    season_to_plot_indices = {
        "Winter" : (0, 0),
        "Spring" : (0, 1),
        "Summer" : (1, 0),
        "Fall"   : (1, 1)
    }

    simlabel_to_path = {
        "CRCM5-R": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-r_spinup.hdf",
        "CRCM5-HCD-R": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r_spinup2.hdf",
        "CRCM5-HCD-RL_K5": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl-kd5_spinup.hdf"
    }

    print "Period of interest: {0}-{1}".format(start_year, end_year)

    pcpObsManager = AnuSplinManager(variable="pcp")
    for simlabel, path in simlabel_to_path.iteritems():
        #validate precipitations
        validate_precip(model_file=path, obs_manager=pcpObsManager,
            season_to_months = season_to_months, simlabel = simlabel,
            season_to_plot_indices=season_to_plot_indices,
            start_year=start_year, end_year=end_year
        )




def main():
    obs_varname = "pcp"
    anusplin = AnuSplinManager(variable=obs_varname)
    validate_precip(obs_manager=anusplin)



