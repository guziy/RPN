from collections import OrderedDict

from matplotlib.gridspec import GridSpec

from application_properties import main_decorator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os



# Compare precipitation biases from CRU and anusplin
from crcm5.analyse_hdf.climate_change import plot_performance_err_with_anusplin
from crcm5.analyse_hdf.climate_change import plot_performance_err_with_cru
from crcm5.analyse_hdf.run_config import RunConfig
from util import plot_utils
from util.array_utils import aggregate_array
from util.geo import quebec_info
from util.seasons_info import DEFAULT_SEASON_TO_MONTHS
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis


from crcm5.analyse_hdf import common_plot_params

BASIN_BOUNDARIES_SHP = quebec_info.BASIN_BOUNDARIES_DERIVED_10km

GL_SHP_FOLDER = "data/shp/Great_lakes_coast_shape"


img_folder = "cc-paper-comments"



@main_decorator
def main():
    """
    Everything is aggregated to the CRU resolution before calculating biases
    """
    season_to_months = DEFAULT_SEASON_TO_MONTHS
    varnames = ["PR", "TT"]

    plot_utils.apply_plot_params(font_size=5, width_pt=None, width_cm=15, height_cm=4)

    # reanalysis_driven_config = RunConfig(data_path="/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5",
    #                                      start_year=1980, end_year=2010, label="ERAI-CRCM5-L")

    reanalysis_driven_config = RunConfig(data_path="/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.4_crcm5-hcd-rl.hdf5",
                                         start_year=1980, end_year=2010, label="ERAI-CRCM5-RL_0.4deg")




    bmp_info = analysis.get_basemap_info(r_config=reanalysis_driven_config)

    field_cmap = cm.get_cmap("jet", 10)

    vname_to_clevels = {
        "TT": np.arange(-30, 32, 2), "PR": np.arange(0, 6.5, 0.5)
    }

    vname_to_anusplin_path = {
        "TT": "/home/huziy/skynet3_rech1/anusplin_links",
        "PR": "/home/huziy/skynet3_rech1/anusplin_links"
    }

    vname_to_cru_path = {
        "TT": "/HOME/data/Validation/CRU_TS_3.1/Original_files_gzipped/cru_ts_3_10.1901.2009.tmp.dat.nc",
        "PR": "/HOME/data/Validation/CRU_TS_3.1/Original_files_gzipped/cru_ts_3_10.1901.2009.pre.dat.nc"
    }

    xx_agg = None
    yy_agg = None


    for vname in varnames:

        # get anusplin obs climatology
        season_to_obs_anusplin = plot_performance_err_with_anusplin.get_seasonal_clim_obs_data(
            rconfig=reanalysis_driven_config,
            vname=vname, season_to_months=season_to_months, bmp_info=bmp_info)


        # get CRU obs values-------------------------
        bmp_info_agg, season_to_obs_cru = plot_performance_err_with_cru.get_seasonal_clim_obs_data(
            rconfig=reanalysis_driven_config, bmp_info=bmp_info, season_to_months=season_to_months,
            obs_path=vname_to_cru_path[vname], vname=vname
        )


        if xx_agg is None:
            xx_agg, yy_agg = bmp_info_agg.get_proj_xy()



        # get model data
        seasonal_clim_fields_model = analysis.get_seasonal_climatology_for_runconfig(run_config=reanalysis_driven_config,
                                                                                     varname=vname,
                                                                                     level=0,
                                                                                     season_to_months=season_to_months)


        ###
        biases_with_anusplin = OrderedDict()
        biases_with_cru = OrderedDict()


        nx_agg_anusplin = 4
        ny_agg_anusplin = 4

        nx_agg_model = 1
        ny_agg_model = 1

        season_to_clim_fields_model_agg = OrderedDict()
        for season, field in seasonal_clim_fields_model.items():
            print(field.shape)
            season_to_clim_fields_model_agg[season] = aggregate_array(field, nagg_x=nx_agg_model, nagg_y=ny_agg_model)

            if vname == "PR":
                season_to_clim_fields_model_agg[season] *= 1.0e3 * 24 * 3600


            biases_with_cru[season] = season_to_clim_fields_model_agg[season] - season_to_obs_cru[season]

            biases_with_anusplin[season] = season_to_clim_fields_model_agg[season] - aggregate_array(season_to_obs_anusplin[season],
                                                                                                     nagg_x=nx_agg_anusplin,
                                                                                                     nagg_y=ny_agg_anusplin)


        # Do the plotting
        fig = plt.figure()
        clevs = [c for c in np.arange(-0.5, 0.55, 0.05)] if vname == "PR" else np.arange(-2, 2.2, 0.2)

        gs = GridSpec(1, len(biases_with_cru) + 1, width_ratios=len(biases_with_cru) * [1., ] + [0.05, ])

        col = 0
        cs = None
        cmap = "seismic"

        fig.suptitle(r"$\left| \delta_{\rm Hopkinson} \right| - \left| \delta_{\rm CRU} \right|$")

        for season, cru_err in biases_with_cru.items():
            anu_err = biases_with_anusplin[season]

            ax = fig.add_subplot(gs[0, col])

            diff = np.abs(anu_err) - np.abs(cru_err)
            cs = bmp_info_agg.basemap.contourf(xx_agg, yy_agg, diff, levels=clevs, ax=ax, extend="both", cmap=cmap)
            bmp_info_agg.basemap.drawcoastlines(ax=ax, linewidth=0.3)


            good = diff[~diff.mask & ~np.isnan(diff)]
            n_neg = sum(good < 0) / sum(good > 0)

            print("season: {}, n-/n+ = {}".format(season, n_neg))

            ax.set_title(season)
            ax.set_xlabel(r"$n_{-}/n_{+} = $" + "{:.1f}".format(n_neg) + "\n" + r"$\overline{\varepsilon} = $" + "{:.2f}".format(good.mean()))

            col += 1


        ax = fig.add_subplot(gs[0, -1])
        plt.colorbar(cs, cax=ax)
        ax.set_title("mm/day" if vname == "PR" else r"${\rm ^\circ C}$")


        fig.savefig(os.path.join(img_folder, "comp_anu_and_cru_biases_for_{}_{}.png".format(vname, reanalysis_driven_config.label)),
                    bbox_inches="tight",
                    dpi=common_plot_params.FIG_SAVE_DPI)




if __name__ == '__main__':
    main()