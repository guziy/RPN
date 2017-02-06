import os
from collections import OrderedDict

from matplotlib import cm
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from crcm5 import infovar
from crcm5.analyse_hdf.run_config import RunConfig
from util import plot_utils
from util.geo import quebec_info
from util.geo.mask_from_shp import get_mask
from util.seasons_info import DEFAULT_SEASON_TO_MONTHS
import crcm5.analyse_hdf.do_analysis_using_pytables as analysis

__author__ = 'huziy'

# Plot on the same panel performance errors and obs data for anusplin and CRU
# for temperature and precipitation separately

from crcm5.analyse_hdf.climate_change import plot_performance_err_with_cru
from crcm5.analyse_hdf.climate_change import plot_performance_err_with_anusplin
import numpy as np

from application_properties import main_decorator

img_folder = Path("cc_paper/perf_err_with_anusplin_and_cru_merged")

BASIN_BOUNDARIES_SHP = quebec_info.BASIN_BOUNDARIES_DERIVED_10km

GL_SHP_FOLDER = "data/shp/Great_lakes_coast_shape"


def _format_axes(ax_list, vname="TT"):
    # erase titles
    for the_ax in ax_list:
        the_ax.set_title("")


@main_decorator
def main():
    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)

    season_to_months = OrderedDict([
        ("Winter (DJF)", (1, 2, 12)),
        ("Spring (MAM)", range(3, 6)),
        ("Summer (JJA)", range(6, 9)),
        ("Fall (SON)", range(9, 12)),
    ])

    varnames = ["TT", "PR"]

    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=20, height_cm=17)

    # reanalysis_driven_config = RunConfig(data_path="/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5",
    #                                      start_year=1980, end_year=2010, label="ERAI-CRCM5-L")
    #

    reanalysis_driven_config = RunConfig(data_path="/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.4_crcm5-hcd-rl.hdf5",
                                         start_year=1980, end_year=2010, label="ERAI-CRCM5-L(0.4)")

    nx_agg_model = 1
    ny_agg_model = 1

    nx_agg_anusplin = 4
    ny_agg_anusplin = 4





    gcm_driven_config = RunConfig(
        data_path="/RESCUE/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5",
        start_year=1980, end_year=2010, label="CanESM2-CRCM5-L")

    bmp_info = analysis.get_basemap_info(r_config=reanalysis_driven_config)
    xx, yy = bmp_info.get_proj_xy()

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

    for vname in varnames:
        fig = plt.figure()
        ncols = len(season_to_months)
        gs = GridSpec(4, ncols + 1, width_ratios=ncols * [1., ] + [0.09, ])

        clevels = vname_to_clevels[vname]

        # get anusplin obs climatology
        season_to_obs_anusplin = plot_performance_err_with_anusplin.get_seasonal_clim_obs_data(
            rconfig=reanalysis_driven_config,
            vname=vname, season_to_months=season_to_months, bmp_info=bmp_info,
            n_agg_x=nx_agg_anusplin, n_agg_y=ny_agg_anusplin)

        row = 0

        # Plot CRU values-------------------------
        bmp_info_agg, season_to_obs_cru = plot_performance_err_with_cru.get_seasonal_clim_obs_data(
            rconfig=reanalysis_driven_config, bmp_info=bmp_info, season_to_months=season_to_months,
            obs_path=vname_to_cru_path[vname], vname=vname
        )

        # Mask out the Great Lakes
        cru_mask = get_mask(bmp_info_agg.lons, bmp_info_agg.lats, shp_path=os.path.join(GL_SHP_FOLDER, "gl_cst.shp"))
        for season in season_to_obs_cru:
            season_to_obs_cru[season] = np.ma.masked_where(cru_mask > 0.5, season_to_obs_cru[season])

        ax_list = [fig.add_subplot(gs[row, j]) for j in range(ncols)]
        cs = None
        xx_agg, yy_agg = bmp_info_agg.get_proj_xy()
        for j, (season, obs_field) in enumerate(season_to_obs_cru.items()):
            ax = ax_list[j]
            cs = bmp_info_agg.basemap.contourf(xx_agg, yy_agg, obs_field.copy(), levels=clevels, ax=ax)
            bmp_info.basemap.drawcoastlines(ax=ax)
            bmp_info.basemap.readshapefile(BASIN_BOUNDARIES_SHP[:-4], "basin", ax=ax)
            ax.set_title(season)

        ax_list[0].set_ylabel("CRU")
        # plt.colorbar(cs, caax=ax_list[-1])
        row += 1

        # Plot ANUSPLIN values-------------------------
        ax_list = [fig.add_subplot(gs[row, j]) for j in range(ncols)]
        cs = None
        for j, (season, obs_field) in enumerate(season_to_obs_anusplin.items()):
            ax = ax_list[j]
            cs = bmp_info.basemap.contourf(xx, yy, obs_field, levels=clevels, ax=ax)
            bmp_info.basemap.drawcoastlines(ax=ax)
            bmp_info.basemap.readshapefile(BASIN_BOUNDARIES_SHP[:-4], "basin", ax=ax)
            ax.set_title(season)

        ax_list[0].set_ylabel("Hopkinson")
        cb = plt.colorbar(cs, cax=fig.add_subplot(gs[:2, -1]))
        cb.ax.set_xlabel(infovar.get_units(vname))
        _format_axes(ax_list, vname=vname)
        row += 1

        # Plot model (CRCM) values-------------------------
        # ax_list = [fig.add_subplot(gs[row, j]) for j in range(ncols)]
        # cs = None
        #
        # season_to_field_crcm = analysis.get_seasonal_climatology_for_runconfig(run_config=reanalysis_driven_config,
        #                                                                        varname=vname, level=0,
        #                                                                        season_to_months=season_to_months)
        #
        # for j, (season, crcm_field) in enumerate(season_to_field_crcm.items()):
        #     ax = ax_list[j]
        #     cs = bmp_info.basemap.contourf(xx, yy, crcm_field * 1000 * 24 * 3600, levels=clevels, ax=ax)
        #     bmp_info.basemap.drawcoastlines(ax=ax)
        #     bmp_info.basemap.readshapefile(BASIN_BOUNDARIES_SHP[:-4], "basin", ax=ax)
        #     ax.set_title(season)
        #
        # ax_list[0].set_ylabel(reanalysis_driven_config.label)
        # cb = plt.colorbar(cs, cax=fig.add_subplot(gs[:2, -1]))
        # cb.ax.set_xlabel(infovar.get_units(vname))
        # _format_axes(ax_list, vname=vname)
        # row += 1


        # Plot (Model - CRU) Performance biases-------------------------
        ax_list = [fig.add_subplot(gs[row, j]) for j in range(ncols)]
        cs = plot_performance_err_with_cru.compare_vars(vname_model=vname, vname_obs=None,
                                                        r_config=reanalysis_driven_config,
                                                        season_to_months=season_to_months,
                                                        obs_path=vname_to_cru_path[vname],
                                                        bmp_info_agg=bmp_info_agg, diff_axes_list=ax_list,
                                                        mask_shape_file=os.path.join(GL_SHP_FOLDER, "gl_cst.shp"),
                                                        nx_agg_model=nx_agg_model, ny_agg_model=ny_agg_model)

        ax_list[0].set_ylabel("{label}\n--\nCRU".format(label=reanalysis_driven_config.label))
        _format_axes(ax_list, vname=vname)
        row += 1

        # Plot performance+BFE errors with respect to CRU (Model - CRU)-------------------------
        # ax_list = [fig.add_subplot(gs[row, j]) for j in range(ncols)]
        # plot_performance_err_with_cru.compare_vars(vname, vname_obs=None, obs_path=vname_to_cru_path[vname],
        #                                            r_config=gcm_driven_config,
        #                                            bmp_info_agg=bmp_info_agg, season_to_months=season_to_months,
        #                                            axes_list=ax_list)
        # _format_axes(ax_list, vname=vname)
        # ax_list[0].set_ylabel("{label}\nvs\nCRU".format(label=gcm_driven_config.label))
        # row += 1


        # Plot performance errors with respect to ANUSPLIN (Model - ANUSPLIN)-------------------------
        ax_list = [fig.add_subplot(gs[row, j]) for j in range(ncols)]
        plot_performance_err_with_anusplin.compare_vars(vname, {vname: season_to_obs_anusplin},
                                                        r_config=reanalysis_driven_config,
                                                        bmp_info_agg=bmp_info, season_to_months=season_to_months,
                                                        axes_list=ax_list)
        _format_axes(ax_list, vname=vname)
        ax_list[0].set_ylabel("{label}\n--\nHopkinson".format(label=reanalysis_driven_config.label))
        row += 1

        # Plot performance+BFE errors with respect to ANUSPLIN (Model - ANUSPLIN)-------------------------
        # ax_list = [fig.add_subplot(gs[row, j]) for j in range(ncols)]
        # plot_performance_err_with_anusplin.compare_vars(vname, {vname: season_to_obs_anusplin},
        #                                                 r_config=gcm_driven_config,
        #                                                 bmp_info_agg=bmp_info, season_to_months=season_to_months,
        #                                                 axes_list=ax_list)
        # _format_axes(ax_list, vname=vname)
        # ax_list[0].set_ylabel("{label}\nvs\nHopkinson".format(label=gcm_driven_config.label))


        cb = plt.colorbar(cs, cax=fig.add_subplot(gs[-2:, -1]))
        cb.ax.set_xlabel(infovar.get_units(vname))

        # Save the plot
        img_file = "{vname}_{sy}-{ey}_{sim_label}.png".format(
            vname=vname, sy=reanalysis_driven_config.start_year, ey=reanalysis_driven_config.end_year,
            sim_label=reanalysis_driven_config.label)

        img_file = img_folder.joinpath(img_file)
        with img_file.open("wb") as f:
            fig.savefig(f, bbox_inches="tight")
        plt.close(fig)


if __name__ == '__main__':
    main()
