from collections import OrderedDict

from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import maskoceans

from application_properties import main_decorator

# Validate simulated seasonal mean SWE with EASE dataset
from crcm5.analyse_hdf.run_config import RunConfig



from crcm5.analyse_hdf import do_analysis_using_pytables as analysis
from data.ease_swe_manager import EaseSweManager
from util import plot_utils
from util.array_utils import aggregate_array
import matplotlib.pyplot as plt
import numpy as np
import os
from crcm5.analyse_hdf import common_plot_params

img_folder = "cc-paper-comments"


@main_decorator
def main():


    vname_model = "I5"
    nx_agg = 2
    ny_agg = 2



    start_year = 1980
    end_year = 2006

    r_config = RunConfig(
        data_path="/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5",
        start_year=start_year, end_year=end_year, label="ERAI-CRCM5-L"
    )


    bmp_info = analysis.get_basemap_info(r_config=r_config)
    bmp_info_agg = bmp_info.get_aggregated(nagg_x=nx_agg, nagg_y=ny_agg)

    season_to_months = OrderedDict([
        ("Winter", [12, 1, 2]),
         ("Spring", [3, 4, 5])
    ])



    # Get the model data
    seasonal_clim_fields_model = analysis.get_seasonal_climatology_for_runconfig(run_config=r_config,
                                                                                 varname=vname_model, level=0,
                                                                                 season_to_months=season_to_months)

    season_to_clim_fields_model_agg = OrderedDict()
    for season, field in seasonal_clim_fields_model.items():
        season_to_clim_fields_model_agg[season] = aggregate_array(field, nagg_x=nx_agg, nagg_y=ny_agg)



    # Get the EASE data
    obs_manager = EaseSweManager()
    season_to_clim_fields_obs = obs_manager.get_seasonal_clim_interpolated_to(target_lon2d=bmp_info_agg.lons, target_lat2d=bmp_info_agg.lats,
                                                                              season_to_months=season_to_months, start_year=start_year, end_year=end_year)


    # Do the plotting
    plot_utils.apply_plot_params(font_size=10, width_cm=16, height_cm=24)
    fig = plt.figure()
    xx, yy = bmp_info_agg.get_proj_xy()

    gs = GridSpec(3, len(season_to_clim_fields_model_agg) + 1, width_ratios=[1.0, ] * len(season_to_clim_fields_model_agg) + [0.05, ])
    clevs = [0, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 500]
    norm = BoundaryNorm(clevs, 256)

    clevs_diff = np.arange(-100, 110, 10)

    cs_val = None
    cs_diff = None

    col = 0

    lons_agg_copy = bmp_info_agg.lons.copy()
    lons_agg_copy[lons_agg_copy > 180] -= 360

    lons_copy = bmp_info.lons.copy()
    lons_copy[lons_copy > 180] -= 360

    xx1, yy1 = bmp_info.get_proj_xy()

    for season, mod_field in seasonal_clim_fields_model.items():


        obs_field = season_to_clim_fields_obs[season]

        row = 0
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(season)

        obs_field = maskoceans(lons_agg_copy, bmp_info_agg.lats, obs_field)
        cs_val = bmp_info_agg.basemap.contourf(xx, yy, obs_field, levels=clevs, norm=norm, ax=ax, extend="max")
        bmp_info_agg.basemap.drawcoastlines(linewidth=0.3, ax=ax)
        if col == 0:
            ax.set_ylabel("NSIDC")

        row += 1
        ax = fig.add_subplot(gs[row, col])
        mod_field = maskoceans(lons_copy, bmp_info.lats, mod_field)
        bmp_info.basemap.contourf(xx1, yy1, mod_field, levels=cs_val.levels, norm=cs_val.norm, ax=ax, extend="max")
        bmp_info.basemap.drawcoastlines(linewidth=0.3, ax=ax)
        if col == 0:
            ax.set_ylabel(r_config.label)

        row += 1
        ax = fig.add_subplot(gs[row, col])
        cs_diff = bmp_info_agg.basemap.contourf(xx, yy, season_to_clim_fields_model_agg[season] - obs_field, levels=clevs_diff, ax=ax, extend="both", cmap="seismic")
        bmp_info_agg.basemap.drawcoastlines(linewidth=0.3, ax=ax)

        if col == 0:
            ax.set_ylabel("{} minus {}".format(r_config.label, "NSIDC"))

        col += 1



    # Add values colorbar
    ax = fig.add_subplot(gs[0, -1])
    plt.colorbar(cs_val, cax=ax)
    ax.set_title("mm")


    # Add differences colorbaar
    ax = fig.add_subplot(gs[-1, -1])
    plt.colorbar(cs_diff, cax=ax)
    ax.set_title("mm")

    fig.tight_layout()
    fig.savefig(os.path.join(img_folder, "NSIDC_vs_CRCM_swe.png"), dpi=common_plot_params.FIG_SAVE_DPI, bbox_inches="tight")



if __name__ == '__main__':
    main()