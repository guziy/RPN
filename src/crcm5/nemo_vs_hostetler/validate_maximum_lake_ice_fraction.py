




# compare temperature and total precip biases in the NEMO- and Hostetler-based simulations
from collections import OrderedDict
from collections import defaultdict
from pathlib import Path

import matplotlib
# matplotlib.use("agg")

from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import maskoceans
from rpn import level_kinds
from rpn.domains.rotated_lat_lon import RotatedLatLon
from rpn.rpn import RPN

from application_properties import main_decorator
from crcm5.analyse_hdf.run_config import RunConfig
from lake_effect_snow import data_source_types

from lake_effect_snow import default_varname_mappings
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.data_manager import DataManager
from lake_effect_snow.default_varname_mappings import T_AIR_2M, U_WE, V_SN, TOTAL_PREC, SWE, LAKE_ICE_FRACTION

import numpy as np

import matplotlib.pyplot as plt

# season name to month list mapping (the order of months is important)
from util import plot_utils
from util.geo import lat_lon

season_to_months = OrderedDict([
    ("DJF", [12, 1, 2]),
    ("MAM", [3, 4, 5]),
    # ("JJA", [6, 7, 8]),
    # ("SON", [9, 10, 11])
])

img_folder = Path("nemo_vs_hostetler")

internal_name_to_title = {
    T_AIR_2M: r"2-m air temperature ($^\circ$C)",
    TOTAL_PREC: "Total recipitation (mm/day)",
    SWE: "SWE (mm)",
    LAKE_ICE_FRACTION: "lake ice fraction"
}

internal_name_to_clevs = {
    T_AIR_2M: np.arange(-30, 30, 2),
    T_AIR_2M + "bias": np.arange(-4, 4.2, 0.2),
    T_AIR_2M + "biasdiff": np.arange(-1, 1.1, 0.1),
    TOTAL_PREC: np.arange(0, 8, 0.5),
    TOTAL_PREC + "bias": np.arange(-3, 3.2, 0.2),
    TOTAL_PREC + "biasdiff": np.arange(-1, 1.1, 0.1),
    SWE: np.arange(0, 610, 10),
    SWE + "bias": np.arange(-200, 210, 10),
    SWE + "biasdiff": np.arange(-15, 15, 1),
    LAKE_ICE_FRACTION: np.arange(0, 1.1, 0.1),
    LAKE_ICE_FRACTION + "bias": np.arange(-1, 1.1, 0.1),
    LAKE_ICE_FRACTION + "biasdiff": np.arange(-1, 1.1, 0.1),
}


def get_clevs(internal_name):
    if internal_name in internal_name_to_clevs:
        return internal_name_to_clevs[internal_name]
    return None


internal_name_to_multiplier = defaultdict(lambda : 1)
internal_name_to_multiplier[TOTAL_PREC] = 1000.0 * 24 * 3600 # Convert precip to mm/day



def get_target_lons_lats_basemap(run_config: RunConfig=None):

    base_dir = Path(run_config.data_path)

    for month_dir in base_dir.iterdir():
        if month_dir.is_dir():
            for f in month_dir.iterdir():

                if f.name.startswith("."):
                    continue

                with RPN(str(f)) as r:
                    assert isinstance(r, RPN)
                    vlist = r.get_list_of_varnames()

                    vname = [v for v in vlist if v not in [">>", "^^", "HY"]][0]

                    r.get_first_record_for_name(vname)

                    lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()


                    rll = RotatedLatLon(**r.get_proj_parameters_for_the_last_read_rec())
                    basemap = rll.get_basemap_object_for_lons_lats(lons2d=lons, lats2d=lats)
                    return lons, lats, basemap




@main_decorator
def main():

    obs_data_path = Path("/RESCUE/skynet3_rech1/huziy/obs_data_for_HLES/interploated_to_the_same_grid/GL_0.1_452x260/anusplin+_interpolated_tt_pr.nc")

    start_year = 1980
    end_year = 2010

    HL_LABEL = "CRCM5_HL"
    NEMO_LABEL = "CRCM5_NEMO"


    vars_of_interest = [
        LAKE_ICE_FRACTION,
    ]

    sim_configs = {

        HL_LABEL: RunConfig(data_path="/RECH2/huziy/coupling/GL_440x260_0.1deg_GL_with_Hostetler/Samples_selected",
                  start_year=start_year, end_year=end_year, label=HL_LABEL),

        NEMO_LABEL: RunConfig(data_path="/RECH2/huziy/coupling/coupled-GL-NEMO1h_30min/selected_fields",
                  start_year=start_year, end_year=end_year, label=NEMO_LABEL),
    }

    sim_labels = [HL_LABEL, NEMO_LABEL]

    vname_to_level = {
        T_AIR_2M: VerticalLevel(1, level_kinds.HYBRID),
        U_WE: VerticalLevel(1, level_kinds.HYBRID),
        V_SN: VerticalLevel(1, level_kinds.HYBRID),
    }


    # Calculations

    # prepare params for interpolation
    lons_t, lats_t, bsmap = get_target_lons_lats_basemap(sim_configs[HL_LABEL])
    xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons_t.flatten(), lats_t.flatten())


    vname_map = {}
    vname_map.update(default_varname_mappings.vname_map_CRCM5)



    # Read and calculate observed seasonal means
    store_config = {
            "base_folder": obs_data_path.parent,
            "data_source_type": data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES_OPEN_EACH_FILE_SEPARATELY,
            "varname_mapping": vname_map,
            "level_mapping": vname_to_level,
            "offset_mapping": default_varname_mappings.vname_to_offset_CRCM5,
            "multiplier_mapping": default_varname_mappings.vname_to_multiplier_CRCM5,
    }

    obs_dm = DataManager(store_config=store_config)
    obs_data = {}

    interp_indices = None
    for vname in vars_of_interest:
        # --
        end_year_for_current_var = end_year
        if vname == SWE:
            end_year_for_current_var = min(1996, end_year)

        # --
        seas_to_year_to_max = obs_dm.get_seasonal_maxima(varname_internal=vname,
                                                     start_year=start_year,
                                                     end_year=end_year_for_current_var,
                                                     season_to_months=season_to_months)

        seas_to_clim = {seas: np.array(list(y_to_means.values())).mean(axis=0) for seas, y_to_means in seas_to_year_to_max.items()}
        obs_data[vname] = seas_to_clim

        if interp_indices is None:
            _, interp_indices = obs_dm.get_kdtree().query(list(zip(xt, yt, zt)))

        for season in seas_to_clim:
            seas_to_clim[season] = seas_to_clim[season].flatten()[interp_indices].reshape(lons_t.shape)

    # Read and calculate simulated seasonal mean biases
    sim_data = defaultdict(dict)
    for label, r_config in sim_configs.items():

        store_config = {
                "base_folder": r_config.data_path,
                "data_source_type": data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT_VNAME_IN_FNAME,
                "varname_mapping": vname_map,
                "level_mapping": vname_to_level,
                "offset_mapping": default_varname_mappings.vname_to_offset_CRCM5,
                "multiplier_mapping": default_varname_mappings.vname_to_multiplier_CRCM5,
        }


        dm = DataManager(store_config=store_config)


        interp_indices = None
        for vname in vars_of_interest:

            # --
            end_year_for_current_var = end_year
            if vname == SWE:
                end_year_for_current_var = min(1996, end_year)

            # --
            seas_to_year_to_max = dm.get_seasonal_maxima(varname_internal=vname,
                                                           start_year=start_year,
                                                           end_year=end_year_for_current_var,
                                                           season_to_months=season_to_months)

            # get the climatology
            seas_to_clim = {seas: np.array(list(y_to_means.values())).mean(axis=0) for seas, y_to_means in seas_to_year_to_max.items()}

            sim_data[label][vname] = seas_to_clim

            if interp_indices is None:
                _, interp_indices = dm.get_kdtree().query(list(zip(xt, yt, zt)))

            for season in seas_to_clim:
                seas_to_clim[season] = seas_to_clim[season].flatten()[interp_indices].reshape(lons_t.shape) - obs_data[vname][season]







    # Plotting: interpolate to the same grid and plot obs and biases
    plot_utils.apply_plot_params(width_cm=32, height_cm=20, font_size=8)



    xx, yy = bsmap(lons_t, lats_t)
    lons_t[lons_t > 180] -= 360
    field_mask = ~maskoceans(lons_t, lats_t, np.zeros_like(lons_t)).mask

    for vname in vars_of_interest:

        fig = plt.figure()

        fig.suptitle(internal_name_to_title[vname] + "\n")

        nrows = len(sim_configs) + 2
        ncols = len(season_to_months)
        gs = GridSpec(nrows=nrows, ncols=ncols)



        # Plot the obs fields
        current_row = 0
        for col, season in enumerate(season_to_months):
            field = obs_data[vname][season]
            ax = fig.add_subplot(gs[current_row, col])
            ax.set_title(season)

            to_plot = np.ma.masked_where(field_mask, field) * internal_name_to_multiplier[vname]
            clevs = get_clevs(vname)

            if clevs is not None:
                bnorm = BoundaryNorm(clevs, len(clevs) - 1)
                cmap = cm.get_cmap("jet", len(clevs) - 1)
            else:
                cmap = "jet"
                bnorm = None

            cs = bsmap.contourf(xx, yy, to_plot, ax=ax, levels=get_clevs(vname), norm=bnorm, cmap=cmap)
            bsmap.drawcoastlines()
            bsmap.colorbar(cs, ax=ax)

            if col == 0:
                ax.set_ylabel("Obs")



        # plot the biases
        for sim_label in sim_labels:
            current_row += 1
            for col, season in enumerate(season_to_months):

                field = sim_data[sim_label][vname][season]

                ax = fig.add_subplot(gs[current_row, col])

                clevs = get_clevs(vname + "bias")
                if clevs is not None:
                    bnorm = BoundaryNorm(clevs, len(clevs) - 1)
                    cmap = cm.get_cmap("bwr", len(clevs) - 1)
                else:
                    cmap = "bwr"
                    bnorm = None

                to_plot = np.ma.masked_where(field_mask, field) * internal_name_to_multiplier[vname]
                cs = bsmap.contourf(xx, yy, to_plot, ax=ax, extend="both", levels=get_clevs(vname + "bias"), cmap=cmap, norm=bnorm)
                bsmap.drawcoastlines()
                bsmap.colorbar(cs, ax=ax)

                if col == 0:
                    ax.set_ylabel("{}\n-\nObs.".format(sim_label))


        # plot differences between the biases
        current_row += 1
        for col, season in enumerate(season_to_months):

            field = sim_data[NEMO_LABEL][vname][season] - sim_data[HL_LABEL][vname][season]

            ax = fig.add_subplot(gs[current_row, col])

            clevs = get_clevs(vname + "biasdiff")
            if clevs is not None:
                bnorm = BoundaryNorm(clevs, len(clevs) - 1)
                cmap = cm.get_cmap("bwr", len(clevs) - 1)
            else:
                cmap = "bwr"
                bnorm = None

            to_plot = np.ma.masked_where(field_mask, field) * internal_name_to_multiplier[vname]
            cs = bsmap.contourf(xx, yy, to_plot, ax=ax, extend="both", levels=get_clevs(vname + "biasdiff"), cmap=cmap, norm=bnorm)
            bsmap.drawcoastlines()
            bsmap.colorbar(cs, ax=ax)

            if col == 0:
                ax.set_ylabel("{}\n-\n{}".format(NEMO_LABEL, HL_LABEL))


        fig.tight_layout()

        # save a figure per variable
        img_file = "seasonal_biases_{}_{}_{}-{}.png".format(vname,
                                                            "-".join([s for s in season_to_months]),
                                                            start_year, end_year)
        img_file = img_folder.joinpath(img_file)

        fig.savefig(str(img_file))

        plt.close(fig)



if __name__ == '__main__':
    main()
