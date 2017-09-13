




# compare temperature and total precip biases in the NEMO- and Hostetler-based simulations
from collections import OrderedDict
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import maskoceans
from rpn import level_kinds
from rpn.domains.rotated_lat_lon import RotatedLatLon
from rpn.rpn import RPN
from scipy.stats import ttest_ind_from_stats

from application_properties import main_decorator
from crcm5.analyse_hdf.run_config import RunConfig
from crcm5.mh_domains import constants
from crcm5.mh_domains import default_domains
from data.robust import data_source_types
from data.robust.data_manager import DataManager
from lake_effect_snow import default_varname_mappings
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.default_varname_mappings import T_AIR_2M, U_WE, V_SN, TOTAL_PREC, SWE, LAKE_ICE_FRACTION
# season name to month list mapping (the order of months is important)
from util import plot_utils
from util.geo import lat_lon

# matplotlib.use("agg")

season_to_months = OrderedDict([
    ("DJF", [12, 1, 2]),
    ("MAM", [3, 4, 5]),
    ("JJA", [6, 7, 8]),
    ("SON", [9, 10, 11])
])

img_folder = Path("mh/engage_report/seasonal_biases")

internal_name_to_title = {
    T_AIR_2M: r"2-m air temperature ($^\circ$C)",
    TOTAL_PREC: "Total recipitation (mm/day)",
    SWE: "SWE (mm)",
    # LAKE_ICE_FRACTION: "lake ice fraction"
}

internal_name_to_clevs = {
    T_AIR_2M: np.arange(-30, 33, 3),
    T_AIR_2M + "bias": np.arange(-4, 4.2, 0.2),
    T_AIR_2M + "biasdiff": np.arange(-1, 1.1, 0.1),
    TOTAL_PREC: np.arange(0, 6, 0.5),
    TOTAL_PREC + "bias": np.arange(-1.2, 1.4, 0.2),
    TOTAL_PREC + "biasdiff": np.arange(-1, 1.1, 0.1),
    SWE: np.arange(0, 210, 10),
    SWE + "bias": np.arange(-150, 160, 10),
    SWE + "biasdiff": np.arange(-15, 16, 1),
    LAKE_ICE_FRACTION: np.arange(0, 1.1, 0.1),
    LAKE_ICE_FRACTION + "bias": np.arange(0, 1.1, 0.1),
    LAKE_ICE_FRACTION + "biasdiff": np.arange(0, 1.1, 0.1),
}



internal_name_to_cmap = {
    T_AIR_2M: cm.get_cmap("bwr", len(internal_name_to_clevs[T_AIR_2M]) - 1),
    TOTAL_PREC: cm.get_cmap("viridis", len(internal_name_to_clevs[TOTAL_PREC]) - 1),
    SWE: cm.get_cmap("viridis", len(internal_name_to_clevs[SWE]) - 1)
}


def get_clevs(internal_name):
    if internal_name in internal_name_to_clevs:
        return internal_name_to_clevs[internal_name]
    return None


internal_name_to_multiplier = defaultdict(lambda : 1)



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



def get_land_fraction(run_config: RunConfig):
    base_dir = Path(run_config.data_path)

    import os

    file_list = []

    for flev1 in base_dir.iterdir():
        if flev1.is_dir():
            for flev2 in flev1.iterdir():
                if flev2.name.startswith("pm"):
                    file_list.append(flev2)
        else:
            if flev1.name.startswith("pm"):
                file_list.append(flev1)


    first_file = next(sorted(file_list, key=lambda p: os.path.getmtime(str(p))))

    with RPN(str(first_file)) as r:
        return r.get_first_record_for_name("MG")









@main_decorator
def main(vars_of_interest=None):
    # Validation with CRU (temp, precip) and CMC SWE

    # obs_data_path = Path("/RESCUE/skynet3_rech1/huziy/obs_data_for_HLES/interploated_to_the_same_grid/GL_0.1_452x260/anusplin+_interpolated_tt_pr.nc")
    obs_data_path = Path("/HOME/huziy/skynet3_rech1/obs_data/mh_churchill_nelson_obs_fields")
    CRU_PRECIP = True

    sim_id = "mh_0.44"
    add_shp_files = [
        default_domains.MH_BASINS_PATH,
        constants.upstream_station_boundaries_shp_path[sim_id]
    ]


    start_year = 1981
    end_year = 2009

    MODEL_LABEL =  "CRCM5 (0.44)"
    # critical p-value for the ttest aka significance level
    # p_crit = 0.05
    p_crit = 1

    coastlines_width = 0.3

    vars_of_interest_default = [
        # T_AIR_2M,
        TOTAL_PREC,
        # SWE,
        # LAKE_ICE_FRACTION
    ]

    if vars_of_interest is None:
        vars_of_interest = vars_of_interest_default


    vname_to_seasonmonths_map = {
        SWE: OrderedDict([("DJF", [12, 1, 2])]),
        T_AIR_2M: season_to_months,
        TOTAL_PREC: OrderedDict([("Annual", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])]) # season_to_months,

    }

    sim_configs = {

        MODEL_LABEL: RunConfig(data_path="/RECH2/huziy/BC-MH/bc_mh_044deg/Samples",
                  start_year=start_year, end_year=end_year, label=MODEL_LABEL),

    }


    grid_config = default_domains.bc_mh_044




    sim_labels = [MODEL_LABEL, ]

    vname_to_level = {
        T_AIR_2M: VerticalLevel(1, level_kinds.HYBRID),
        U_WE: VerticalLevel(1, level_kinds.HYBRID),
        V_SN: VerticalLevel(1, level_kinds.HYBRID),
        SWE: VerticalLevel(-1, level_kinds.ARBITRARY)
    }

    vname_map = {
        default_varname_mappings.TOTAL_PREC: "pre",
        default_varname_mappings.T_AIR_2M: "tmp",
        default_varname_mappings.SWE: "SWE"
    }

    filename_prefix_mapping = {
        default_varname_mappings.SWE: "pm",
        default_varname_mappings.TOTAL_PREC: "pm",
        default_varname_mappings.T_AIR_2M: "dm"
    }


    # Try to get the land_fraction for masking if necessary
    land_fraction = None
    try:
        land_fraction = get_land_fraction(sim_configs[MODEL_LABEL])
    except Exception:
        pass



    # Calculations

    # prepare params for interpolation
    lons_t, lats_t, bsmap = get_target_lons_lats_basemap(sim_configs[MODEL_LABEL])

    bsmap, reg_of_interest_mask = grid_config.get_basemap_using_shape_with_polygons_of_interest(lons=lons_t, lats=lats_t,
                                                                                                shp_path=default_domains.MH_BASINS_PATH,
                                                                                                mask_margin=2, resolution="i")

    xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons_t.flatten(), lats_t.flatten())










    obs_multipliers = default_varname_mappings.vname_to_multiplier_CRCM5.copy()

    # Read and calculate observed seasonal means
    store_config = {
            "base_folder": obs_data_path.parent if not obs_data_path.is_dir() else obs_data_path,
            "data_source_type": data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES_OPEN_EACH_FILE_SEPARATELY,
            "varname_mapping": vname_map,
            "level_mapping": vname_to_level,
            "offset_mapping": default_varname_mappings.vname_to_offset_CRCM5,
            "multiplier_mapping": obs_multipliers,
    }

    obs_dm = DataManager(store_config=store_config)
    obs_data = {}


    # need to save it for ttesting
    obs_vname_to_season_to_std = {}
    obs_vname_to_season_to_nobs = {}

    interp_indices = None
    for vname in vars_of_interest:
        # --
        end_year_for_current_var = end_year
        if vname == SWE:
            end_year_for_current_var = min(1996, end_year)

        # --
        seas_to_year_to_mean = obs_dm.get_seasonal_means(varname_internal=vname,
                                                     start_year=start_year,
                                                     end_year=end_year_for_current_var,
                                                     season_to_months=vname_to_seasonmonths_map[vname])





        seas_to_clim = {seas: np.array(list(y_to_means.values())).mean(axis=0) for seas, y_to_means in seas_to_year_to_mean.items()}

        # convert precip from mm/month (CRU) to mm/day
        if vname in [TOTAL_PREC] and CRU_PRECIP:
            for seas in seas_to_clim:
                seas_to_clim[seas] *= 1. / (365.25 / 12)
                seas_to_clim[seas] = np.ma.masked_where(np.isnan(seas_to_clim[seas]), seas_to_clim[seas])


                print("{}: min={}, max={}".format(seas, seas_to_clim[seas].min(), seas_to_clim[seas].max()))


        obs_data[vname] = seas_to_clim

        if interp_indices is None:
            _, interp_indices = obs_dm.get_kdtree().query(list(zip(xt, yt, zt)))




        # need for ttests
        season_to_std = {}
        obs_vname_to_season_to_std[vname] = season_to_std

        season_to_nobs = {}
        obs_vname_to_season_to_nobs[vname] = season_to_nobs

        for season in seas_to_clim:
            seas_to_clim[season] = seas_to_clim[season].flatten()[interp_indices].reshape(lons_t.shape)



            # save the yearly means for ttesting
            season_to_std[season] = np.asarray([field.flatten()[interp_indices].reshape(lons_t.shape)
                                                         for field in seas_to_year_to_mean[season].values()]).std(axis=0)


            season_to_nobs[season] = np.ones_like(lons_t) * len(seas_to_year_to_mean[season])


        plt.show()



    # Read and calculate simulated seasonal mean biases
    mod_label_to_vname_to_season_to_std = {}
    mod_label_to_vname_to_season_to_nobs = {}

    model_data_multipliers = defaultdict(lambda: 1)
    model_data_multipliers[TOTAL_PREC] = 1000 * 24 * 3600

    sim_data = defaultdict(dict)
    for label, r_config in sim_configs.items():

        store_config = {
                "base_folder": r_config.data_path,
                "data_source_type": data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT,
                "varname_mapping": default_varname_mappings.vname_map_CRCM5,
                "level_mapping": vname_to_level,
                "offset_mapping": default_varname_mappings.vname_to_offset_CRCM5,
                "multiplier_mapping": model_data_multipliers,
                "filename_prefix_mapping": filename_prefix_mapping
        }


        dm = DataManager(store_config=store_config)

        mod_label_to_vname_to_season_to_std[label] = {}
        mod_label_to_vname_to_season_to_nobs[label] = {}


        interp_indices = None
        for vname in vars_of_interest:

            # --
            end_year_for_current_var = end_year
            if vname == SWE:
                end_year_for_current_var = min(1996, end_year)

            # --
            seas_to_year_to_mean = dm.get_seasonal_means(varname_internal=vname,
                                                         start_year=start_year,
                                                         end_year=end_year_for_current_var,
                                                         season_to_months=vname_to_seasonmonths_map[vname])


            # get the climatology
            seas_to_clim = {seas: np.array(list(y_to_means.values())).mean(axis=0) for seas, y_to_means in seas_to_year_to_mean.items()}

            sim_data[label][vname] = seas_to_clim



            if interp_indices is None:
                _, interp_indices = dm.get_kdtree().query(list(zip(xt, yt, zt)))


            season_to_std = {}
            mod_label_to_vname_to_season_to_std[label][vname] = season_to_std

            season_to_nobs = {}
            mod_label_to_vname_to_season_to_nobs[label][vname] = season_to_nobs

            for season in seas_to_clim:
                interpolated_field = seas_to_clim[season].flatten()[interp_indices].reshape(lons_t.shape)
                seas_to_clim[season] = interpolated_field - obs_data[vname][season]

                # calculate standard deviations of the interpolated fields
                season_to_std[season] = np.asarray([field.flatten()[interp_indices].reshape(lons_t.shape) for field in seas_to_year_to_mean[season].values()]).std(axis=0)

                # calculate numobs for the ttest
                season_to_nobs[season] = np.ones_like(lons_t) * len(seas_to_year_to_mean[season])






    xx, yy = bsmap(lons_t, lats_t)
    lons_t[lons_t > 180] -= 360

    field_mask = maskoceans(lons_t, lats_t, np.zeros_like(lons_t)).mask


    for vname in vars_of_interest:

        if vname not in [SWE]:
            field_mask = np.zeros_like(field_mask, dtype=bool)


        # Plotting: interpolate to the same grid and plot obs and biases
        plot_utils.apply_plot_params(width_cm=32 / 4 * (len(vname_to_seasonmonths_map[vname])),
                                     height_cm=25 / 3.0 * (len(sim_configs) + 1), font_size=8 * len(vname_to_seasonmonths_map[vname]))

        fig = plt.figure()

        # fig.suptitle(internal_name_to_title[vname] + "\n")

        nrows = len(sim_configs) + 2
        ncols = len(vname_to_seasonmonths_map[vname])
        gs = GridSpec(nrows=nrows, ncols=ncols)



        # Plot the obs fields
        current_row = 0
        for col, season in enumerate(vname_to_seasonmonths_map[vname]):
            field = obs_data[vname][season]
            ax = fig.add_subplot(gs[current_row, col])
            ax.set_title(season)

            to_plot = np.ma.masked_where(field_mask, field) * internal_name_to_multiplier[vname]
            clevs = get_clevs(vname)

            to_plot = np.ma.masked_where(~reg_of_interest_mask, to_plot)

            if clevs is not None:
                bnorm = BoundaryNorm(clevs, len(clevs) - 1)
                cmap = cm.get_cmap("Blues", len(clevs) - 1)
            else:
                cmap = "jet"
                bnorm = None

            bsmap.drawmapboundary(fill_color="0.75")

            # cs = bsmap.contourf(xx, yy, to_plot, ax=ax, levels=get_clevs(vname), norm=bnorm, cmap=cmap)
            cs = bsmap.pcolormesh(xx, yy, to_plot, ax=ax, norm=bnorm, cmap=internal_name_to_cmap[vname])

            bsmap.drawcoastlines(linewidth=coastlines_width)
            # bsmap.drawstates(linewidth=0.1)
            # bsmap.drawcountries(linewidth=0.2)
            bsmap.colorbar(cs, ax=ax)

            i = 0
            bsmap.readshapefile(str(add_shp_files[i])[:-4], "field_{}".format(i), linewidth=0.5, color="m")


            if col == 0:
                ax.set_ylabel("Obs")



        # plot the biases
        for sim_label in sim_labels:
            current_row += 1
            for col, season in enumerate(vname_to_seasonmonths_map[vname]):

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


                # ttest
                a = sim_data[sim_label][vname][season] + obs_data[vname][season]  # Calculate the simulation data back from biases
                std_a = mod_label_to_vname_to_season_to_std[sim_label][vname][season]
                nobs_a = mod_label_to_vname_to_season_to_nobs[sim_label][vname][season]

                b = obs_data[vname][season]
                std_b =  obs_vname_to_season_to_std[vname][season]
                nobs_b = obs_vname_to_season_to_nobs[vname][season]



                t, p = ttest_ind_from_stats(mean1=a, std1=std_a, nobs1=nobs_a,
                                            mean2=b, std2=std_b, nobs2=nobs_b, equal_var=False)

                # Mask non-significant differences as given by the ttest
                to_plot = np.ma.masked_where(p > p_crit, to_plot)

                # only focus on the basins of interest
                to_plot = np.ma.masked_where(~reg_of_interest_mask, to_plot)


                # cs = bsmap.contourf(xx, yy, to_plot, ax=ax, extend="both", levels=get_clevs(vname + "bias"), cmap=cmap, norm=bnorm)

                bsmap.drawmapboundary(fill_color="0.75")


                cs = bsmap.pcolormesh(xx, yy, to_plot, ax=ax, cmap=cmap, norm=bnorm)
                bsmap.drawcoastlines(linewidth=coastlines_width)
                bsmap.colorbar(cs, ax=ax, extend="both")





                for i, shp in enumerate(add_shp_files[1:], start=1):
                    bsmap.readshapefile(str(shp)[:-4], "field_{}".format(i), linewidth=0.5, color="k")

                if col == 0:
                    ax.set_ylabel("{}\n-\nObs.".format(sim_label))




        fig.tight_layout()



        # save a figure per variable
        img_file = "seasonal_biases_{}_{}_{}-{}.png".format(vname,
                                                            "-".join([s for s in vname_to_seasonmonths_map[vname]]),
                                                            start_year, end_year)


        if not img_folder.exists():
            img_folder.mkdir(parents=True)

        img_file = img_folder / img_file
        fig.savefig(str(img_file), bbox_inches="tight", dpi=300)

        plt.close(fig)



if __name__ == '__main__':
    main(vars_of_interest=[TOTAL_PREC])
    # main(vars_of_interest=[T_AIR_2M])
    # main(vars_of_interest=[SWE])
