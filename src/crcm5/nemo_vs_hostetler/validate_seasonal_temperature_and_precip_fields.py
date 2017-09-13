




# compare temperature and total precip biases in the NEMO- and Hostetler-based simulations
from collections import OrderedDict
from collections import defaultdict
from pathlib import Path, PurePath

import matplotlib
from matplotlib.axes import Axes

from rpn_utils.get_coord_data_from_rpn_file import IndexSubspace

matplotlib.use("agg")

from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import maskoceans
from rpn import level_kinds
from rpn.domains.rotated_lat_lon import RotatedLatLon
from rpn.rpn import RPN

from application_properties import main_decorator
from crcm5.analyse_hdf.run_config import RunConfig
from data.robust import data_source_types

from lake_effect_snow import default_varname_mappings
from lake_effect_snow.base_utils import VerticalLevel
from data.robust.data_manager import DataManager
from lake_effect_snow.default_varname_mappings import T_AIR_2M, U_WE, V_SN, TOTAL_PREC, SWE, LAKE_ICE_FRACTION

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import ttest_ind_from_stats

# season name to month list mapping (the order of months is important)
from util import plot_utils
from util.geo import lat_lon

season_to_months = OrderedDict([
    ("Winter", [12, 1, 2]),
    ("Spring", [3, 4, 5]),
    ("Summer", [6, 7, 8]),
    ("Fall", [9, 10, 11])
])

img_folder = Path("nemo_vs_hostetler")

internal_name_to_title = {
    T_AIR_2M: r"2-m air temperature ($^\circ$C)",
    TOTAL_PREC: "Total recipitation (mm/day)",
    SWE: "SWE (mm)",
    # LAKE_ICE_FRACTION: "lake ice fraction"
}

internal_name_to_clevs = {
    T_AIR_2M: np.arange(-30, 30, 2),
    T_AIR_2M + "bias": np.arange(-4.5, 4.6, 1),
    T_AIR_2M + "biasdiff": np.arange(-4.5, 4.6, 1), # np.arange(-1.05, 1.1, 0.1),
    TOTAL_PREC: np.arange(0, 8, 0.5),
    TOTAL_PREC + "bias": np.arange(-3.25, 3.26, 0.5),
    TOTAL_PREC + "biasdiff": np.arange(-3.25, 3.26, 0.5), # np.arange(-1.1, 1.2, 0.2),
    SWE: [0, 0.1, 1, 10, 20, 50, 100, 150, 200, 250, 300, 400],
    SWE + "bias": np.arange(-105, 110, 10),
    SWE + "biasdiff": np.arange(-45, 55, 10),
    LAKE_ICE_FRACTION: np.arange(0, 1.1, 0.1),
    LAKE_ICE_FRACTION + "bias": np.arange(-1.05, 1.06, 0.1),
    LAKE_ICE_FRACTION + "biasdiff": np.arange(-1.1, 1.1, 0.2),
}


def get_clevs(internal_name):
    if internal_name in internal_name_to_clevs:
        return internal_name_to_clevs[internal_name]
    return None


internal_name_to_multiplier = defaultdict(lambda : 1)
internal_name_to_multiplier[TOTAL_PREC] = 1000.0 * 24 * 3600 # Convert precip to mm/day



def get_target_lons_lats_basemap(run_config: RunConfig=None, sub_space:IndexSubspace=None, **basemap_kwargs):

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

                    if sub_space is None:
                        basemap = rll.get_basemap_object_for_lons_lats(lons2d=lons, lats2d=lats, **basemap_kwargs)
                    else:
                        lons_sel = lons[sub_space.get_islice(), sub_space.get_jslice()]
                        lats_sel = lats[sub_space.get_islice(), sub_space.get_jslice()]
                        basemap = rll.get_basemap_object_for_lons_lats(lons2d=lons_sel, lats2d=lats_sel, **basemap_kwargs)

                    return lons, lats, basemap




def get_land_fraction(first_timestep_file: PurePath):
    with RPN(str(first_timestep_file)) as r:
        return r.get_first_record_for_name("MG")



@main_decorator
def main():

    obs_data_path = Path("/RESCUE/skynet3_rech1/huziy/obs_data_for_HLES/interploated_to_the_same_grid/GL_0.1_452x260/anusplin+_interpolated_tt_pr.nc")

    start_year = 1980
    end_year = 2010

    HL_LABEL = "CRCM5_HL"
    NEMO_LABEL = "CRCM5_NEMO"

    # critical p-value for the ttest aka significance level
    p_crit = 0.1

    vars_of_interest = [
 #       T_AIR_2M,
 #       TOTAL_PREC,
 #       SWE,
        LAKE_ICE_FRACTION
    ]

    coastline_width = 0.3


    vname_to_seasonmonths_map = {
        SWE: OrderedDict([("November", [11]),
                          ("December", [12]),
                          ("January", [1,])]),
        LAKE_ICE_FRACTION: OrderedDict([
                         ("February", [2,]),
                          ("March", [3, ]),]),
        T_AIR_2M: season_to_months,
        TOTAL_PREC:  OrderedDict([
            ("Winter", [12, 1, 2]),
            ("Summer", [6, 7, 8]),
        ])
    }

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


    # Try to get the land_fraction for masking if necessary
    land_fraction = None
    try:
        first_ts_file = Path(sim_configs[HL_LABEL].data_path).parent / "pm1979010100_00000000p"

        land_fraction = get_land_fraction(first_timestep_file=first_ts_file)
    except Exception as err:
        raise err
        pass



    # Calculations

    # prepare params for interpolation
    lons_t, lats_t, bsmap = get_target_lons_lats_basemap(sim_configs[HL_LABEL])

    # get a subdomain of the simulation domain
    nx, ny = lons_t.shape
    iss = IndexSubspace(i_start=20, j_start=20, i_end=nx // 2, j_end=ny/2)
    # just to change basemap limits
    lons_t, lats_t, bsmap = get_target_lons_lats_basemap(sim_configs[HL_LABEL], sub_space=iss, resolution="i", area_thresh=2000)


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




    # Read and calculate simulated seasonal mean biases
    mod_label_to_vname_to_season_to_std = {}
    mod_label_to_vname_to_season_to_nobs = {}

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



    # Plotting: interpolate to the same grid and plot obs and biases



    xx, yy = bsmap(lons_t, lats_t)
    lons_t[lons_t > 180] -= 360


    draw_only_first_sim_biases = True
    for vname in vars_of_interest:

        field_mask = maskoceans(lons_t, lats_t, np.zeros_like(lons_t), inlands=vname in [SWE]).mask
        field_mask_lakes = maskoceans(lons_t, lats_t, np.zeros_like(lons_t), inlands=True).mask

        nrows = len(sim_configs) + 2 - 1 * int(draw_only_first_sim_biases)
        ncols = len(vname_to_seasonmonths_map[vname])

        plot_utils.apply_plot_params(width_cm=8 * len(vname_to_seasonmonths_map[vname]), height_cm=4.5 * nrows, font_size=8)
        fig = plt.figure()



        gs = GridSpec(nrows=nrows, ncols=ncols, hspace=0.2, wspace=0.02)

        extend = "both" if vname not in [LAKE_ICE_FRACTION] else "neither"

        # Plot the obs fields
        current_row = 0
        for col, season in enumerate(vname_to_seasonmonths_map[vname]):
            field = obs_data[vname][season]
            ax = fig.add_subplot(gs[current_row, col])
            # ax.set_title(season)


            the_mask = field_mask_lakes if vname in [T_AIR_2M, TOTAL_PREC, SWE] else field_mask
            to_plot = np.ma.masked_where(the_mask, field) * internal_name_to_multiplier[vname]
            clevs = get_clevs(vname)

            if clevs is not None:
                bnorm = BoundaryNorm(clevs, len(clevs) - 1)
                cmap = cm.get_cmap("viridis", len(clevs) - 1)
            else:
                cmap = "viridis"
                bnorm = None

            cs = bsmap.contourf(xx, yy, to_plot, ax=ax, levels=clevs, norm=bnorm, cmap=cmap)
            bsmap.drawcoastlines(linewidth=coastline_width)
            cb = bsmap.colorbar(cs, ax=ax, location="bottom")

            ax.set_frame_on(vname not in [LAKE_ICE_FRACTION, ])

            cb.ax.set_visible(col == 0)

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

                the_mask = field_mask_lakes if vname in [T_AIR_2M, TOTAL_PREC, SWE] else field_mask
                to_plot = np.ma.masked_where(the_mask, field) * internal_name_to_multiplier[vname]


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


                # temporary plot the actual values

                cs = bsmap.contourf(xx, yy, to_plot, ax=ax, extend=extend, levels=get_clevs(vname + "bias"), cmap=cmap, norm=bnorm)
                bsmap.drawcoastlines(linewidth=coastline_width)
                cb = bsmap.colorbar(cs, ax=ax, location="bottom")

                ax.set_frame_on(vname not in [LAKE_ICE_FRACTION, ])
                cb.ax.set_visible(False)

                if col == 0:
                    ax.set_ylabel("{}\n-\nObs.".format(sim_label))

            # draw biases only for the first simulation
            if draw_only_first_sim_biases:
                break


        # plot differences between the biases
        current_row += 1
        for col, season in enumerate(vname_to_seasonmonths_map[vname]):

            field = sim_data[NEMO_LABEL][vname][season] - sim_data[HL_LABEL][vname][season]

            ax = fig.add_subplot(gs[current_row, col])

            clevs = get_clevs(vname + "bias")
            if clevs is not None:
                bnorm = BoundaryNorm(clevs, len(clevs) - 1)
                cmap = cm.get_cmap("bwr", len(clevs) - 1)
            else:
                cmap = "bwr"
                bnorm = None


            to_plot = field * internal_name_to_multiplier[vname]
            # to_plot = np.ma.masked_where(field_mask, field) * internal_name_to_multiplier[vname]



            # ttest
            a = sim_data[NEMO_LABEL][vname][season] + obs_data[vname][season]  # Calculate the simulation data back from biases
            std_a = mod_label_to_vname_to_season_to_std[NEMO_LABEL][vname][season]
            nobs_a = mod_label_to_vname_to_season_to_nobs[NEMO_LABEL][vname][season]

            b = sim_data[HL_LABEL][vname][season] + obs_data[vname][season]  # Calculate the simulation data back from biases
            std_b = mod_label_to_vname_to_season_to_std[HL_LABEL][vname][season]
            nobs_b = mod_label_to_vname_to_season_to_nobs[HL_LABEL][vname][season]


            t, p = ttest_ind_from_stats(mean1=a, std1=std_a, nobs1=nobs_a,
                                        mean2=b, std2=std_b, nobs2=nobs_b, equal_var=False)

            # Mask non-significant differences as given by the ttest
            to_plot = np.ma.masked_where(p > p_crit, to_plot)


            # mask the points with not sufficient land fraction
            if land_fraction is not None and vname in [SWE, ]:
                to_plot = np.ma.masked_where(land_fraction < 0.1, to_plot)


            # print("land fractions for large differences ", land_fraction[to_plot > 30])


            cs = bsmap.contourf(xx, yy, to_plot, ax=ax, extend=extend, levels=clevs, cmap=cmap, norm=bnorm)
            bsmap.drawcoastlines(linewidth=coastline_width)
            cb = bsmap.colorbar(cs, ax=ax, location="bottom")

            ax.text(0.99, 1.1, season, va="top", ha="right", fontsize=16, transform=ax.transAxes)

            cb.ax.set_visible(col == 0)

            assert isinstance(ax, Axes)
            ax.set_frame_on(False)

            if col == 0:
                ax.set_ylabel("{}\n-\n{}".format(NEMO_LABEL, HL_LABEL))


        # fig.tight_layout()

        # save a figure per variable
        img_file = "seasonal_biases_{}_{}_{}-{}.png".format(vname,
                                                            "-".join([s for s in vname_to_seasonmonths_map[vname]]),
                                                            start_year, end_year)
        img_file = img_folder.joinpath(img_file)

        fig.savefig(str(img_file), dpi=300, bbox_inches="tight")

        plt.close(fig)



if __name__ == '__main__':
    main()
