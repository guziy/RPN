# annual number of summer days (i.e. the number of days when TX > 25C)

import dask
import matplotlib
from memory_profiler import profile
from scipy.stats import ttest_ind_from_stats

from crcm5.basic_validation.plot_area_averages import plot_area_avg
from crcm5.basic_validation.plot_monthly_panels_from_daily_clim_xarray import plot_monthly_panels
from data.robust import data_source_types
from data.robust.data_manager import DataManager
from lake_effect_snow import default_varname_mappings


from collections import OrderedDict
from pathlib import Path

from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap, maskoceans
from rpn import level_kinds

from crcm5.basic_validation.diag_manager import DiagCrcmManager
from cru.temperature import CRUDataManager
from data.highres_data_manager import HighResDataManager
from lake_effect_snow.base_utils import VerticalLevel
from util import plot_utils
from util.seasons_info import MonthPeriod

import numpy as np
import matplotlib.pyplot as plt

var_name_to_level = {
    "TT": VerticalLevel(1, level_type=level_kinds.HYBRID),
    "PR": VerticalLevel(-1, level_type=level_kinds.ARBITRARY)
}

clevs = {
    "mean": {
        default_varname_mappings.T_AIR_2M: np.arange(-40, 42, 2),
        default_varname_mappings.T_AIR_2M_DAILY_MIN: np.arange(-40, 42, 2),
        default_varname_mappings.T_AIR_2M_DAILY_MAX: np.arange(-40, 42, 2),
        "PR_max": np.arange(0, 10.5, 0.5),
        default_varname_mappings.TOTAL_PREC: np.arange(0, 10.5, 0.5),
        default_varname_mappings.T_AIR_2M + "diff": list(np.arange(-5, 0, 1)) + list(np.arange(1, 6, 1)),
        default_varname_mappings.TOTAL_PREC + "diff": list(np.arange(-5, -0.5, 1)) + [-0.5, 0.5] + list(np.arange(1, 6, 1))

    },
    "std": {
        "TT": list(np.arange(-3, -0.0, 0.5)) + [-0.1, 0.1] + list(np.arange(0.5, 3.5, 0.5)),
        "PR": list(np.arange(-3, -0.0, 0.5)) + [-0.1, 0.1] + list(np.arange(0.5, 3.5, 0.5))
    }
}

cmaps = {
    "mean": {
        default_varname_mappings.T_AIR_2M: "bwr",
        default_varname_mappings.T_AIR_2M_DAILY_MAX: "bwr",
        default_varname_mappings.T_AIR_2M_DAILY_MIN: "bwr",
        default_varname_mappings.T_AIR_2M_DAILY_AVG: "bwr",
        default_varname_mappings.TOTAL_PREC: "jet_r",
        default_varname_mappings.T_AIR_2M + "diff": "bwr",
        default_varname_mappings.TOTAL_PREC + "diff": "bwr",
    },
    "std": {
        default_varname_mappings.T_AIR_2M: "jet",
        default_varname_mappings.TOTAL_PREC: "jet"
    }
}
var_name_to_cru_name = {
    "TT": "tmp", "PR": "pre"
}

var_name_to_file_prefix = {
    "TT": "dm", "PR": "pm"
}

var_name_to_mul_default = {
    "TT": 1, "PR": 1000 * 24 * 3600
}

area_thresh_km2 = 5000



def main():
    # dask.set_options(pool=ThreadPool(20))
    img_folder = Path("nei_validation")
    img_folder.mkdir(parents=True, exist_ok=True)

    pval_crit = 0.1

    start_year = 1980
    end_year = 1998


    # TT_min and TT_max mean daily min and maximum temperatures
    var_names = [
        default_varname_mappings.T_AIR_2M_DAILY_MAX,
        default_varname_mappings.T_AIR_2M_DAILY_MIN,
        default_varname_mappings.TOTAL_PREC
    ]

    var_name_to_rolling_window_days = {
        default_varname_mappings.T_AIR_2M_DAILY_MIN: 5,
        default_varname_mappings.T_AIR_2M_DAILY_MAX: 5,
        default_varname_mappings.TOTAL_PREC: 29
    }

    var_name_to_percentile = {
        default_varname_mappings.T_AIR_2M_DAILY_MIN: 0.9,
        default_varname_mappings.T_AIR_2M_DAILY_MAX: 0.1,
        default_varname_mappings.TOTAL_PREC: 0.9,
    }

    # needed for the 3hourly temperature model outputs, when Tmin and Tmax daily are not available
    var_name_to_daily_agg_func = {
        default_varname_mappings.TOTAL_PREC: np.mean,
        default_varname_mappings.T_AIR_2M_DAILY_MAX: np.max,
        default_varname_mappings.T_AIR_2M_DAILY_MIN: np.min,
        default_varname_mappings.T_AIR_2M_DAILY_AVG: np.mean
    }




    model_vname_to_multiplier = {
        default_varname_mappings.TOTAL_PREC: 1000 * 24 * 3600
    }


    WC_044_DEFAULT_LABEL = "WC_0.44deg_default"
    WC_044_CTEM_FRSOIL_DYNGLA_LABEL = "WC_0.44deg_ctem+frsoil+dyngla"
    WC_011_CTEM_FRSOIL_DYNGLA_LABEL = "WC_0.11deg_ctem+frsoil+dyngla"

    sim_paths = OrderedDict()
    sim_paths[WC_011_CTEM_FRSOIL_DYNGLA_LABEL] = Path("/snow3/huziy/NEI/WC/NEI_WC0.11deg_Crr1/Samples")
    sim_paths[WC_044_DEFAULT_LABEL] = Path("/snow3/huziy/NEI/WC/NEI_WC0.44deg_default/Samples")
    sim_paths[WC_044_CTEM_FRSOIL_DYNGLA_LABEL] = Path("/snow3/huziy/NEI/WC/debug_NEI_WC0.44deg_Crr1/Samples")


    mod_spatial_scales = OrderedDict([
        (WC_044_DEFAULT_LABEL, 0.44),
        (WC_044_CTEM_FRSOIL_DYNGLA_LABEL, 0.44),
        (WC_011_CTEM_FRSOIL_DYNGLA_LABEL, 0.11)
    ])

    # -- daymet daily (initial spatial res)
    # daymet_vname_to_path = {
    #     "prcp": "/snow3/huziy/Daymet_daily/daymet_v3_prcp_*_na.nc4",
    #     "tavg": "/snow3/huziy/Daymet_daily/daymet_v3_tavg_*_na.nc4",
    #     "tmin": "/snow3/huziy/Daymet_daily/daymet_v3_tmin_*_na.nc4",
    #     "tmax": "/snow3/huziy/Daymet_daily/daymet_v3_tmax_*_na.nc4",
    # }

    # -- daymet daily (spatially aggregated)
    daymet_vname_to_path = {
        default_varname_mappings.TOTAL_PREC: "/snow3/huziy/Daymet_daily_derivatives/daymet_spatial_agg_prcp_10x10",
        default_varname_mappings.T_AIR_2M_DAILY_AVG: "/snow3/huziy/Daymet_daily_derivatives/daymet_spatial_agg_tavg_10x10",
        default_varname_mappings.T_AIR_2M_DAILY_MIN: "/snow3/huziy/Daymet_daily_derivatives/daymet_spatial_agg_tmin_10x10",
        default_varname_mappings.T_AIR_2M_DAILY_MAX: "/snow3/huziy/Daymet_daily_derivatives/daymet_spatial_agg_tmax_10x10",
    }

    daymet_vname_to_model_vname_internal = {
        default_varname_mappings.T_AIR_2M_DAILY_MIN: default_varname_mappings.T_AIR_2M,
        default_varname_mappings.T_AIR_2M_DAILY_MAX: default_varname_mappings.T_AIR_2M,
        default_varname_mappings.TOTAL_PREC: default_varname_mappings.TOTAL_PREC,
    }

    plot_utils.apply_plot_params(font_size=8)


    # observations
    obs_spatial_scale = 0.1  # 10x10 aggregation from ~0.01 daymet data


    varnames_list = [
        default_varname_mappings.TOTAL_PREC,
        default_varname_mappings.T_AIR_2M_DAILY_MIN,
        default_varname_mappings.T_AIR_2M_DAILY_MAX
    ]


    data_dict = {vn: {} for vn in varnames_list}
    bias_dict = {vn: {} for vn in varnames_list}

    # calculate the percentiles for each simulation and obs data (obs data interpolated to the model grid)
    for model_label, base_dir in sim_paths.items():
        # model outputs manager
        dm = DataManager(
            store_config={
                DataManager.SP_BASE_FOLDER: base_dir,
                DataManager.SP_DATASOURCE_TYPE: data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT,
                DataManager.SP_INTERNAL_TO_INPUT_VNAME_MAPPING: default_varname_mappings.vname_map_CRCM5,
                DataManager.SP_LEVEL_MAPPING: default_varname_mappings.vname_to_level_map,
                DataManager.SP_VARNAME_TO_FILENAME_PREFIX_MAPPING: default_varname_mappings.vname_to_fname_prefix_CRCM5
            }
        )



        for vname_daymet in varnames_list:



            obs_manager = DataManager(
                store_config={
                    DataManager.SP_BASE_FOLDER: daymet_vname_to_path[vname_daymet],
                    DataManager.SP_DATASOURCE_TYPE: data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES,
                    DataManager.SP_INTERNAL_TO_INPUT_VNAME_MAPPING: default_varname_mappings.daymet_vname_mapping,
                    DataManager.SP_LEVEL_MAPPING: {}
                }
            )

            vname_model = daymet_vname_to_model_vname_internal[vname_daymet]

            nd_rw = var_name_to_rolling_window_days[vname_daymet]
            q = var_name_to_percentile[vname_daymet]
            daily_agg_func = var_name_to_daily_agg_func[vname_daymet]



            # model data
            # TODO: change for the number of summer days
            mod = dm.compute_climatological_quantiles(start_year=start_year, end_year=end_year,
                                                      daily_agg_func=daily_agg_func,
                                                      rolling_mean_window_days=nd_rw,
                                                      q=q,
                                                      varname_internal=vname_model)


            mod = mod * model_vname_to_multiplier.get(vname_model, 1)

            data_source_mod = f"{model_label}_ndrw{nd_rw}_q{q}_vn{vname_daymet}_{start_year}-{end_year}"



            # obs data
            nneighbors = int(mod_spatial_scales[model_label] / obs_spatial_scale)
            nneighbors = max(nneighbors, 1)


            obs = obs_manager.compute_climatological_quantiles(start_year=start_year,
                                                               end_year=end_year,
                                                               daily_agg_func=daily_agg_func,  # does not have effect for daymet data because it is daily
                                                               rolling_mean_window_days=nd_rw,
                                                               q=q,
                                                               varname_internal=vname_daymet,
                                                               lons_target=mod.coords["lon"].values,
                                                               lats_target=mod.coords["lat"].values,
                                                               nneighbors=nneighbors)


            # only use model data wherever the obs is not null
            mod = mod.where(obs.notnull())



            data_source_obs = f"DAYMETaggfor_{model_label}_ndrw{nd_rw}_q{q}_vn{vname_daymet}_{start_year}-{end_year}"

            data_source_diff = f"{model_label}vsDAYMET_ndrw{nd_rw}_q{q}_vn{vname_daymet}_{start_year}-{end_year}"


            # save data for line plots
            data_dict[vname_daymet][data_source_mod] = mod
            data_dict[vname_daymet][data_source_obs] = obs
            bias_dict[vname_daymet][data_source_mod] = mod - obs


            bmap = dm.get_basemap(varname_internal=vname_model, resolution="i", area_thresh=area_thresh_km2)


            # plot model data
            plot_monthly_panels(mod, bmap, img_dir=str(img_folder), data_label=data_source_mod,
                                color_levels=clevs["mean"][vname_model], cmap=cmaps["mean"][vname_model])






            # plot obs data
            plot_monthly_panels(obs, bmap, img_dir=str(img_folder), data_label=data_source_obs,
                                color_levels=clevs["mean"][vname_model], cmap=cmaps["mean"][vname_model])


            plot_monthly_panels(mod - obs, bmap, img_dir=str(img_folder), data_label=data_source_diff,
                                color_levels=clevs["mean"][vname_model + "diff"], cmap=cmaps["mean"][vname_model + "diff"])





    for vn in data_dict:

        if len(data_dict[vn]) == 0:
            continue

        plot_area_avg(data_dict[vn], bias_dict[vn], panel_titles=(vn, ""), img_dir=img_folder / "extremes_1d")




if __name__ == '__main__':
    main()