from collections import defaultdict

import numpy as np
from pendulum import datetime, Period
from rpn import level_kinds

from data.robust import data_source_types
from data.robust.data_manager import DataManager
from lake_effect_snow import default_varname_mappings
from lake_effect_snow.base_utils import VerticalLevel
from util.geo.mask_from_shp import get_mask


import logging
logging.basicConfig()
logger = logging.getLogger(__name__)


all_known_variables = [
    default_varname_mappings.HLES_AMOUNT,
    default_varname_mappings.HLES_FREQUENCY,
    default_varname_mappings.T_AIR_2M,
    default_varname_mappings.TOTAL_PREC,
    default_varname_mappings.LAKE_ICE_FRACTION,
    default_varname_mappings.CAO
]


def get_data(vname=default_varname_mappings.T_AIR_2M,
             season_to_months=None,
             data_query=None):
    """
    Get seasonal means for each year for vname, obs and mod
    :param data_query: dict, specifies root_dir, beg_year, end_year for mod and obs
    :param season_to_months:
    :param vname:
    """

    if season_to_months is None:
        raise ValueError("The season name to corresponding months mapping should be provided")

    obs_dir = data_query["obs"]["root_dir"]
    obs_beg_year = data_query["obs"]["beg_year"]
    obs_end_year = data_query["obs"]["end_year"]

    mod_dir = data_query["mod"]["root_dir"]
    mod_beg_year = data_query["mod"]["beg_year"]
    mod_end_year = data_query["mod"]["end_year"]

    res = {
        "mod": None, "obs": None
    }

    if vname not in all_known_variables:
        raise ValueError(f"Unknown variable {vname}")

    vname_map = {
        default_varname_mappings.T_AIR_2M: "TT",
        default_varname_mappings.TOTAL_PREC: "PR",
        default_varname_mappings.LAKE_ICE_FRACTION: "LC",
        default_varname_mappings.HLES_AMOUNT: "hles_snow"
    }

    level_map = {v: VerticalLevel(1, level_kinds.HYBRID) for v in all_known_variables}

    if "obs_multipliers" not in data_query["obs"]:
        obs_multipliers = {
            default_varname_mappings.TOTAL_PREC: 1.,  # converted to mm/day on netcdf export
            default_varname_mappings.T_AIR_2M: 1.,
            default_varname_mappings.LAKE_ICE_FRACTION: 1.,
            default_varname_mappings.HLES_AMOUNT: 100.  # M/day -> cm/day
        }
    else:
        obs_multipliers = data_query["obs"]["multipliers"]

    if not "mod_multipliers" in data_query["mod"]:
        mod_multipliers = obs_multipliers.copy()
        mod_multipliers[default_varname_mappings.TOTAL_PREC] = 1.  # converted to mm/day on netcdf export
    else:
        mod_multipliers = data_query["mod"]["multipliers"]

    offset_map = defaultdict(lambda: 0)
    obs_store_config = {
        DataManager.SP_BASE_FOLDER: obs_dir,
        DataManager.SP_DATASOURCE_TYPE: data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES_OPEN_EACH_FILE_SEPARATELY,
        DataManager.SP_INTERNAL_TO_INPUT_VNAME_MAPPING: vname_map,
        DataManager.SP_OFFSET_MAPPING: offset_map,
        DataManager.SP_MULTIPLIER_MAPPING: obs_multipliers,
        DataManager.SP_LEVEL_MAPPING: level_map
    }

    mod_store_config = obs_store_config.copy()
    mod_store_config[DataManager.SP_BASE_FOLDER] = mod_dir
    mod_store_config[DataManager.SP_MULTIPLIER_MAPPING] = mod_multipliers

    logger.debug(obs_store_config)
    logger.debug(mod_store_config)

    dm_obs = DataManager(store_config=obs_store_config)
    dm_mod = DataManager(store_config=mod_store_config)

    beg_dummy = datetime(data_query["obs"]["beg_year"], 1, 1)
    vname_dummy = vname

    # calculate seasonal means
    if vname in [default_varname_mappings.HLES_FREQUENCY]:
        res["mod"] = dm_mod.get_mean_number_of_hles_days(mod_beg_year, mod_end_year, season_to_months=season_to_months,
                                                         hles_vname=default_varname_mappings.HLES_AMOUNT)

        res["obs"] = dm_obs.get_mean_number_of_hles_days(obs_beg_year, obs_end_year, season_to_months=season_to_months,
                                                         hles_vname=default_varname_mappings.HLES_AMOUNT)
        vname_dummy = default_varname_mappings.HLES_AMOUNT

    elif vname in [default_varname_mappings.CAO]:
        res["mod"] = dm_mod.get_mean_number_of_cao_days(mod_beg_year, mod_end_year, season_to_months=season_to_months,
                                                        temperature_vname=default_varname_mappings.T_AIR_2M)

        res["obs"] = dm_obs.get_mean_number_of_cao_days(obs_beg_year, obs_end_year, season_to_months=season_to_months,
                                                        temperature_vname=default_varname_mappings.T_AIR_2M)

        vname_dummy = default_varname_mappings.T_AIR_2M

    else:
        res["mod"] = dm_mod.get_seasonal_means(start_year=mod_beg_year, end_year=mod_end_year,
                                               season_to_months=season_to_months,
                                               varname_internal=vname)

        res["obs"] = dm_obs.get_seasonal_means(start_year=obs_beg_year, end_year=obs_end_year,
                                               season_to_months=season_to_months,
                                               varname_internal=vname)

    # read some data to get coordinates
    dm_obs.read_data_for_period(Period(beg_dummy, beg_dummy.add(months=1)),
                                varname_internal=vname_dummy)

    gl_mask = get_mask(lons2d=dm_obs.lons, lats2d=dm_obs.lats,
                       shp_path="data/shp/Great_Lakes/Great_Lakes.shp") > 0.5

    # mask great lakes or everything except the GL depending on the variable
    if vname in [default_varname_mappings.LAKE_ICE_FRACTION]:
        gl_mask = ~gl_mask

    for case in res:
        for season, data in res[case].items():
            for year in data:
                data[year] = np.ma.masked_where(gl_mask, data[year])

    return res, dm_obs.lons, dm_obs.lats