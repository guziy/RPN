from collections import defaultdict

from rpn import level_kinds

from lake_effect_snow.base_utils import VerticalLevel

mps_per_knot = 0.514444

T_AIR_2M = "t_air_2m"
T_AIR_2M_DAILY_MAX = "t_air_2m_daily_max"
T_AIR_2M_DAILY_MIN = "t_air_2m_daily_min"
T_AIR_2M_DAILY_AVG = "t_air_2m_daily_avg"

TOTAL_PREC = "total_prec"
SNOWFALL_RATE = "snow_fall"
SENSIBLE_HF = "sensible_heat_flux"
LATENT_HF = "latent_heat_flux"
LWRAD_DOWN = "longwave_rad_down"
SWRAD_DOWN = "shortwave_rad_down"
LAKE_WATER_TEMP = "lake_water_temp"


U_WE = "u_we"
V_SN = "v_sn"

LAKE_ICE_FRACTION = "lake_ice_fraction"
SWE = "snow_water_equivalent"


# maps the internal variable name to the variable names in files
vname_map_CRCM5 = {
    T_AIR_2M: "TT",
    TOTAL_PREC: "PR",
    SNOWFALL_RATE: "U3",
    U_WE: "UU",
    V_SN: "VV",
    LAKE_ICE_FRACTION: "LC",
    SWE: "I5",
    SENSIBLE_HF: "FC",
    LATENT_HF: "FV",
    LWRAD_DOWN: "AD",
    SWRAD_DOWN: "N4",
    LAKE_WATER_TEMP: "L1",


}


def get_default_multiplier():
    return 1


def get_default_offset():
    return 0


def get_default_file_prefix():
    return "dm"



# Maps varnames to multipliers for unit conversions
vname_to_multiplier_CRCM5 = defaultdict(get_default_multiplier)
vname_to_multiplier_CRCM5[U_WE] = mps_per_knot
vname_to_multiplier_CRCM5[V_SN] = mps_per_knot


# Maps varnames to offsets for unit conversions
vname_to_offset_CRCM5 = defaultdict(get_default_offset)


# variable name to the prefix of a file mapping
vname_to_fname_prefix_CRCM5 = defaultdict(get_default_file_prefix)
vname_to_fname_prefix_CRCM5[SNOWFALL_RATE] = "pm"
vname_to_fname_prefix_CRCM5[TOTAL_PREC] = "pm"
vname_to_fname_prefix_CRCM5[T_AIR_2M] = "dm"
vname_to_fname_prefix_CRCM5[LAKE_ICE_FRACTION] = "pm"
vname_to_fname_prefix_CRCM5[U_WE] = "dm"
vname_to_fname_prefix_CRCM5[V_SN] = "dm"


vname_map_netcdf = {

}

vname_to_level_map = {
    T_AIR_2M: VerticalLevel(1, level_kinds.HYBRID),
    U_WE: VerticalLevel(1, level_kinds.HYBRID),
    V_SN: VerticalLevel(1, level_kinds.HYBRID),
    TOTAL_PREC: VerticalLevel(0, level_type=level_kinds.PRESSURE)
}

# For daymet obs dataset
daymet_vname_mapping = {
    TOTAL_PREC: "prcp",
    T_AIR_2M_DAILY_AVG: "tavg",
    T_AIR_2M_DAILY_MIN: "tmin",
    T_AIR_2M_DAILY_MAX: "tmax"
}