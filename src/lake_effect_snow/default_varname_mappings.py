from collections import defaultdict


mps_per_knot = 0.514444

T_AIR_2M = "t_air_2m"
TOTAL_PREC = "total_prec"
SNOWFALL_RATE = "snow_fall"

U_WE = "u_we"
V_SN = "v_sn"


# maps the internal variable name to the variable names in files
vname_map_CRCM5 = {
    T_AIR_2M: "TT",
    TOTAL_PREC: "PR",
    SNOWFALL_RATE: "SN",
    U_WE: "UU",
    V_SN: "VV"

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


vname_map_netcdf = {

}



