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


# Maps varnames to multipliers for unit conversions
vname_to_multiplier_CRCM5 = defaultdict(lambda: 1)
vname_to_multiplier_CRCM5[U_WE] = mps_per_knot
vname_to_multiplier_CRCM5[V_SN] = mps_per_knot

# Maps varnames to offsets for unit conversions
vname_to_offset_CRCM5 = defaultdict(lambda: 0)



vname_map_netcdf = {

}