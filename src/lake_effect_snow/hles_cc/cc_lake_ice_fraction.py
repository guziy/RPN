from collections import OrderedDict
from pathlib import Path


import xarray

#
# plot the map of CC to the maximum lake ice fraction for each grid cell, i.e. *f - *c
#
#
"""
Example of the netcdf file layout:
netcdf CRCM5_NEMO_CanESM2_RCP85_1989-2010_lkeff_snfl_1999-1999_m02-02_daily {
    dimensions:
        x = 188 ;
        y = 89 ;
        t = 28 ;
    variables:
        float lon(x, y) ;
            lon:_FillValue = NaNf ;
        float lat(x, y) ;
            lat:_FillValue = NaNf ;
        int64 t(t) ;
            t:units = "days since 1999-02-01 00:00:00" ;
            t:calendar = "proleptic_gregorian" ;
        float hles_snow(t, x, y) ;
            hles_snow:_FillValue = NaNf ;
            hles_snow:coordinates = "lat lon" ;
        float lake_ice_fraction(t, x, y) ;
            lake_ice_fraction:_FillValue = NaNf ;
            lake_ice_fraction:coordinates = "lat lon" ;
        float u_we(t, x, y) ;
            u_we:_FillValue = NaNf ;
            u_we:coordinates = "lat lon" ;
        float v_sn(t, x, y) ;
            v_sn:_FillValue = NaNf ;
            v_sn:coordinates = "lat lon" ;
}
"""


def read_var_in_memory(dir, common_suffix="daily.nc", varname="lake_ice_fraction"):
    """
    :param dir:
    :param common_suffix:
    """
    with xarray.open_mfdataset(f"{dir}/*{common_suffix}") as ds:
        d_arr = ds[varname].load()
        return d_arr


def get_quantile_for_months(data_array: xarray.DataArray, months_of_interest=None, q = 0.9):

    if months_of_interest is None:
        months_of_interest = list(range(1, 13))





def main():

    from lake_effect_snow.hles_cc import common_params

    data_root = common_params.data_root

    cur_climate_label2path = OrderedDict(
        ("CRCM5_NEMOc", data_root / "lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_1989-2010_1989-2010")
    )

    fut_climate_label2path = OrderedDict(
        ("CRCM5_NEMOf", data_root / "lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_2079-2100_2079-2100")
    )


    months_of_interest = [1, 12]

    # ---



if __name__ == '__main__':
    main()