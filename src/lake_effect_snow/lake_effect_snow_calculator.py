
from lake_effect_snow import base_utils
from lake_effect_snow import common_params




def calculate_lake_effect_snowrate(u_10m=None, v_10m=None, tair_2m_degc=None, precip_m_per_s=None,
                                                         snowfall_m_per_s=None, lake_fraction=None, lake_ice_fraction=None):

    # If snowfall is not passed => calculate it from the 2m air temperature and precip
    """
    :param u_10m:
    :param v_10m:
    :param tair_2m_degc:
    :param precip_m_per_s: Total precipitation in M/s
    :param snowfall_m_per_s:
    """
    if snowfall_m_per_s is None:
        assert tair_2m_degc is not None and precip_m_per_s is not None
        snowfall_m_per_s = base_utils.get_snow_fall_m_per_s(precip_m_per_s=precip_m_per_s, tair_deg_c=tair_2m_degc)

    if None in [u_10m, v_10m]:
        calculate_lake_effect_snowrate_based_on_snowfall(snowfall_m_per_s=snowfall_m_per_s)
    else:
        calculate_lake_effect_snowrate(u_10m=u_10m, v_10m=v_10m, snowfall_m_per_s=snowfall_m_per_s,
                                       lake_fraction=lake_fraction, lake_ice_fraction=lake_ice_fraction)



def calculate_lake_effect_snowrate_based_on_snowfall(snowfall_m_per_s,
                                                     lower_lakeeffect_limit=common_params.lower_limit_of_daily_snowfall):
    """
    Set to 0 all snowrates below the limit
    :param lower_lakeeffect_limit:
    :param snowfall_m_per_s:
    """

    res = snowfall_m_per_s.copy()
    res[snowfall_m_per_s <= lower_lakeeffect_limit] = 0
    return res



def main():
    # for tests
    pass

if __name__ == '__main__':
    main()