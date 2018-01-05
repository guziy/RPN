
import numpy as np

from crcm5.nemo_vs_hostetler import commons

FRESH_SNOW_MIN_DENSITY_KG_PER_M3 = 50.0
WATER_DENSITY_KG_PER_M3 = 1000.0



class VerticalLevel(object):
    def __init__(self, value, level_type=-1):
        self.value = value
        self.level_type = level_type

    def get_value_and_kind(self):
        """

        :return:
        """
        return self.value, self.level_type


def get_snow_density_kg_per_m3(tair_deg_c):

    """
    Get density of the falling snow based on the 2-meter air temperature
    :param tair_deg_c:
    :return:
    """
    rhos = np.zeros_like(tair_deg_c)

    valid = ~np.isnan(tair_deg_c)

    rhos[valid & (tair_deg_c < -15)] = FRESH_SNOW_MIN_DENSITY_KG_PER_M3

    where_not_very_cold = valid & (tair_deg_c >= -15)
    if np.any(where_not_very_cold):
        rhos[where_not_very_cold] = FRESH_SNOW_MIN_DENSITY_KG_PER_M3 + 1.7 * (tair_deg_c[where_not_very_cold] + 15) ** 1.5

    return rhos


def get_snow_fall_m_per_s(precip_m_per_s, tair_deg_c):
    """

    :param precip_m_per_s: total precipitation field in M/s
    :param tair_deg_c: 2-m air temperature in degC
    Formula is based on Notaro et al 2015
    returns snowfall in M/s - same as precip units

    Note: it is actual snow fall, not water equivalent
    """

    result = np.zeros_like(precip_m_per_s)


    where_cold = tair_deg_c < 0

    # if there is no points below the freezing point
    if not np.any(where_cold):
        return result


    rhos = get_snow_density_kg_per_m3(tair_deg_c)

    result[~where_cold] = 0.0

    result[where_cold] = precip_m_per_s[where_cold] * WATER_DENSITY_KG_PER_M3 / rhos[where_cold]

    return result