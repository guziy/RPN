from datetime import datetime
import math

__author__ = 'huziy'

import numpy as np

#Constants
DAYS_IN_YEAR = 365.0

#for earth inclination calculation
A = np.asarray([0.006918, -0.399912, -0.006758, -0.002697])
B = np.asarray([0,  0.070257,  0.000907,  0.001480])
N = np.asarray([0, 1, 2, 3])


def get_cos_of_zenith_angle(lat, local_date_time):
    """
    Taken from CLASS-offline
    :param lon:
    :param lat:
    :param local_date_time:
    :return: cosine of the zenith angle
    """

    assert isinstance(local_date_time, datetime)

    day_of_year = local_date_time.timetuple().tm_yday

    psi = (2 * np.pi * (day_of_year - 1.0)) / DAYS_IN_YEAR
    sun_declination = np.sum(A * np.cos(N * psi) + B * np.sin(N * psi))

    #Assuming circular earth orbit
    hour_angle = np.radians(360.0 * (12 - local_date_time.hour) / 24.0)

    lat_rad = np.radians(lat)

    return np.sin(lat_rad) * np.sin(sun_declination) + \
           np.cos(lat_rad) * np.cos(sun_declination) * np.cos(hour_angle)


