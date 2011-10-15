from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

__author__ = 'huziy'

from rpn import RPN
import os
import application_properties
import numpy as np


def calculate_mean_field(data_path = "", field_name = "STFL", file_prefix = None):
    """
    Calculates annual mean field from rpn files
    data_path = path to the Samples folder
    """
    result = None
    field_count = 0.0
    lons, lats = None, None
    for monthFolder in os.listdir(data_path):

        monthPath = os.path.join(data_path, monthFolder)

        for fName in os.listdir(monthPath):

            if file_prefix is not None:
                if not fName.startswith(file_prefix):
                    continue

            rObj = RPN(os.path.join(monthPath, fName))
            field = rObj.get_first_record_for_name(field_name)

            vDate = rObj.get_current_validity_date()
            originDate = rObj.get_dateo_of_last_read_record()


            print "-" * 10
            print "validity date, origin date", vDate, originDate
            print rObj.get_datetime_for_the_last_read_record()
            print "-" * 10


            if result is None:
                result = field
                lons, lats = rObj.get_longitudes_and_latitudes()
            else:
                result = (field + result * field_count) / ( field_count + 1.0 )
            rObj.close()
            field_count += 1.0

    return lons, lats, result

def calculate_seasonal_mean(data_path = "", field_name = "STFL", file_prefix = None, months = None):
    """
    calculates seasonal means,
    months - list of months when the averaging is performed 1 = Jan, ..., 12 = Dec
    TODO: implement
    """

    result = None
    field_count = 0.0
    lons, lats = None, None
    for monthFolder in os.listdir(data_path):

        monthPath = os.path.join(data_path, monthFolder)

        for fName in os.listdir(monthPath):

            if file_prefix is not None:
                if not fName.startswith(file_prefix):
                    continue

            rObj = RPN(os.path.join(monthPath, fName))
            field = rObj.get_first_record_for_name(field_name)

            vDate = rObj.get_current_validity_date()
            originDate = rObj.get_dateo_of_last_read_record()


            print "-" * 10
            print "validity date, origin date", vDate, originDate
            print "-" * 10

            if result is None:
                result = field
                lons, lats = rObj.get_longitudes_and_latitudes()
            else:
                result = (field + result * field_count) / ( field_count + 1.0 )
            rObj.close()
            field_count += 1.0




    pass

def calculate_and_plot():
    """

    """
    basemap = Basemap()
    lons, lats, data = calculate_mean_field(
                        data_path = "data/gemclim/quebec/Samples",
                        file_prefix = "pm"
    )

    lons, lats = basemap(lons, lats)
    lons[lons > 180] -= 360

    print np.min(lons), np.max(lons)
    basemap.pcolormesh(lons, lats, np.ma.masked_where(data < 0, data, copy = False))
    basemap.drawcoastlines()
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    application_properties.set_current_directory()
    calculate_and_plot()