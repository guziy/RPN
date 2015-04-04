__author__="huziy"
__date__ ="$May 11, 2011 8:35:15 PM$"

name_format = '%d%02d'
import os

from rpn import RPN, plot_field_2d

import matplotlib as mpl
import matplotlib.pyplot as plt

import application_properties
application_properties.set_current_directory()

#get longitudes and latitudes
def get_lon_lat(path = '/home/huziy/skynet3_rech1/gemclim/quebec/Samples/quebec_220x220_199912/pm1999050100_00068760p'):
    r = RPN(path)
    lons, lats = r.get_longitudes_and_latitudes()
    r.close()
    return lons, lats

#returns the sum and the number of the elements added
def calculate_monthly_sum(data_folder = '/home/huziy/skynet3_rech1/gemclim/quebec/Samples',
                          month = 6, year = 1999, prefix = 'quebec_220x220_', var_name = 'GWDI', level = -1):

    suffix = name_format % (year, month)
    folder_for_month = os.path.join(data_folder, prefix + suffix)
    count = 0.0
    result_data = None

    if not os.path.isdir(folder_for_month):
        return 0, 0

    for file in os.listdir(folder_for_month):
        if not file.startswith('pm'):
            continue

        the_path = os.path.join(folder_for_month, file)
        print(the_path)
        r = RPN(the_path)
        the_data = r.get_first_record_for_name_and_level(var_name, level = level)
        if result_data == None:
            result_data = the_data
        else:
            result_data += the_data
        count += 1
        r.close()
        
    return result_data, count
    pass

#months start from 1 to 12
def calculate_mean_for_month_range(month_range = range(6,8), year_range = range(1999,2000), figure_file = 'figure.png',
                                   var_name = 'GWDI', level = -1 ):


    the_sum = 0.0
    the_quantity = 0.0
    for month in month_range:
        for year in year_range:
            print(month, year)
            sum1, quantity1 = calculate_monthly_sum(month = month, year = year, var_name = var_name, level = level)
            the_sum += sum1
            the_quantity += quantity1



    lon, lat = get_lon_lat()
    print('the_quantity = ',  the_quantity)
    plot_field_2d(lon, lat, the_sum / the_quantity, color_map = mpl.cm.get_cmap('jet_r', 10), start_lon = -80, end_lon = -50)
    plt.title('${\\rm m^3/s}$')
    plt.savefig(figure_file, bbox_inches = 'tight')
    pass

if __name__ == "__main__":
    calculate_mean_for_month_range(month_range = list(range(6,9)), year_range = list(range(1999,2000)), figure_file = 'summer_TDR.png', var_name = 'TDR', level = 5)
    calculate_mean_for_month_range(month_range = list(range(9,12)), year_range = list(range(1999,2000)), figure_file = 'autumn_TDR.png', var_name = 'TDR', level = 5)
    calculate_mean_for_month_range(month_range = [12,1,2], year_range = list(range(1999,2000)), figure_file = 'winter_TDR.png', var_name = 'TDR', level = 5)
    print("Hello World")
