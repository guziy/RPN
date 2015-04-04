
import os.path

__author__="huziy"
__date__ ="$Apr 13, 2011 1:47:04 PM$"

from rpn import RPN
from test_rpn import plot_field_2d
import matplotlib.pyplot as plt
import numpy as np

import os
import application_properties
application_properties.set_current_directory()

from datetime import datetime

import level_kinds

import matplotlib.cm as cm

from my_colormaps import *

def get_lakefraction(path = 'data/geophys_lam_na_0.5deg_170x158_class', margin = 20):
    r = RPN(path)
    data = r.get_first_record_for_name_and_level('VF', level = 3)
    r.close()
    return data[margin:-margin, margin:-margin, 0]
    pass

#Lv = 2.27e6            #J/kg
Lv = 2501000.0
water_density = 1000.0 #kg/m^3

#get longitudes and latitudes
def get_lon_lat(path = 'data/pm1957090100_00589248p'):
    print('reading lons and lats from the file %s' % path)
    r = RPN(path)
    lons, lats = r.get_longitudes_and_latitudes()
    r.close()
    return lons, lats


def test_FV_AV(path = 'data/pm1957090100_00589248p'):
    r = RPN(path)
    av = r.get_first_record_for_name('AV')
    fv = r.get_first_record_for_name_and_level(varname = 'FV', level = 5)
    
    nx, ny = av[:,:,0].shape
    ratio = np.zeros((nx, ny))

    for i in range(nx):
        for j in range(ny):
            if fv[i, j] != 0:
                ratio[i, j] = av[i, j] / fv[i, j]

    

    print(np.max(av))
    print(np.max(fv))
    print(np.max(av) / np.max(fv))

    plt.figure()
    plt.imshow(av[:,:,0])
    plt.colorbar()

    plt.figure()
    plt.imshow(fv[:,:,0])
    plt.colorbar()

    plt.show()


#selects data from the time interval [start_date, end_date] inclusively.
def calculate_time_integral(data_folder = 'data/crcm_sim_with_lakes/data_selected',
                            start_date = None, end_date = None, var_name = 'FV', level = 7,
                            level_kind = level_kinds.ARBITRARY,
                            name_pattern = 'Coupled11_36cpu_Test_C_%Y%m',
                            dt = 3 * 60 * 60,
                            file_size_limit_bytes = None
                            ):


    date = start_date

    time_integral = None
    while date <= end_date:
        current_folder = os.path.join(data_folder, date.strftime(name_pattern))
        
        for file in os.listdir(current_folder):
            if not file.startswith('pm'): #skip non physics files
                continue

            file_path = os.path.join(current_folder, file)

            if file_size_limit_bytes != None:
                if os.path.getsize(file_path) < file_size_limit_bytes:
                    continue

            r = RPN(file_path)
            print('current folder ', current_folder)
            print('reading file {0}'.format(file))
            data = r.get_first_record_for_name_and_level(varname = var_name, level = level, level_kind = level_kind)
            data = data[:,:,0]
            if date == start_date:
                time_integral = data 
            else:
                time_integral += data 
            r.close()
        #add month
        if date.month == 12:
            date = datetime(date.year + 1, 1, date.day, date.hour, date.minute)
        else:
            date = datetime(date.year, date.month + 1, date.day, date.hour, date.minute)
    return time_integral * dt
    pass


def main():
    
    lake_fraction = get_lakefraction()
    print(lake_fraction.shape)
    
 
    start_date = datetime(1985,1,1,0,0,0)
    end_date = datetime(1990,12,31,0,0,0)

#hostetler
    data_folder = 'data/crcm_sim_with_lakes/data_selected'
    name_pattern = 'Coupled11_36cpu_Test_C_%Y%m'
#   without lakes
#    data_folder = 'data/crcm_sim_without_lakes'
#    name_pattern = 'NEW_nolakes_%Y%m'

    #
#    data_folder = '/home/martynov/w2/gemclim/Out/Samples'
#    name_pattern = 'HS_NA_BIG_0.44deg_CORDEX_snow_%Y%m'
    
    
    folder_path = os.path.join(data_folder, start_date.strftime(name_pattern))
    for file in os.listdir(folder_path):
        lons, lats = get_lon_lat(path = os.path.join(folder_path, file))
        break

    print(np.min(lons), np.max(lons))
    print(np.min(lats), np.max(lats))



    latentHeat = calculate_time_integral(data_folder = data_folder, 
                                         start_date = start_date,
                                         end_date = end_date,
                                         var_name = 'FV', level = 7,
                                         name_pattern = name_pattern, dt = 3 * 60 * 60,
                                         file_size_limit_bytes = 100000
                                         )
                                         #TODO: determine dt from data in rpn file
    evaporation = latentHeat / (Lv * water_density)

    precipitation = calculate_time_integral(data_folder = data_folder, 
                                            start_date = start_date,
                                            end_date = end_date,
                                            var_name = 'PR', level = -1,
                                            name_pattern = name_pattern, dt = 3 * 60 * 60,
                                            file_size_limit_bytes = 100000
                                            )


    x = (precipitation - evaporation) 

    print('min, max = ', np.min(x), np.max(x))

    d_max = np.max(np.abs(x))
    d_min = - d_max
    
    x = np.ma.masked_where(lake_fraction < 0.01, x)

    
    plot_field_2d(lons, lats, x, color_map = get_red_blue_colormap(10) ,#cm.get_cmap('jet', 32),
                                 minmax = (d_min, d_max),
                                 start_lon = -175,
                                 end_lon = -50)
#    plt.title(name_pattern)
#    plt.title('aggregated $P-E$ in meters, integrated \n during 1985-1990 year \n (only cells with lake fraction > 0.01 are shown)')
    plt.savefig(name_pattern + '.png', bbox_inches = 'tight')
    pass


if __name__ == "__main__":
    main()
    #test_FV_AV()
    print("Hello World")
