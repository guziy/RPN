
__author__="huziy"
__date__ ="$Jul 14, 2011 8:35:52 AM$"

import matplotlib.pyplot as plt
import pylab
from math import *
from mpl_toolkits.basemap import Basemap, maskoceans
from matplotlib.ticker import LinearLocator
from rpn import RPN
import numpy as np


inches_per_pt = 1.0 / 72.27               # Convert pt to inch
golden_mean = (sqrt(5) - 1.0) / 2.0       # Aesthetic ratio
fig_width = 1000 * inches_per_pt          # width in inches
fig_height = fig_width * golden_mean      # height in inches
fig_size = [fig_width, fig_height]

params = {
        'axes.labelsize': 14,
        'font.size':18,
        'text.fontsize': 18,
        'legend.fontsize': 14,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'figure.figsize': fig_size
        }

pylab.rcParams.update(params)



def plot_field_2d(lons_2d, lats_2d, field_2d, start_lon = -180, end_lon = 0, color_map = None,
                   minmax = (None, None)  ):


    plt.figure()
    m = Basemap(llcrnrlon = start_lon,llcrnrlat = np.min(lats_2d),
                urcrnrlon = end_lon,urcrnrlat = np.max(lats_2d), resolution = 'l')

    m.drawmeridians(range(start_lon,end_lon,10))
    m.drawparallels(range(-90,90,10))


 #   y, x = meshgrid(lats_2d, lons_2d)
#    lons_2d[lons_2d < start_lon] = lons_2d[lons_2d < start_lon] + 360
    x, y = m(lons_2d, lats_2d)


    x -= 360 ###########CONVERTING LONGITUDE TO -180:180
    field_2d = maskoceans(x, y, field_2d)


    m.pcolormesh(x, y, field_2d, cmap = color_map, vmin = minmax[0], vmax = minmax[1])
    m.drawcoastlines()
    #plt.imshow(np.transpose(data[:,:]), origin = 'lower') #for plotting in order to see i,j we supply j,i
    numticks = color_map.N + 1 if color_map != None else 10
    plt.colorbar(ticks = LinearLocator(numticks = numticks), format = '%.01f',
                 orientation = 'vertical', shrink = 0.6)


def test():

    path = 'data/pm1957090100_00589248p'
    #path = 'data/crcm_sim_with_lakes/data_selected/Coupled11_36cpu_Test_C_198505/pm1957090100_00727920p'
    rpn = RPN(path)
    data = rpn.get_first_record_for_name_and_level(varname = 'FV', level = 7 )

    lons, lats = rpn.get_longitudes_and_latitudes()

    print(lons.shape, np.min(lons), np.max(lons))
    print(lats.shape, np.min(lats), np.max(lats))
#    print data.shape

#    plot_field_2d(lons, lats, data[:,:,0])

    print(data.shape)
    plot_field_2d(lons, lats, data)


    plt.savefig('plot.png')
    #plt.figure()
    #plt.imshow(np.transpose(lons[:,:]), origin = 'lower')
    #plt.imshow(np.transpose(data[:,:]), origin = 'lower') #for plotting in order to see i,j we supply j,i

    plt.show()


import application_properties
if __name__ == "__main__":
    application_properties.set_current_directory()
    test()
    print("Hello World")
