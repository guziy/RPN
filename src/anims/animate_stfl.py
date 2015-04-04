from datetime import datetime
from matplotlib.collections import QuadMesh
from mpl_toolkits.basemap import Basemap

__author__ = 'huziy'

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from crcm5.model_data import Crcm5ModelDataManager

class Animator():
    def __init__(self, data, basemap, x, y, t0):
        self.data = data
        self.basemap = basemap
        self.x = x
        self.y = y
        self.qm = basemap.pcolormesh(x, y, data[t0])

        basemap.drawcoastlines()
        basemap.drawrivers()

        pass

    def animate(self, t):
        plt.title(t)
        self.basemap.pcolormesh(self.x, self.y, self.data[t])
        pass

def main():
    var_name = "STFL"
    data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_without_lakes_v3_old_snc"
    data_manager = Crcm5ModelDataManager(samples_folder_path=data_path,
        var_name=var_name, all_files_in_samples_folder=True, file_name_prefix="pm")

    lons2D, lats2D = data_manager.lons2D, data_manager.lats2D

    basemap = Basemap(projection="omerc", no_rot=True, lon_1=-68, lat_1=52, lon_2=16.65, lat_2=0,
            llcrnrlon=lons2D[0,0], llcrnrlat=lats2D[0,0],
            urcrnrlon=lons2D[-1,-1], urcrnrlat=lats2D[-1, -1],
            resolution="l"
    )


    fig = plt.figure()
    data = data_manager.get_date_to_field_dict()
    print(list(data.keys()))
    print(len(list(data.keys())))
    times = list( sorted( [datetime.strptime(s, "%Y-%m-%d %H:%M") for s in list(data.keys())]) )

    times = [t.strftime("%Y-%m-%d %H:%M") for t in times]


    ims = []
    x, y = basemap(lons2D, lats2D)

    aniObj = Animator(data, basemap, x, y, times[0])

    fa = animation.FuncAnimation(fig, aniObj.animate, times, interval=50)

    plt.show()
    #fa.save("animateflow.mpg")
    data.close()
    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print("Hello world")
  