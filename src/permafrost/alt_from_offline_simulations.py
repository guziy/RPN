import os
import pickle
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from active_layer_thickness import CRCMDataManager
import my_colormaps
import draw_regions
from sounding_plotter import SoundingPlotter
from sounding_and_cross_section_plotter import SoundingAndCrossSectionPlotter
from rpn.rpn import RPN

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt


def plot_alt(alt, tmin, tmax, levelheights, annual_means):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    basemap, lons2d, lats2d = draw_regions.get_basemap_and_coords(resolution="c",
         llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-20, urcrnrlat=74
     )
    assert isinstance(basemap, Basemap)

    lons2d[lons2d > 180] -= 360
    x, y = basemap(lons2d, lats2d)


    permafrost_mask = draw_regions.get_permafrost_mask(lons2d, lats2d)
    mask_cond = (permafrost_mask <= 0) | (permafrost_mask >= 3) | (alt < 0)

    alt = np.ma.masked_where(mask_cond, alt)


    cmap = my_colormaps.get_lighter_jet_cmap(ncolors=10) #cm.get_cmap("jet",10)
    bounds = [0,0.1,0.5,1,2,3,5,8,9,10,11]
    norm = BoundaryNorm(boundaries=bounds,ncolors=cmap.N, clip=True)

    qm = basemap.pcolormesh(x, y, alt, cmap=cmap, norm = norm, ax = ax)
    #qm = basemap.contourf(x, y, alt, levels = bounds, ax = ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = fig.colorbar(qm,  cax = cax, extend = "max", ticks = bounds)

    basemap.drawcoastlines(ax = ax, linewidth=0.5)

    basemap.readshapefile("data/pf_4/permafrost8_wgs84/permaice", name="zone",
                ax=ax, linewidth=1.5)


    #crossPlotter = SoundingPlotter(ax, basemap, tmin, tmax, lons2d, lats2d, levelheights=levelheights)

    sectPlotter = SoundingAndCrossSectionPlotter(ax, basemap, tmin, tmax, lons2d, lats2d,
        levelheights=levelheights, temporal_data=annual_means)

    plt.show()


#data holder of annual means and last 12 months
class TempArrDerivs:
    pass




def main():


    cache_file = "temp_arr.bin"
    path = "/home/huziy/skynet1_rech3/cordex/Offline_CLASS_simulations/CLASSoff_300yrs_LAM_NA_ERA40_CORDEX_clim_I0.rpn"
    if not os.path.isfile(cache_file):

        temp_arr_derivs = TempArrDerivs()
        rpnObj = RPN(path)

        temp = rpnObj.get_4d_field_fc_hour_as_time(name="I0")
        times = temp.keys()


        sorted_times = list( sorted(times) )
        print "Start time: ", sorted_times[0]
        print "End time: ", sorted_times[-1]
        print "n times = ", len(times)

        last_months = sorted_times[-12:]

        levels = list( sorted( temp[sorted_times[0]].keys() ) )

        temp0 = temp[sorted_times[0]][levels[0]]
        # array shape (t, x, y, z)
        print "trying to allocate a large array"
        temp_arr = -np.ones((len(last_months), temp0.shape[0], temp0.shape[1], len(levels)))
        print temp_arr.shape
        for ti, t in enumerate(last_months):
            for levi, lev in enumerate(levels):
                temp_arr[ti,:,:, levi] = temp[t][lev]

        temp_arr_derivs.temp_arr_last = temp_arr


        nyears = 150 #len(sorted_times) // 12
        nx, ny = temp0.shape[0], temp0.shape[1]
        nz = len(levels)
        temp_arr_derivs.annual_means = -np.ones((nyears, nx , ny, nz))
        for y in range(nyears):
            data = -np.ones((12,nx, ny,nz))
            for levi, lev in enumerate(levels):
                for ti in range(12 * y, 12 * (y + 1)):
                    data[ti % 12,:,:,levi] = temp[sorted_times[ti]][lev]

            temp_arr_derivs.annual_means[y,:,:,:] = data.mean(axis=0)



        print temp_arr.shape
        pickle.dump(temp_arr_derivs, open(cache_file, mode="w"))
    else:
        temp_arr_derivs = pickle.load(open(cache_file))

    temp_arr_last = temp_arr_derivs.temp_arr_last[:,20:-20, 20:-20,:]
    tmax = np.max(temp_arr_last, axis=0)
    tmin = np.min(temp_arr_last, axis = 0)
    print tmax.shape

    crcm = CRCMDataManager()
    #alt = crcm.get_alt(tmax + crcm.T0)

    tmin += crcm.T0
    tmax += crcm.T0

    alt = crcm.get_alt_considering_min_temp(tmax, tmin)

    print "plotting"
    plot_alt(alt, tmin, tmax, crcm.level_heights, temp_arr_derivs.annual_means[:,20:-20, 20:-20,:])




    #TODO: implement
    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print "Hello world"
  
