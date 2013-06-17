from datetime import datetime
from netCDF4 import Dataset, num2date
import os
from crcm5.model_data import Crcm5ModelDataManager
from pick_plotters.TimeSeriesPlotter import TimeSeriesPlotter

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt




def main():

    sim1 = "crcm5-r"
    sim2 = "crcm5-hcd-r"

    start_date = datetime(1985,1,1)
    end_date = datetime(1985,12,31)

    sims = [sim1, sim2]

    season_months = [4,5,6]

    ncdb_path = "/home/huziy/skynet3_rech1/crcm_data_ncdb"
    fname_pattern = "{0}_all.nc4"
    varname = "TT"

    figure = plt.figure()


    file_paths = [ os.path.join(ncdb_path, the_sim, fname_pattern.format(varname)) for the_sim in sims ]

    dsList = [ Dataset(fPath) for fPath in file_paths ]

    lons2d, lats2d, time = dsList[0].variables["lon"][:], dsList[0].variables["lat"][:], dsList[0].variables["time"][:]

    basemap = Crcm5ModelDataManager.get_rotpole_basemap_using_lons_lats(lons2d = lons2d, lats2d = lats2d)

    varsDict = {}


    time = num2date(time, dsList[0].variables["time"].units)
    sel_times = np.where([t.month in season_months for t in time])[0]

    sim_name_to_mean = {}
    for sim, ds in zip( sims, dsList ):
        varsDict["{0} ({1})".format(varname, sim)] = ds.variables[varname]

        data = ds.variables[varname][sel_times,0,:,:].mean(axis = 0)
        sim_name_to_mean[sim] = data


    x, y = basemap(lons2d, lats2d)

    #plot clickable field
    ax = plt.gca()
    basemap.contourf(x, y, sim_name_to_mean[sim2] - sim_name_to_mean[sim1])
    basemap.colorbar()
    basemap.drawcoastlines()
    ax.set_title("({0})-({1})".format(sim2, sim1).upper())

    TimeSeriesPlotter(ax, basemap, lons2d, lats2d, varsDict, time, start_date, end_date)

    plt.show()









    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print "Hello world"
  