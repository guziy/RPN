from datetime import datetime
from matplotlib.cm import get_cmap
from matplotlib.colors import BoundaryNorm
from crcm5.model_data import Crcm5ModelDataManager

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt

def main():


    path1 = "/home/huziy/skynet3_exec1/from_guillimin/new_outputs/quebec_86x86_0.5deg_wo_lakes_and_wo_lakeroff"
    base_data_manager1 = Crcm5ModelDataManager(samples_folder_path=path1,
        all_files_in_samples_folder=True)


    basemap = base_data_manager1.get_omerc_basemap()
    x, y = basemap(base_data_manager1.lons2D, base_data_manager1.lats2D)

    start_date = datetime(1986, 1, 1)
    end_date = datetime(1990, 12, 31)
    months = range(4,6)

    traf1 = base_data_manager1.get_monthly_climatology(start_date=start_date, end_date=end_date, months=months,
                                                        varname="TRAF", level = 1)
    traf1 = np.mean(traf1, axis = 0)

    tdra1 = base_data_manager1.get_monthly_climatology(start_date=start_date, end_date=end_date, months=months,
                                                        varname="TDRA", level = 1)
    tdra1 = np.mean(tdra1, axis = 0)



    ratio1 = tdra1.copy()
    denom_pos = (traf1 + tdra1) > 0
    ratio1[denom_pos] /= (traf1 + tdra1)[denom_pos]
    ratio1 = np.ma.masked_where(~denom_pos, ratio1)

    print np.min(ratio1), np.max(ratio1)
    print np.min(tdra1), np.max(tdra1)
    print np.min(traf1), np.max(traf1)





    path2 = "/home/huziy/skynet3_exec1/from_guillimin/new_outputs/quebec_lowres_005"
    base_data_manager2 = Crcm5ModelDataManager(samples_folder_path=path2,
            all_files_in_samples_folder=True)
    traf2 = base_data_manager2.get_monthly_climatology(start_date=start_date, end_date=end_date, months=months,
                                                        varname="TRAF", level=1)
    traf2 = np.mean(traf2, axis = 0)

    tdra2 = base_data_manager2.get_monthly_climatology(start_date=start_date, end_date=end_date, months=months,
                                                        varname="TDRA", level =1)
    tdra2 = np.mean(tdra2, axis = 0)

    ratio2 = tdra2.copy()
    denom_pos = (traf2 + tdra2) > 0
    ratio2[denom_pos] /= (traf2 + tdra2)[denom_pos]
    ratio2 = np.ma.masked_where(~denom_pos, ratio2)

    levels = [0,0.01, 0.02, 0.03, 0.04, 0.05,0.06,0.1,0.15,0.18,0.20, 0.4,0.6,0.8,1.0]
    cMap = get_cmap("jet", len(levels) - 1 )
    bn = BoundaryNorm(levels, cMap.N)


    plt.figure()
    min_change = 0.01
    change = (ratio2 - ratio1)/ratio1
    change =  np.ma.masked_where(change <=min_change, change)
    basemap.pcolormesh(x, y, change, vmin = levels[0], vmax = levels[-1], norm = bn, cmap=cMap)
    plt.title("(intfl - no_intfl)/no_intfl")
    plt.colorbar()
    basemap.drawcoastlines()


    plt.figure()
    basemap.pcolormesh(x, y, ratio2)
    plt.title("intfl")
    plt.colorbar()
    basemap.drawcoastlines()

    plt.figure()
    basemap.pcolormesh(x, y, ratio1)
    plt.title("no_intfl")
    plt.colorbar()
    basemap.drawcoastlines()


    plt.show()




    #TODO: implement
    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print "Hello world"
  