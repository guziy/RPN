from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.basemap import Basemap

__author__ = 'huziy'

import numpy as np

from model_data import Crcm5ModelDataManager
import matplotlib.pyplot as plt
#Produces a plot (v_mean1 - v_mean_base)/v_mean_base for the given 2d variable
#with the option to mask the points where the changes are not significant
#initially designed to compare streamflow, with the base streamflow corresponding to
#the case without lakes and without lake runoff

from matplotlib.cm import get_cmap


def main():
    var_name = "PR"
    perform_sign_test = False
    case_name = "0.1 deg, wo lakes, with lakeroff, no gw"

    base_data_path =  "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_wo_lakes_and_wo_lakeroff"

    case_data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_260x260_wo_lakes_and_with_lakeroff_nogw"

    base_data_manager = Crcm5ModelDataManager(samples_folder_path=base_data_path, all_files_in_samples_folder=True)
    case_data_manager = Crcm5ModelDataManager(samples_folder_path=case_data_path, all_files_in_samples_folder=True)

    #baseMeans = base_data_manager.get_annual_mean_fields(varname="PR",start_year=1986, end_year=1990)
    caseMeans = case_data_manager.get_annual_mean_fields(varname="PR", start_year=1986, end_year=1990)


    lons2d = case_data_manager.lons2D
    lats2d = case_data_manager.lats2D



    b = base_data_manager.get_omerc_basemap()
    x, y = b(lons2d, lats2d)

    levels = [0,100, 200, 300, 400,500,600, 700, 800, 900, 1000, 1200, 1500, 1800,2000, 2500]
    cMap = get_cmap("jet", len(levels) - 1 )
    bn = BoundaryNorm(levels, cMap.N)

    assert isinstance(b, Basemap)
    b.pcolormesh(x, y, caseMeans[1987] * 86400 * 365 * 1000, norm=bn, vmin=0, vmax=levels[-1], cmap = cMap)
    b.drawcoastlines()
    plt.colorbar(ticks = levels, extend = "max")
    plt.show()

    return



    years = baseMeans.keys()
    means = []
    case_means = []
    for y in years:
        data = baseMeans[y]
        means.append(data[data >= 0].sum())

        data = caseMeans[y]
        case_means.append(data[data >= 0].sum())



        #print y, np.min(data[data>=0]), np.max(data)

    plt.plot(years, case_means, label=case_name)
    plt.plot(years, means, label="base")
    ax = plt.gca()
    assert isinstance(ax, Axes)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.legend()
    plt.show()






    #TODO: implement
    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print "Hello world"
  