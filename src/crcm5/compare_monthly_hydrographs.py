from netCDF4 import Dataset
from matplotlib import gridspec
import os
from crcm5.model_data import Crcm5ModelDataManager

__author__ = 'huziy'
from scipy.interpolate import spline


import numpy as np


sim_names = ["crcm5-r",
             "crcm5-hcd-r",
             "crcm5-hcd-rl",
             "crcm5-hcd-rl-intfl"]


import numpy as np
from scipy import stats

start_year = 1980
end_year = 1986

field_names = ["TT", "PR", "AU", "AV", "STFA"]


nc_db_folder = "/home/huziy/skynet3_rech1/crcm_data_ncdb"

#needed for basemap
rpn_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_{0}_spinup".format(sim_names[1])



import matplotlib.pyplot as plt

def _get_values(sim_name, varname):
    print sim_name, varname
    nc_data_folder = os.path.join(nc_db_folder, sim_name)
    ds = Dataset(os.path.join(nc_data_folder, "{0}.nc".format(varname)))
    years = ds.variables["year"][:]
    sel = (start_year <= years) & (years <= end_year)
    tt = ds.variables[varname][sel,:,:,:]
    ds.close()
    return tt



def main():
    #TODO: implement



    var_name_to_ij = {
        "TT": (0,0),
        "PR": (0,1),
        "AU": (1,0),
        "AV": (1,1),
        "STFA": (0,2)
    }

    dmManager = Crcm5ModelDataManager(samples_folder_path=rpn_folder, file_name_prefix="dm", all_files_in_samples_folder=True)
    basemap = dmManager.get_omerc_basemap()
    lons, lats = dmManager.lons2D, dmManager.lats2D
    x, y = basemap(lons, lats)

    lkfr = dmManager.lake_fraction
    acc_area = dmManager.accumulation_area_km2


    i_arr, j_arr = np.where((lkfr > 0.4) & (lkfr < 0.6) & (acc_area > 10000))
    name_to_data = {}
    for sim in sim_names:
        name_to_data[sim] = _get_values(sim, "STFA").mean(axis = 0)

    fig = plt.figure()

    k = 0
    nrows = 3
    ncols = 3
    gs = gridspec.GridSpec(nrows, ncols)
    name_to_handle = {}
    for row in range(nrows):
        for col in range(ncols):
            i, j = i_arr[k],  j_arr[k]
            ax = fig.add_subplot(gs[row, col])
            for sim in sim_names:

                xnew = np.linspace(1, 13, 100)

                power_smooth = spline(np.arange(1, 13), name_to_data[sim][:,i,j] ,xnew, order = 3)


                h = ax.plot(np.arange(1, 13), name_to_data[sim][:,i,j], label = sim,  lw = 2)
                ax.xaxis.set_ticks(np.arange(1, 13))
                ax.set_title("{0}, {1}".format(lons[i,j], lats[i,j]))
                if not k:
                    name_to_handle[sim] = h

                if row == 1 and col == 0:
                    ax.set_ylabel("Streamflow ($m^3/s$)")
                if row == 2 and col == 1:
                    ax.set_xlabel("Month")

            k += 4

    names = sorted( name_to_handle.keys(), key=lambda x: len(x) )
    fig.legend((name_to_handle[name] for name in names), names)
    plt.show()


    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    from util import plot_utils
    plot_utils.apply_plot_params(width_pt=None, width_cm=30, height_cm=30, font_size=14)

    main()
    print "Hello world"
  