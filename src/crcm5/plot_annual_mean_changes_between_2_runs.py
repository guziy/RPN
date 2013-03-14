from netCDF4 import Dataset
from matplotlib.cm import get_cmap
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.basemap import maskoceans
import os
from crcm5.model_data import Crcm5ModelDataManager

__author__ = 'huziy'

import numpy as np
from scipy import stats

start_year = 1979
end_year = 1986

field_names = ["TT", "PR", "AU", "AV", "STFA"]

#field_names = ["TRAF", "TDRA"]

field_name_to_long_name = {
    "TT": "Temperature",
    "PR": "Precipitation",
    "AU": "Sensible heat flux",
    "AV": "Latent heat flux",
    "STFA" :"Streamflow",
    "TRAF" :"Surface runoff",
    "TDRA" : "Subsurface runoff"
}


field_name_to_units = {
    "TT": "${\\rm ^\circ C}$",
    "PR": "mm/day",
    "AU": "${\\rm W/m^2}$",
    "AV": "${\\rm W/m^2}$",
    "STFA" :"${\\rm m^3/s}$",
    "TRAF" : "mm/day",
    "TDRA" : "mm/day"

}

field_name_to_clevels = {
    "TT": [-2,-1, -0.5, -0.25, 0,1,2,4,6],
    "PR": [-0.4,-0.3,-0.2,-0.1, 0,0.25, 0.5, 0.75,  1],
    "AU": [-55, -35, -25, -10, -5, 0, 2, 4,6, 8,10],
    "AV": [-0.002, -0.0015, -0.001, 0, 0.003, 0.005, 0.007],
    "STFA" : [-3500, -1000, -100, -50 , -10, -5, 0,1,2,4,6,8, 10],
    "TRAF" : np.arange(0, 0.35, 0.02),
    "TDRA": np.arange(-0.3, 0.05, 0.02)

}



sim_name1 = "crcm5-r"
sim_name2 = "crcm5-hcd-r"
nc_db_folder = "/home/huziy/skynet3_rech1/crcm_data_ncdb"

#needed for basemap
rpn_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_{0}_spinup".format(sim_name2)


def _get_values(sim_name, varname):
    print varname, sim_name
    nc_data_folder = os.path.join(nc_db_folder, sim_name)
    ds = Dataset(os.path.join(nc_data_folder, "{0}.nc".format(varname)))
    years = ds.variables["year"][:]
    sel = (start_year <= years) & (years <= end_year)
    tt = ds.variables[varname][sel,:,:,:]
    ds.close()
    return tt

    pass


def main():
    import matplotlib.pyplot as plt
    fig = plt.figure()


    var_name_to_ij = {
        "TT": (0,0),
        "PR": (0,1),
        "AU": (1,0),
        "AV": (1,1),
        "STFA": (0,2),


        "TRAF": (0, 0), "TDRA" : (0, 1)
    }

    dmManager = Crcm5ModelDataManager(samples_folder_path=rpn_folder, file_name_prefix="dm", all_files_in_samples_folder=True)
    basemap = dmManager.get_omerc_basemap()
    lons, lats = dmManager.lons2D, dmManager.lats2D
    x, y = basemap(lons, lats)

    lkfr = dmManager.lake_fraction



    gs = GridSpec(2,3, width_ratios=[1,1,1], height_ratios=[1, 1])

    fig.suptitle("{0} minus {1}".format(sim_name2, sim_name1))



    for var_name in field_names:
        levels = field_name_to_clevels[var_name]
        if var_name == "TT":
            cmap = get_cmap("RdBu_r",len(levels) - 1)
        else:
            cmap = get_cmap("RdBu", len(levels) - 1)

        bn = BoundaryNorm(levels, cmap.N)

        coef = 1
        if var_name == "PR":
            coef = 24 * 60 * 60 * 1000
        elif var_name in ["TDRA", "TRAF"]:
            coef = 24 * 60 * 60
        cmap.set_bad("0.6")
        print var_name
        v1 = _get_values(sim_name1, var_name)
        v2 = _get_values(sim_name2, var_name)

        #calculate annual means, for each year
        v1 = v1.mean(axis=1)
        v1m = v1.mean(axis = 0)

        v2 = v2.mean(axis = 1)
        v2m = v2.mean(axis = 0)

        i,j = var_name_to_ij[var_name]

        ax = fig.add_subplot(gs[i,j])
        dv = (v2m -  v1m) * coef

        t, p = stats.ttest_ind(v1, v2)

        if var_name == "STFA":
            dv = np.ma.masked_where((p > 0.05) | (lkfr >= 0.6), dv)
            print "lf_max = {0}".format(lkfr[lkfr < 0.6].max())
            print "lf_susp = {0}".format(lkfr[np.where(dv == dv.min())])

            dv = maskoceans(lons, lats, dv)

        elif var_name == "TRAF":
            dv = np.ma.masked_where(dv < 0, dv)
            dv = maskoceans(lons, lats, dv)
        elif var_name == "TDRA":
            dv = np.ma.masked_where(dv > 0, dv)
            dv = maskoceans(lons, lats, dv)
        else:
            dv = np.ma.masked_where(p > 0.05, dv)
            pass


        print "{0}: min = {1}; max = {2}".format(var_name, dv.min(), dv.max())
        the_img = basemap.pcolormesh(x, y, dv, cmap=cmap, norm = bn)


        #add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "10%", pad="3%")
        cb = plt.colorbar(the_img,cax = cax)

        #coast lines
        basemap.drawcoastlines(ax = ax, linewidth=0.1)

        ax.set_title( "$\Delta$ " + field_name_to_long_name[var_name] + " ({0})".format(field_name_to_units[var_name]))

    fig.savefig("annual_mean_diffs_{0}_minus_{1}.png".format(sim_name2, sim_name1))
    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    from util import plot_utils
    plot_utils.apply_plot_params(width_pt=None, width_cm=40, height_cm=25, font_size=14)
    main()
    print "Hello world"
  