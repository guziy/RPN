from datetime import datetime
from netCDF4 import Dataset
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.basemap import maskoceans
import os
from crcm5.model_data import Crcm5ModelDataManager
import my_colormaps

__author__ = 'huziy'

import numpy as np
from scipy import stats

start_year = 1979
end_year = 1988


field_names = ["TT", "PR", "AH", "AV", "STFL", "TRAF", "TDRA"]
#field_names = ["STFL"]

#field_names = ["TRAF", "TDRA"]

field_name_to_long_name = {
    "TT": "Temperature",
    "PR": "Precip.",
    "AH": "Sensible heat flux",
    "AV": "Latent heat flux",
    "STFL" :"Streamflow",
    "TRAF" :"Surface runoff",
    "TDRA" : "Subsurface runoff"
}


field_name_to_units = {
    "TT": "${\\rm ^\circ C}$",
    "PR": "mm/day",
    "AH": "${\\rm W/m^2}$",
    "AV": "${\\rm W/m^2}$",
    "STFL" :"${\\rm m^3/s}$",
    "TRAF" : "mm/day",
    "TDRA" : "mm/day"

}

field_name_to_clevels = {
    "TT": [-2,-1, -0.5, -0.25, 0,1,2,4,6],
    "PR": [-0.4,-0.3,-0.2,-0.1, 0,0.25, 0.5, 0.75,  1],
    "AH": [-55, -35, -25, -10, -5, 0, 2, 4,6, 8,10],
    "AV": [-0.002, -0.0015, -0.001, 0, 0.003, 0.005, 0.007],
    "STFL" : [-3500, -1000, -100, -50 , -10, -5, 0,1,2,4,6,8, 10],
    "TRAF" : np.arange(-0.008,0.001, 0.001),
    "TDRA": np.arange(-0.3, 0.05, 0.02)

}



sim_name1 = "crcm5-hcd-rl"
sim_name2 = "crcm5-hcd-rl-intfl"
nc_db_folder = "/home/huziy/skynet3_rech1/crcm_data_ncdb"

#needed for basemap
rpn_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_{0}_spinup2/Samples_all_in_one".format(sim_name2)


def _get_values(sim_name, varname, months = None):
    print varname, sim_name

    if months is None:
        months = np.array(range(12))
    else:
        months = np.array(months) - 1



    nc_data_folder = os.path.join(nc_db_folder, sim_name)
    path = os.path.join(nc_data_folder, "{0}.nc4".format(varname))
    print "reading {0}".format(path)
    ds = Dataset(path)
    years = ds.variables["year"][:]
    sel = np.where( (start_year <= years) & (years <= end_year) )[0]
    tt = ds.variables[varname][sel,months,:,:]
    print tt.shape
    ds.close()
    return tt




def main(months = None):
    import matplotlib.pyplot as plt
    fig = plt.figure()


    var_name_to_ij = {
        "TT": (1,0),
        "PR": (2,0),
        "AH": (1,1),
        "AV": (2,1),
        "STFL": (0,1),
        "TRAF": (1, 2),
        "TDRA" : (2, 2)
    }

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 3))

    dmManager = Crcm5ModelDataManager(samples_folder_path=rpn_folder, file_name_prefix="dm", all_files_in_samples_folder=True)
    basemap = dmManager.get_rotpole_basemap()
    lons, lats = dmManager.lons2D, dmManager.lats2D
    x, y = basemap(lons, lats)

    lkfr = dmManager.lake_fraction



    gs = GridSpec(3,3, width_ratios=[1,1,1], height_ratios=[1, 1, 1])

    #fig.suptitle("{0} minus {1}".format(sim_name2, sim_name1))
    month_dates = [datetime(2001, m, 1) for m in months]

    ax = None
    for var_name in field_names:
        levels = field_name_to_clevels[var_name]
        if var_name == "TT":
            #cmap = get_cmap("RdBu_r",len(levels) - 1)
            cmap = my_colormaps.get_cmap_from_ncl_spec_file(path="colormap_files/BlueRed.rgb", ncolors=10)
        else:
            cmap = my_colormaps.get_cmap_from_ncl_spec_file(path="colormap_files/BlueRed.rgb", ncolors=10)
            #cmap = get_cmap("RdBu", len(levels) - 1)



        coef = 1
        if var_name == "PR":
            coef = 24 * 60 * 60 * 1000
        elif var_name in ["TDRA", "TRAF"]:
            coef = 24 * 60 * 60
        #cmap.set_bad("0.6")
        print var_name
        v1 = _get_values(sim_name1, var_name, months = months)
        v2 = _get_values(sim_name2, var_name, months = months)

        #calculate annual means, for each year
        v1 = v1.mean(axis=1)
        v1m = v1.mean(axis = 0)

        v2 = v2.mean(axis = 1)
        v2m = v2.mean(axis = 0)

        i,j = var_name_to_ij[var_name]

        ax = fig.add_subplot(gs[i,j])
        dv = (v2m -  v1m) * coef

        t, p = stats.ttest_ind(v1, v2)

        if var_name in ["STFL"]:
            dv = np.ma.masked_where((lkfr >= 0.6), dv)
            dv = np.ma.masked_where((lkfr >= 0.6), dv)
            print "lf_max = {0}".format(lkfr[lkfr < 0.6].max())
            print "lf_susp = {0}".format(lkfr[np.where(dv == dv.min())])
            dv = maskoceans(lons, lats, dv)
        elif var_name in ["TDRA", "TRAF"]:
            dv = maskoceans(lons, lats, dv)
        else:
            pass

        dv = np.ma.masked_where(p > 0.1, dv) #mask changes not significant to the 10 % level

        if not np.all(dv.mask):
            print "{0}: min = {1}; max = {2}".format(var_name, dv.min(), dv.max())
            max_abs = np.max(np.abs(dv))
            delta = max_abs
            the_power = np.log10(delta)

            the_power = np.ceil(the_power) if the_power >= 0 else np.floor(the_power)




            delta = np.ceil( delta / 10.0 ** the_power ) * 10.0 ** the_power
            while delta > 2 * max_abs:
                delta /= 2.0

            step = delta / 5.0 #10 ** (the_power - 1)  * 2 if the_power >= 0 else 10 ** the_power * 2
            levels = np.arange(-delta, delta + step, step)
            bn = BoundaryNorm(levels, len(levels)) if len(levels) > 0 else None

            print "delta={0}; step={1}; the_power={2}".format(delta, step, the_power)

            the_img = basemap.pcolormesh(x, y, dv, cmap=cmap, vmin = -delta, vmax = delta, norm = bn)



            #add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", "15%", pad="5%")
            cb = plt.colorbar(the_img,cax = cax, ticks = levels, format = fmt)


        #coast lines
        basemap.drawcoastlines(ax = ax, linewidth=0.5)

        ax.set_title( "$\Delta$ " + field_name_to_long_name[var_name] + " ({0})".format(field_name_to_units[var_name]))


    assert isinstance(ax, Axes)
    bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="wheat", ec="b", lw=2)
    label = "-".join(map(lambda d: d.strftime("%b"), month_dates)) + " ({0}-{1})".format(start_year, end_year)
    ax.annotate(label, xy = (0.1, 0.9), xycoords = "figure fraction", bbox = bbox_props)


    fig.tight_layout()
    if months is None:
        fig.savefig("annual_mean_diffs_{0}_minus_{1}.jpeg".format(sim_name2, sim_name1))
    else:
        fig.savefig("seasonal_mean_{0}_diffs_{1}_minus_{2}.jpeg".format("_".join(map(str, months)),
            sim_name2, sim_name1) )
    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    from util import plot_utils
    plot_utils.apply_plot_params(width_pt=None, width_cm=40, height_cm=30, font_size= 16)
    #main(months=[12,1,2])
    main(months=range(6,7))
    #main(months=range(6,9))
    #main(months=range(9,12))
    print "Hello world"
  