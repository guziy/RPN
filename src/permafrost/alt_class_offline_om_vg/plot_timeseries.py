import os
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap
from rpn.domains.rotated_lat_lon import RotatedLatLon
from rpn.rpn import RPN
from permafrost.active_layer_thickness import CRCMDataManager
import numpy as np

__author__ = 'huziy'

import matplotlib.pyplot as plt


def plot_time_series(data=None, i_interest=-1, j_interest=-1, soil_levels=None,
                     basemap=None, longitude=None, latitude=None, exp_name=""):

    assert isinstance(basemap, Basemap)

    fig = plt.figure()
    gs = GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    basemap.drawcoastlines(linewidth=0.5, ax=ax1)
    x, y = basemap(longitude, latitude)
    basemap.scatter(x, y, c="red")


    ax2 = fig.add_subplot(gs[1, 0])
    assert isinstance(ax2, Axes)

    t = range(1, data.shape[0] + 1)
    h2d, t2d = np.meshgrid(soil_levels, t)

    clevs = [0, 0.04, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    bn = BoundaryNorm(clevs, len(clevs) - 1)
    cmap = cm.get_cmap("jet", len(clevs) - 1)
    cs = ax2.contourf(t2d, h2d, data, levels=clevs, norm=bn, cmap=cmap)
    ax2.invert_yaxis()
    plt.colorbar(cs, ax=ax2, ticks=clevs)
    ax2.set_title(exp_name)

    fig.savefig("{}_{}_{}_thwat_ts.png".format(exp_name, i_interest, j_interest),
                bbox_inches="tight")


def main():
    path = "/skynet1_rech3/huziy/class_offline_simulations_VG/dpth_3.6m/CLASS_output_CLASSoff1_Arctic_0.5_ERA40_dpth_to_bdrck_constant_spinup_200years_dpth_3.6m/THWAT.rpn"
    read_and_plot_ts_cross(path=path, exp_name="DPTH=3.6m")

    path = "/skynet1_rech3/huziy/class_offline_simulations_VG/dpth_var/CLASS_output_CLASSoff1_Arctic_0.5_ERA40_dpth_to_bdrck_var_spinup_200years/THWAT.rpn"
    read_and_plot_ts_cross(path=path, exp_name="DPTH=real")





def read_and_plot_ts_cross(path="", exp_name=""):
    var_interest = "ADD"


    path_to_dpth_to_bedrock = "/skynet1_rech3/huziy/CLASS_offline_VG/GEOPHYSICAL_FIELDS/test_analysis.rpn"

    # read depth to bedrock
    r = RPN(path_to_dpth_to_bedrock)
    _ = r.get_first_record_for_name("DPTH")

    lons2d, lats2d = r.get_longitudes_and_latitudes_for_the_last_read_rec()
    rll = RotatedLatLon(**r.get_proj_parameters_for_the_last_read_rec())
    b = rll.get_basemap_object_for_lons_lats(lons2d=lons2d, lats2d=lats2d, resolution="c")
    r.close()

    layer_widths = [0.1, 0.2, 0.3, 0.5, 0.9, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]


    nlayers = 7
    nt = 200*12
    layer_widths = layer_widths[:nlayers]

    print len(layer_widths)


    #calculate depths of soil layer centers
    soil_lev_tops = np.cumsum([0, ] + layer_widths[:-1])
    soil_lev_bottoms = np.cumsum(layer_widths)
    soil_levs = 0.5 * (soil_lev_tops + soil_lev_bottoms)


    i_interest_list, j_interest_list = [120, 120, 160, 170], [50, 60, 60, 60]

    r = RPN(path)

    data = r.get_4d_field_fc_hour_as_time(name=var_interest)
    lev_sorted = list(sorted(data.items()[0][1].keys()))[:nlayers]
    fc_sorted = list(sorted(data.keys()))[:nt]

    for i_interest, j_interest in zip(i_interest_list, j_interest_list):
        data1 = np.asarray(
            [[data[fc][lev][i_interest, j_interest] for lev in lev_sorted] for fc in fc_sorted])

        plot_time_series(data=data1, soil_levels=soil_levs, basemap=b, i_interest=i_interest,
                         j_interest=j_interest,
                         longitude=lons2d[i_interest, j_interest],
                         latitude=lats2d[i_interest, j_interest], exp_name=exp_name)


if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()
    main()
