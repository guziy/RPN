import os

from matplotlib import cm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from mpl_toolkits.basemap import maskoceans
from netCDF4 import Dataset
from rpn.domains.rotated_lat_lon import RotatedLatLon
from scipy.spatial import KDTree

from application_properties import main_decorator

from rpn.rpn import RPN

import matplotlib.pyplot as plt

from permafrost.draw_regions import save_pf_mask_to_netcdf
from util import plot_utils
import numpy as np

from util.geo import lat_lon

img_folder = "misc_plots"


@main_decorator
def main():
    plot_permafrost = False
    plot_glaciers = False

    if not os.path.isdir(img_folder):
        print(os.mkdir(img_folder))


    geophy_file = "/BIG1/skynet1_exec1/winger/Floods/Geophys/geophys_CORDEX_NA_0.11deg_695x680_filled_grDes_barBor_Crop2Gras_peat"

    # Save the permafrost mask to file
    pfmask_file = os.path.join(img_folder, "pf_mask_na_0.11deg.nc")

    if not os.path.isfile(pfmask_file):
        save_pf_mask_to_netcdf(rpn_field_name_with_target_grid="VF", path_to_rpn_with_target_grid=geophy_file, path=pfmask_file)

    with Dataset(pfmask_file) as ds:
        pf_mask = ds.variables["pf_type"][:]



    r = RPN(geophy_file)
    lkfr = r.get_record_for_name_and_level(varname="VF", level=3)


    glac = r.get_record_for_name_and_level(varname="VF", level=2)

    lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()

    proj_params = r.get_proj_parameters_for_the_last_read_rec()
    rll = RotatedLatLon(**proj_params)



    ill, jll = 0, 250


    right_margin = 50
    lkfr = lkfr[ill:-right_margin, jll:]
    lons = lons[ill:-right_margin, jll:]
    lats = lats[ill:-right_margin, jll:]
    glac = glac[ill:-right_margin, jll:]
    pf_mask = pf_mask[ill:-right_margin, jll:]

    bmp = rll.get_basemap_object_for_lons_lats(lons2d=lons, lats2d=lats, resolution="l")

    xx, yy = bmp(lons, lats)


    plot_utils.apply_plot_params(font_size=8, width_cm=10, height_cm=10)

    fig = plt.figure()

    # Plot lakes
    clevs = np.arange(0, 1.1, 0.1)
    lkfr = np.ma.masked_where(lkfr <= 0, lkfr)
    cs = bmp.contourf(xx, yy, lkfr, levels=clevs, cmap="Blues")
    # bmp.colorbar(cs)
    bmp.drawcoastlines(linewidth=0.1)
    bmp.drawcountries(linewidth=0.1)



    # plot glaciers
    glval = 0.65
    glac = (glac > 0.001) * glval
    glac = np.ma.masked_where(glac < 0.5, glac)
    cmap = cm.get_cmap("Greys")

    if plot_glaciers:
        cs = bmp.pcolormesh(xx, yy, glac, cmap="Greys", alpha=0.7, vmin=0, vmax=1)
        #bmp.readshapefile(os.path.join(img_folder, "sasha_glaciers/sasha_glaciers"), "glacier_poly", color=cmap(glval))
        plt.legend([Rectangle((0, 0), 5, 5, fc=cmap(glval)), ], ["Glaciers", ], loc=3)



    # Plot permafrost boundaries
    with Dataset("permafrost_types_arctic_using_contains.nc") as ds:
        # pf_mask = ds.variables["pf_type"][:]
        # lons_s, lats_s = ds.variables["longitude"][:], ds.variables["latitude"][:]
        #
        # x, y, z = lat_lon.lon_lat_to_cartesian(lons_s.flatten(), lats_s.flatten())
        # ktree = KDTree(list(zip(x, y, z)))
        #
        # lons_copy = lons.copy()
        # lons_copy[lons > 180] -= 360
        # tmp = np.zeros_like(lons)
        # tmp = maskoceans(lons_copy, lats, tmp, inlands=False)
        #
        # xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons[~tmp.mask], lats[~tmp.mask])
        # dists, inds = ktree.query(list(zip(xt, yt, zt)))
        #

        # pf_mask1[~tmp.mask] = pf_mask.flatten()[inds]

        pf_mask1 = np.ma.masked_where((pf_mask > 2) | (pf_mask < 1), pf_mask)
        pf_mask1[~pf_mask1.mask] = 1



        # bmp.contourf(xx, yy, pf_mask, levels=[2.5, 3.5, 4.5], cmap=ListedColormap(["yellow", "orange", "red"]), alpha=0.7)
        #bmp.contour(xx, yy, pf_mask, levels=[2.5, 3.5, 4.5], colors="k", linewidths=1)

        # bmp.drawrivers(color=cm.get_cmap("Blues")(0.8))

        ncolors = 2
        norm = BoundaryNorm([0.5, 1.5, 2.5], ncolors=ncolors)
        cmap = ListedColormap(["c", "violet"])
        alpha = 0.7

        if plot_permafrost:
            bmp.pcolormesh(xx, yy, pf_mask1, norm=norm, cmap=cmap, alpha=alpha)
            plt.legend([Rectangle((0, 0), 5, 5, fc=cmap(0), alpha=alpha), ], ["Permafrost",], loc=3)




    # tree line
    # bmp.readshapefile(os.path.join(img_folder, "Bernardo/treeline_current"), "treeline", linewidth=0.5, color="orange")
    # bmp.readshapefile(os.path.join(img_folder, "Bernardo/treeline_future"), "treeline", linewidth=0.5, color="orange")


    print(os.path.join(img_folder, "lakes_glaciers_pf_veg.png"))
    fig.savefig(os.path.join(img_folder, "lakes_glaciers_pf_veg.png"), dpi=400, bbox_inches="tight", transparent=True)

if __name__ == '__main__':
    main()