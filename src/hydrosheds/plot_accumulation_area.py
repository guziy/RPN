from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from domains.rotated_lat_lon import RotatedLatLon
from util import direction_and_value

__author__ = 'huziy'

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from domains import grid_config
import numpy as np

from rpn.rpn import RPN


def plot_acc_area_with_glaciers():
    gmask_vname = "VF"
    gmask_level = 2
    # gmask_path = "/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/geophys_West_NA_0.25deg_144x115_GLNM_PRSF_CanHR85"
    gmask_path = "/RESCUE/skynet3_rech1/huziy/CNRCWP/Calgary_flood/geophys_CORDEX_NA_0.11deg_695x680_filled_grDes_barBor_Crop2Gras_peat"
    
    # stab reading of the glacier mask
    # r = RPN(gmask_path)
    # gmask = r.get_first_record_for_name_and_level(varname=gmask_vname,
    #                                              level=gmask_level)
    
    r = RPN("/RESCUE/skynet3_rech1/huziy/CNRCWP/Calgary_flood/pm2013061400_00000000p")
    r.get_first_record_for_name("PR") # Because I almost sure that PR is there
    proj_params = r.get_proj_parameters_for_the_last_read_rec()
    rll = RotatedLatLon(**proj_params)
    lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()
    basemap = rll.get_basemap_object_for_lons_lats(lons2d=lons, lats2d=lats, resolution="i")
    # gmask = np.ma.masked_where(gmask < 0.01, gmask)
    gmask = np.ma.masked_all(lons.shape)
    mask_value = 0.25
    gmask[~gmask.mask] = mask_value

    path = "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Java/DDM/directions_WestCaUs_dx0.11deg.nc"
    # path = "/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/infocell_West_NA_0.25deg_104x75_GLNM_PRSF_CanHR85_104x75.nc"
    ds = Dataset(path)


    margin = 20

    var_name = "accumulation_area"
    data = ds.variables[var_name][margin:-margin, margin:-margin]

    data = np.ma.masked_where(data <= 0, data)

    # flow directions
    fldr = ds.variables["flow_direction_value"][:][margin:-margin, margin:-margin]

    i_shifts, j_shifts = direction_and_value.flowdir_values_to_shift(fldr)



    x, y = basemap(lons, lats)
    fig = plt.figure(figsize=(15, 15))

    dx = (x[-1, -1] - x[0, 0]) / float(x.shape[0])
    dy = (y[-1, -1] - y[0, 0]) / float(y.shape[1])

    x1 = x - dx / 2.0
    y1 = y - dy / 2.0

    # im = basemap.pcolormesh(x1, y1, data, norm=LogNorm(vmin=1e3, vmax=1e7), cmap=cm.get_cmap("jet", 12))
    # cb = basemap.colorbar(im)
    # cb.ax.tick_params(labelsize=25)

    cmap = cm.get_cmap("gray_r", 10)

    basemap.pcolormesh(x1, y1, gmask, cmap=cmap, vmin=0., vmax=1.)

    nx, ny = x.shape
    inds_j, inds_i = np.meshgrid(range(ny), range(nx))
    inds_i_next = inds_i + i_shifts
    inds_j_next = inds_j + j_shifts

    inds_i_next = np.ma.masked_where((inds_i_next == nx) | (inds_i_next == -1), inds_i_next)
    inds_j_next = np.ma.masked_where((inds_j_next == ny) | (inds_j_next == -1), inds_j_next)


    u = np.ma.masked_all_like(x)
    v = np.ma.masked_all_like(x)

    good = (~inds_i_next.mask) & (~inds_j_next.mask)
    u[good] = x[inds_i_next[good], inds_j_next[good]] - x[inds_i[good], inds_j[good]]
    v[good] = y[inds_i_next[good], inds_j_next[good]] - y[inds_i[good], inds_j[good]]

    basemap.quiver(x, y, u, v,
                   pivot="tail", width=0.0005, scale_units="xy", headlength=20, headwidth=15, scale=1)

    basemap.drawcoastlines(linewidth=0.5)

    basemap.drawrivers(color="b")

    # plt.legend([Rectangle((0, 0), 5, 5, fc=cmap(mask_value)), ], ["Glaciers", ], loc=3)

    # basemap.readshapefile("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/lat_lon/athabasca/athabasca_dissolved", "atabaska",
    #                      zorder=2, linewidth=3, color="m")

    # basemap.readshapefile("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/lat_lon/fraizer/fraizer", "frazier",
    #                      zorder=2, linewidth=3, color="m")

    # basemap.readshapefile("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/lat_lon/South_sas/South_sas_dissolved", "south_sask",
    #                      zorder=2, linewidth=3, color="m")

    # basemap.readshapefile("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/lat_lon/north_sas/north_sas", "north_sask",
    #                      zorder=2, linewidth=3, color="m")

    # basemap.readshapefile("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/lat_lon/watersheds_up_sas/watershed_up_sas_proj",
    #                      "upsas",
    #                      zorder=2, linewidth=3, color="m")

    # basemap.readshapefile("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/lat_lon/network/network", "rivers",
    #                      zorder=2, linewidth=0.5, color="b")

    # basemap.readshapefile("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/lat_lon/network_up_sas/network_up_sout_sas_proj", "network",
    #                      zorder=2, linewidth=0.5, color="b")


    basemap.readshapefile("/skynet3_exec2/aganji/NE_can/bow_river/bow_projected", "basin", color="m", linewidth=2)

    # plt.savefig("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/directions.png", bbox_inches="tight")
    plt.savefig("/RESCUE/skynet3_rech1/huziy/CNRCWP/Calgary_flood/directions.png", bbox_inches="tight")
    
    plt.show()


def main():
    path = "/b2_fs2/huziy/directions_north_america_0.1375deg.nc"
    ds = Dataset(path)

    var_name = "accumulation_area"
    data = ds.variables[var_name][:]

    data = np.ma.masked_where(data <= 0, data)

    print(list(ds.variables.keys()))

    lons = ds.variables["lon"][:]
    lats = ds.variables["lat"][:]

    rll = grid_config.get_rotpole_for_na_glaciers()

    basemap = rll.get_basemap_object_for_lons_lats(lons2d=lons, lats2d=lats)

    x, y = basemap(lons, lats)
    im = basemap.pcolormesh(x, y, data, norm=LogNorm())
    basemap.drawcoastlines(linewidth=0.5)
    basemap.colorbar(im)
    plt.show()


if __name__ == "__main__":
    # main()
    plot_acc_area_with_glaciers()
