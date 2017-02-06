from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from scipy.spatial import KDTree

from domains.rotated_lat_lon import RotatedLatLon
from util import direction_and_value
from util.geo import lat_lon

__author__ = 'huziy'

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from domains import grid_config
import numpy as np

from rpn.rpn import RPN


def plot_acc_area_with_glaciers(gmask_vname: str="VF", gmask_level=2,
                                gmask_path="/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/geophys_West_NA_0.25deg_144x115_GLNM_PRSF_CanHR85",
                                route_data_path="/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/directions_north_america_0.25deg_glaciers.nc",
                                lons_target=None, lats_target=None,
                                basin_shape_files=None):

    plot_scatter = False


    # stab reading of the glacier mask
    with RPN(gmask_path) as r:
        gmask = r.get_first_record_for_name_and_level(varname=gmask_vname,
                                                      level=gmask_level)

        # r = RPN("/RESCUE/skynet3_rech1/huziy/CNRCWP/Calgary_flood/pm2013061400_00000000p")
        r.get_first_record_for_name("VF")  # Because I almost sure that PR is there
        proj_params = r.get_proj_parameters_for_the_last_read_rec()
        rll = RotatedLatLon(**proj_params)
        lons_gmask, lats_gmask = r.get_longitudes_and_latitudes_for_the_last_read_rec()


        gl_fraction_limit = 0.01

        gmask = np.ma.masked_where(gmask < gl_fraction_limit, gmask)
        mask_value = 0.25
        gmask[~gmask.mask] = mask_value



    if str(route_data_path).endswith(".nc"):
        print("route_data_path ends with .nc => assuming netcdf format: {}".format(route_data_path))
        with Dataset(route_data_path) as ds:
            var_name = "accumulation_area"
            data = ds.variables[var_name][:]
            # flow directions
            fldr = ds.variables["flow_direction_value"][:]

            coord_names = ["lon", "lat"] if "lon" in ds.variables else ["longitudes", "latitudes"]
            lons_route, lats_route = [ds.variables[k] for k in coord_names]

    else:
        print("route_data_path does not end with .nc => assuming rpn format: {}".format(route_data_path))
        with RPN(route_data_path) as r:
            data = r.get_first_record_for_name("FACC")
            fldr = r.get_first_record_for_name("FLDR")

            lons_route, lats_route = r.get_longitudes_and_latitudes_for_the_last_read_rec()

    # do the spatial interpolation if required
    xg, yg, zg = lat_lon.lon_lat_to_cartesian(lons_gmask.flatten(), lats_gmask.flatten())
    xr, yr, zr = lat_lon.lon_lat_to_cartesian(lons_route.flatten(), lats_route.flatten())



    if lons_target is None or lats_target is None:
        lons_target, lats_target = lons_route, lats_route
        xt, yt, zt = xr, yr, zr
    else:
        xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons_target.flatten(), lats_target.flatten())

    basemap = rll.get_basemap_object_for_lons_lats(lons2d=lons_target, lats2d=lats_target, resolution="i")


    ktree_route = KDTree(list(zip(xr, yr, zr)))
    dists_route, inds_route = ktree_route.query(list(zip(xt, yt, zt)))
    data = data.flatten()[inds_route].reshape(lons_target.shape)
    fldr = fldr.flatten()[inds_route].reshape(lons_target.shape)


    ktree_gmask = KDTree(list(zip(xg, yg, zg)))
    dists_gmask, inds_gmask = ktree_gmask.query(list(zip(xt, yt, zt)))
    gmask = gmask.flatten()[inds_gmask].reshape(lons_target.shape)


    data = np.ma.masked_where(data <= 0, data)


    i_shifts, j_shifts = direction_and_value.flowdir_values_to_shift(fldr)

    xx, yy = basemap(lons_target, lats_target)
    fig = plt.figure(figsize=(15, 15))

    dx = (xx[-1, -1] - xx[0, 0]) / float(xx.shape[0])
    dy = (yy[-1, -1] - yy[0, 0]) / float(yy.shape[1])

    x1 = xx - dx / 2.0
    y1 = yy - dy / 2.0


    # Uncomment to plot the accumulation areas
    im = basemap.pcolormesh(x1, y1, data, norm=LogNorm(vmin=1e3, vmax=1e7), cmap=cm.get_cmap("jet", 12))
    cb = basemap.colorbar(im)



    cmap = cm.get_cmap("gray_r", 10)

    basemap.pcolormesh(x1, y1, gmask, cmap=cmap, vmin=0., vmax=1.)

    nx, ny = xx.shape
    inds_j, inds_i = np.meshgrid(range(ny), range(nx))
    inds_i_next = inds_i + i_shifts
    inds_j_next = inds_j + j_shifts

    inds_i_next = np.ma.masked_where((inds_i_next == nx) | (inds_i_next == -1), inds_i_next)
    inds_j_next = np.ma.masked_where((inds_j_next == ny) | (inds_j_next == -1), inds_j_next)

    u = np.ma.masked_all_like(xx)
    v = np.ma.masked_all_like(xx)

    good = (~inds_i_next.mask) & (~inds_j_next.mask)
    u[good] = xx[inds_i_next[good], inds_j_next[good]] - xx[inds_i[good], inds_j[good]]
    v[good] = yy[inds_i_next[good], inds_j_next[good]] - yy[inds_i[good], inds_j[good]]

    basemap.quiver(xx, yy, u, v,
                   pivot="tail", width=0.0005, scale_units="xy", headlength=20, headwidth=15, scale=1)

    basemap.drawcoastlines(linewidth=0.5, zorder=5)

    basemap.drawrivers(color="lightcoral", zorder=5, linewidth=3)

    plt.legend([Rectangle((0, 0), 5, 5, fc=cmap(mask_value)), ], [r"Glacier ($\geq {}\%$)".format(gl_fraction_limit * 100), ], loc=3)

    watershed_bndry_width = 4


    if basin_shape_files is not None:
        for i, the_shp in enumerate(basin_shape_files):
            basemap.readshapefile(the_shp[:-4], "basin_{}".format(i), zorder=2, color="m", linewidth=watershed_bndry_width)


    # basemap.readshapefile("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/lat_lon/athabasca/athabasca_dissolved", "atabaska",
    #                       zorder=2, linewidth=watershed_bndry_width, color="m")
    #
    # basemap.readshapefile("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/lat_lon/fraizer/fraizer", "frazier",
    #                       zorder=2, linewidth=watershed_bndry_width, color="m")
    #
    # basemap.readshapefile("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/lat_lon/South_sas/South_sas_dissolved", "south_sask",
    #                       zorder=2, linewidth=watershed_bndry_width, color="m")
    #
    # basemap.readshapefile("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/lat_lon/north_sas/north_sas", "north_sask",
    #                       zorder=2, linewidth=watershed_bndry_width, color="m")

    # basemap.readshapefile("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/lat_lon/watersheds_up_sas/watershed_up_sas_proj",
    #                      "upsas",
    #                      zorder=2, linewidth=3, color="m")

    # basemap.readshapefile("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/lat_lon/network/network", "rivers",
    #                      zorder=2, linewidth=0.5, color="b")

    # basemap.readshapefile("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/lat_lon/network_up_sas/network_up_sout_sas_proj", "network",
    #                      zorder=2, linewidth=0.5, color="b")




    if plot_scatter:
        points_lat = [51.54, 49.2476]
        points_lon = [-122.277, -122.784]

        point_x, point_y = basemap(points_lon, points_lat)
        basemap.scatter(point_x, point_y, c="g", s=20, zorder=3)

    plt.savefig("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/directions_only.png", bbox_inches="tight", dpi=300)
    # plt.savefig("/RESCUE/skynet3_rech1/huziy/CNRCWP/Calgary_flood/directions.png", bbox_inches="tight")

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
