import os
from mpl_toolkits.basemap import Basemap
from domains.rotated_lat_lon import RotatedLatLon
from nemo import nemo_commons
from nemo.create_initial_temperature_and_salinity_files import create_file_with_field
from nemo.generate_grid.nemo_domain_properties import known_domains
from nemo.interpolate_dfs_data import Interpolator

__author__ = 'huziy'

# Example header of the coordinates file
# netcdf coordinates {
# dimensions:
# 	x = 355 ;
# 	y = 435 ;
# variables:
# 	double glamt(y, x) ;
# 		glamt:units = "degrees_east" ;
# 	double glamu(y, x) ;
# 		glamu:units = "degrees_east" ;
# 	double glamv(y, x) ;
# 		glamv:units = "degrees_east" ;
# 	double glamf(y, x) ;
# 		glamf:units = "degrees_east" ;
# 	double gphit(y, x) ;
# 		gphit:units = "degrees_north" ;
# 	double gphiu(y, x) ;
# 		gphiu:units = "degrees_north" ;
# 	double gphiv(y, x) ;
# 		gphiv:units = "degrees_north" ;
# 	double gphif(y, x) ;
# 		gphif:units = "degrees_north" ;
# 	double e1t(y, x) ;
# 	double e1u(y, x) ;
# 	double e1v(y, x) ;
# 	double e1f(y, x) ;
# 	double e2t(y, x) ;
# 	double e2u(y, x) ;
# 	double e2v(y, x) ;
# 	double e2f(y, x) ;
# }


from netCDF4 import Dataset
import numpy as np
from geopy import distance as gpy_dist
from scipy.spatial import distance as sp_dist

import matplotlib.pyplot as plt


def generate_grid_coordinates(dom_props=None):
    lons_rot = np.asarray([dom_props.lonref + (i - dom_props.iref) * dom_props.dx for i in range(1, dom_props.nx + 1)])
    lats_rot = np.asarray([dom_props.latref + (i - dom_props.jref) * dom_props.dy for i in range(1, dom_props.ny + 1)])

    lats_rot, lons_rot = np.meshgrid(lats_rot, lons_rot)
    print(lats_rot.shape)
    lons_rot[lons_rot < 0] += 360

    rll = dom_props.rll

    truepole_lonr, truepole_latr = rll.get_true_pole_coords_in_rotated_system()
    rotpole_lon, rotpole_lat = rll.get_north_pole_coords()

    llcrnrlon, llcrnrlat = rll.toGeographicLonLat(lons_rot[0, 0], lats_rot[0, 0])
    urcrnrlon, urcrnrlat = rll.toGeographicLonLat(lons_rot[-1, -1], lats_rot[-1, -1])

    b = Basemap(projection="rotpole", lon_0=truepole_lonr - 180, o_lat_p=rotpole_lat, o_lon_p=rotpole_lon,
                llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)

    print(lons_rot[0, 0], lats_rot[0, 0], lons_rot[-1, -1], lats_rot[-1, -1])



    lons_real, lats_real = b(lons_rot, lats_rot, inverse=True)



    print("Check consistency of the transformations (below): ")
    # from RotatedLatlon
    print("from rotated lat/lon: ", llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat)

    # from basemap
    print("from basemap: ", lons_real[0, 0], lats_real[0, 0], lons_real[-1, -1], lats_real[-1, -1])

    # t point coordinates
    tlons, tlats = lons_real, lats_real

    # calculate gridspacing for t grid in x direction
    # Not greatcircle distance expects p = (lat, lon)
    geo_metric = lambda p1, p2: gpy_dist.GreatCircleDistance(p1, p2, radius=nemo_commons.mean_earth_radius_km_crcm5).m

    e1t = [geo_metric(p1, p2) for p1, p2 in zip(list(zip(lats_rot.flatten(), lons_rot.flatten())),
                                                list(zip(lats_rot.flatten(), lons_rot.flatten() + dom_props.dx)))]

    e2t = [geo_metric(p1, p2) for p1, p2 in zip(list(zip(lats_rot.flatten(), lons_rot.flatten())),
                                                list(zip(lats_rot.flatten() + dom_props.dy, lons_rot.flatten())))]

    # u grid sapcing
    e1u = [geo_metric(p1, p2) for p1, p2 in zip(list(zip(lats_rot.flatten(), lons_rot.flatten() + dom_props.dx / 2.0)),
                                                list(zip(lats_rot.flatten(), lons_rot.flatten() + dom_props.dx * 1.5)))]

    e2u = [geo_metric(p1, p2) for p1, p2 in zip(list(zip(lats_rot.flatten(), lons_rot.flatten() + dom_props.dx / 2.0)),
                                                list(zip(lats_rot.flatten() + dom_props.dy,
                                                    lons_rot.flatten() + dom_props.dx / 2.0)))]


    # v grid sapcing
    e1v = [geo_metric(p1, p2) for p1, p2 in zip(list(zip(lats_rot.flatten() + dom_props.dy / 2.0, lons_rot.flatten())),
                                                list(zip(lats_rot.flatten() + dom_props.dy / 2.0,
                                                    lons_rot.flatten() + dom_props.dx)))]

    e2v = [geo_metric(p1, p2) for p1, p2 in zip(list(zip(lats_rot.flatten() + dom_props.dy / 2.0, lons_rot.flatten())),
                                                list(zip(lats_rot.flatten() + dom_props.dy * 1.5, lons_rot.flatten())))]


    # f grid sapcing
    e1f = [geo_metric(p1, p2) for p1, p2 in
           zip(list(zip(lats_rot.flatten() + dom_props.dy / 2.0, lons_rot.flatten() + dom_props.dx / 2.0)),
               list(zip(lats_rot.flatten() + dom_props.dy / 2.0, lons_rot.flatten() + dom_props.dx * 1.5)))]

    e2f = [geo_metric(p1, p2) for p1, p2 in
           zip(list(zip(lats_rot.flatten() + dom_props.dy / 2.0, lons_rot.flatten() + dom_props.dx / 2.0)),
               list(zip(lats_rot.flatten() + dom_props.dy * 1.5, lons_rot.flatten() + dom_props.dx / 2.0)))]

    scales = [e1t, e2t, e1u, e2u, e1v, e2v, e1f, e2f]

    for i in range(len(scales)):
        scales[i] = np.asarray(scales[i])
        scales[i].shape = lons_rot.shape
        scales[i] = scales[i].transpose()
        print(scales[i].min(), scales[i].max())

    e1t, e2t, e1u, e2u, e1v, e2v, e1f, e2f = scales

    result = {"T": (tlons.transpose(), tlats.transpose(), e1t, e2t)}

    # u point coordinates
    u_rot_lons, u_rot_lats = lons_rot + dom_props.dx / 2.0, lats_rot
    ulons, ulats = b(u_rot_lons, u_rot_lats, inverse=True)
    result["U"] = (ulons.transpose(), ulats.transpose(), e1u, e2u)

    # v point coordinates
    v_rot_lons, v_rot_lats = lons_rot, lats_rot + dom_props.dy / 2.0
    vlons, vlats = b(v_rot_lons, v_rot_lats, inverse=True)
    result["V"] = (vlons.transpose(), vlats.transpose(), e1v, e2v)

    # f point coordinates
    f_rot_lons, f_rot_lats = lons_rot + dom_props.dx / 2.0, lats_rot + dom_props.dy / 2.0
    flons, flats = b(f_rot_lons, f_rot_lats, inverse=True)
    result["F"] = (flons.transpose(), flats.transpose(), e1f, e2f)
    print(flons.shape)

    return result


def main():
    out_folder = "nemo_grids"

    # config_name = "GLK_452x260_0.1deg"
    #config_name = "GLK_210x130_0.1deg"
    config_name = "GLK_452x260_0.1deg_shift"
    dom_props = known_domains[config_name]
    dom_props.config_name = config_name

    out_file = "coordinates_{0}.nc".format(dom_props.config_name)
    out_path = os.path.join(out_folder, out_file)
    ds = Dataset(out_path, "w", format="NETCDF3_CLASSIC")
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)



    print(dom_props.nx, dom_props.ny)

    ds.createDimension("x", size=dom_props.nx)
    ds.createDimension("y", size=dom_props.ny)

    glamt = ds.createVariable("glamt", "f4", ("y", "x"))
    gphit = ds.createVariable("gphit", "f4", ("y", "x"))

    glamu = ds.createVariable("glamu", "f4", ("y", "x"))
    gphiu = ds.createVariable("gphiu", "f4", ("y", "x"))

    glamv = ds.createVariable("glamv", "f4", ("y", "x"))
    gphiv = ds.createVariable("gphiv", "f4", ("y", "x"))

    glamf = ds.createVariable("glamf", "f4", ("y", "x"))
    gphif = ds.createVariable("gphif", "f4", ("y", "x"))

    e1t = ds.createVariable("e1t", "f4", ("y", "x"))
    e2t = ds.createVariable("e2t", "f4", ("y", "x"))

    e1u = ds.createVariable("e1u", "f4", ("y", "x"))
    e2u = ds.createVariable("e2u", "f4", ("y", "x"))

    e1v = ds.createVariable("e1v", "f4", ("y", "x"))
    e2v = ds.createVariable("e2v", "f4", ("y", "x"))

    e1f = ds.createVariable("e1f", "f4", ("y", "x"))
    e2f = ds.createVariable("e2f", "f4", ("y", "x"))

    coords = generate_grid_coordinates(dom_props=dom_props)
    glamt[:], gphit[:], e1t[:], e2t[:] = coords["T"]
    glamu[:], gphiu[:], e1u[:], e2u[:] = coords["U"]
    glamv[:], gphiv[:], e1v[:], e2v[:] = coords["V"]
    glamf[:], gphif[:], e1f[:], e2f[:] = coords["F"]

    ds.close()

    # interpolate bathymetry
    interpolator = Interpolator(coord_file=out_path)
    interpolator.interpolate_file("/RESCUE/skynet3_rech1/huziy/GLK_bathymetry/GLK_bathymetry_from_EC/bathy_meter.nc",
                                  os.path.join(out_folder, "bathy_meter_{}.nc".format(config_name)))

    #  Create initial conditions file
    t_file_name = "IC_T_{}.nc".format(config_name)
    t_var_name = "votemper"

    s_file_name = "IC_S_{}.nc".format(config_name)
    s_var_name = "vosaline"

    # the_shape = 35, ny, nx
    the_shape = 23, dom_props.ny, dom_props.nx
    initial_temperature = 4.2 * np.ones(the_shape)
    initial_salinity = 0.0 * np.ones(the_shape)

    create_file_with_field(folder=out_folder, fname=t_file_name, var_name=t_var_name, data=initial_temperature)
    create_file_with_field(folder=out_folder, fname=s_file_name, var_name=s_var_name, data=initial_salinity)


if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()
    main()
