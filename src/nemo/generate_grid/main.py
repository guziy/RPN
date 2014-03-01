import os
from mpl_toolkits.basemap import Basemap
from domains.rotated_lat_lon import RotatedLatLon
from nemo import nemo_commons

__author__ = 'huziy'

#Example header of the coordinates file
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
import nemo_domain_properties as dom_props
import numpy as np
from geopy import distance as gpy_dist
from scipy.spatial import distance as sp_dist


def generate_grid_coordinates():
    lons_rot = np.asarray([dom_props.lonref + (i - dom_props.iref) * dom_props.dx for i in range(1, dom_props.nx + 1)])
    lats_rot = np.asarray([dom_props.latref + (i - dom_props.jref) * dom_props.dy for i in range(1, dom_props.ny + 1)])

    lats_rot, lons_rot = np.meshgrid(lats_rot, lons_rot)
    print lats_rot.shape
    lons_rot[lons_rot < 0] += 360

    rll = RotatedLatLon(lon1=dom_props.lon1, lat1=dom_props.lat1,
                        lon2=dom_props.lon2, lat2=dom_props.lat2)

    truepole_lonr, truepole_latr = rll.get_true_pole_coords_in_rotated_system()
    rotpole_lon, rotpole_lat = rll.get_north_pole_coords()

    llcrnrlon, llcrnrlat = rll.toGeographicLonLat(lons_rot[0, 0], lats_rot[0, 0])
    urcrnrlon, urcrnrlat = rll.toGeographicLonLat(lons_rot[-1, -1], lats_rot[-1, -1])

    b = Basemap(projection="rotpole", lon_0=truepole_lonr - 180, o_lat_p=rotpole_lat, o_lon_p=rotpole_lon,
                llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)

    print lons_rot[0, 0], lats_rot[0, 0], lons_rot[-1, -1], lats_rot[-1, -1]
    lons_real, lats_real = b(lons_rot, lats_rot, inverse = True)

    print "Check consistency of the transformations (below): "
    #from RotatedLatlon
    print "from rotated lat/lon: ", llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat

    #from basemap
    print "from baseap: ", lons_real[0, 0], lats_real[0, 0], lons_real[-1, -1], lats_real[-1, -1]

    #t point coordinates
    tlons, tlats = lons_real, lats_real

    #calculate gridspacing for t grid in x direction
    # Not greatcircle distance expects p = (lat, lon)
    geo_metric = lambda p1, p2: gpy_dist.GreatCircleDistance(p1, p2, radius=nemo_commons.mean_earth_radius_km_crcm5).m

    e1t = [geo_metric(p1, p2) for p1, p2 in zip(zip(lats_rot.flatten(), lons_rot.flatten()),
                                                zip(lats_rot.flatten(), lons_rot.flatten() + dom_props.dx))]

    e2t = [geo_metric(p1, p2) for p1, p2 in zip(zip(lats_rot.flatten(), lons_rot.flatten()),
                                                zip(lats_rot.flatten() + dom_props.dy, lons_rot.flatten()))]

    #u grid sapcing
    e1u = [geo_metric(p1, p2) for p1, p2 in zip(zip(lats_rot.flatten(), lons_rot.flatten() + dom_props.dx / 2.0),
                                                zip(lats_rot.flatten(), lons_rot.flatten() + dom_props.dx * 1.5))]

    e2u = [geo_metric(p1, p2) for p1, p2 in zip(zip(lats_rot.flatten(), lons_rot.flatten() + dom_props.dx / 2.0),
                                                zip(lats_rot.flatten() + dom_props.dy, lons_rot.flatten() + dom_props.dx / 2.0))]


    #v grid sapcing
    e1v = [geo_metric(p1, p2) for p1, p2 in zip(zip(lats_rot.flatten() + dom_props.dy / 2.0, lons_rot.flatten()),
                                                zip(lats_rot.flatten() + dom_props.dy / 2.0, lons_rot.flatten() + dom_props.dx))]

    e2v = [geo_metric(p1, p2) for p1, p2 in zip(zip(lats_rot.flatten() + dom_props.dy / 2.0, lons_rot.flatten()),
                                                zip(lats_rot.flatten() + dom_props.dy * 1.5, lons_rot.flatten()))]


    #f grid sapcing
    e1f = [geo_metric(p1, p2) for p1, p2 in zip(zip(lats_rot.flatten() + dom_props.dy / 2.0, lons_rot.flatten() + dom_props.dx / 2.0),
                                                zip(lats_rot.flatten()+ dom_props.dy / 2.0, lons_rot.flatten() + dom_props.dx * 1.5))]

    e2f = [geo_metric(p1, p2) for p1, p2 in zip(zip(lats_rot.flatten() + dom_props.dy / 2.0, lons_rot.flatten() + dom_props.dx / 2.0),
                                                zip(lats_rot.flatten() + dom_props.dy * 1.5, lons_rot.flatten() + dom_props.dx / 2.0))]


    scales = [e1t, e2t, e1u, e2u, e1v, e2v, e1f, e2f]

    for i in range(len(scales)):
        scales[i] = np.asarray(scales[i])
        scales[i].shape = lons_rot.shape
        scales[i] = scales[i].transpose()
        print scales[i].min(), scales[i].max()

    e1t, e2t, e1u, e2u, e1v, e2v, e1f, e2f = scales

    result = {"T": (tlons.transpose(), tlats.transpose(), e1t, e2t)}

    #u point coordinates
    u_rot_lons, u_rot_lats = lons_rot + dom_props.dx / 2.0, lats_rot
    ulons, ulats = b(u_rot_lons, u_rot_lats, inverse = True)
    result["U"] = (ulons.transpose(), ulats.transpose(), e1u, e2u)

    #v point coordinates
    v_rot_lons, v_rot_lats = lons_rot, lats_rot + dom_props.dy / 2.0
    vlons, vlats = b(v_rot_lons, v_rot_lats, inverse = True)
    result["V"] = (vlons.transpose(), vlats.transpose(), e1v, e2v)

    #f point coordinates
    f_rot_lons, f_rot_lats = lons_rot + dom_props.dx / 2.0, lats_rot + dom_props.dy / 2.0
    flons, flats = b(f_rot_lons, f_rot_lats, inverse = True)
    result["F"] = (flons.transpose(), flats.transpose(), e1f, e2f)
    print flons.shape

    return result




def main():

    out_folder = "nemo_grids"
    out_file = "coordinates_{0}.nc".format(dom_props.config_name)
    out_path = os.path.join(out_folder, out_file)
    ds = Dataset(out_path, "w", format="NETCDF3_CLASSIC")
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    print dom_props.nx, dom_props.ny

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

    coords = generate_grid_coordinates()
    glamt[:], gphit[:], e1t[:], e2t[:] = coords["T"]
    glamu[:], gphiu[:], e1u[:], e2u[:] = coords["U"]
    glamv[:], gphiv[:], e1v[:], e2v[:] = coords["V"]
    glamf[:], gphif[:], e1f[:], e2f[:] = coords["F"]

    ds.close()


if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()
    main()
    pass
