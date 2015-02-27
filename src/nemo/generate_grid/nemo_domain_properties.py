from rpn.domains.rotated_lat_lon import RotatedLatLon

__author__ = 'huziy'



# nx, dx = 170, 0.1
# ny, dy = 90, 0.1

nx, dx = 210, 0.1
ny, dy = 130, 0.1

# Indices are 1-based
iref, jref = 105, 100
lonref, latref = -84, 48

# projection properties (rotated lat/lon)
lon1, lat1 = 180, 0
lon2, lat2 = -84, 1

config_name = "rotpole_nx{0}_ny{1}_dx{2}_dy{3}".format(nx, ny, dx, dy)



known_domains = {
    "GLK_210x130_0.1deg": RotatedLatLon(lon1=180., lat1=0., lon2=-84., lat2=1.0)
}