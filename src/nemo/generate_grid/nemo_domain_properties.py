from rpn.domains.rotated_lat_lon import RotatedLatLon

from domains.grid_config import GridConfig

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



class Grid(object):
    def __init__(self, rll=None, nx=None, ny=None,
                 iref=None, jref=None, lonref=None, latref=None, dx=None, dy=None):

        self.latref = latref
        self.lonref = lonref
        self.jref = jref
        self.iref = iref
        self.rll = rll
        self.nx = nx
        self.ny = ny
        self.config_name = "unknown"
        self.dx = dx
        self.dy = dy

    def to_gridconfig(self):
        return GridConfig(rll=self.rll, ni=self.nx, nj=self.ny, iref=self.iref, jref=self.jref,
                          dx=self.dx, dy=self.dy, xref=self.lonref, yref=self.latref)



known_projections = {
    "GLK_210x130_0.1deg": RotatedLatLon(lon1=180., lat1=0., lon2=-84., lat2=1.0)
}


known_domains = {
    "GLK_210x130_0.1deg": Grid(rll=RotatedLatLon(lon1=180., lat1=0., lon2=-84., lat2=1.0),
                               nx=210, ny=130, iref=105, jref=100, lonref=-84, latref=48, dx=0.1, dy=0.1),

    "GLK_440x260_0.1deg": Grid(rll=RotatedLatLon(lon1=180., lat1=0., lon2=-84., lat2=1.0),
                               nx=440, ny=260, iref=135, jref=120, lonref=-84, latref=48, dx=0.1, dy=0.1),

    "GLK_452x260_0.1deg": Grid(rll=RotatedLatLon(lon1=180., lat1=0., lon2=-84., lat2=1.0),
                               nx=452, ny=260, iref=135, jref=120, lonref=-84, latref=48, dx=0.1, dy=0.1)
}





