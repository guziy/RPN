# I use SI units whenever it is not specified (at least try to)

lower_limit_of_daily_snowfall = 10.0e-2 / (24.0 * 3600.0)   # i.e. = 10 cm/day (Note: it is in snowfall depth not water equivalent)
snfl_local_amplification_m_per_s = 4.0e-2 / (24.0 * 3600.0)   # i.e. = 4 cm/day



class LatLonLimits(object):

    def __init__(self, lon_min, lon_max, lat_min, lat_max):
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max

    def get_mask_for_coords(self, lons2d, lats2d):
        mask = (lons2d > self.lon_min) & (lons2d < self.lon_max)
        mask = (lats2d > self.lat_min) & (lats2d < self.lat_max) & mask
        return mask



great_lakes_limits = LatLonLimits(265, 288, 40, 50)

GL_COAST_SHP_PATH = "data/shp/Great_lakes_coast_shape/gl_cst.shp"