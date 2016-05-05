from domains.grid_config import GridConfig
from domains.rotated_lat_lon import RotatedLatLon

default_projection = RotatedLatLon(lon1=-97.0, lat1=47.5, lon2=-7.0, lat2=0.)

gc_cordex_011 = GridConfig(rll=default_projection, dx=0.1, dy=0.1)



MH_BASINS_PATH = "data/shape/Churchill-Nelson Watershed/Churchill-Nelson Watershed.shp"