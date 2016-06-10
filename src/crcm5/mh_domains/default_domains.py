from domains.grid_config import GridConfig
from domains.rotated_lat_lon import RotatedLatLon

default_projection = RotatedLatLon(lon1=-97.0, lat1=47.5, lon2=-7.0, lat2=0.)


# iref and jref are 1-based indices coming from gemclim_settings.nml

gc_cordex_011 = GridConfig(rll=default_projection, dx=0.11, dy=0.11, ni=695, nj=680, iref=21, jref=580, xref=145.955, yref=28.525)
gc_cordex_022 = GridConfig(rll=default_projection, dx=0.22, dy=0.22, ni=380, nj=360, iref=21, jref=300, xref=146.01, yref=28.47)
gc_cordex_044 = GridConfig(rll=default_projection, dx=0.44, dy=0.44, ni=212, nj=200, iref=21, jref=160, xref=146.12, yref=28.36)



MH_BASINS_PATH = "data/shape/Churchill-Nelson Watershed/Churchill-Nelson Watershed.shp"

GRDC_BASINS_PATH = "data/shape/GIS_dataset_BAFG_GRDC/GRDC_405_basins_from_mouth.shp"