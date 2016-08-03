from domains.grid_config import GridConfig, gridconfig_from_grid_nml
from domains.rotated_lat_lon import RotatedLatLon

default_projection = RotatedLatLon(lon1=-97.0, lat1=47.5, lon2=-7.0, lat2=0.)


# iref and jref are 1-based indices coming from gemclim_settings.nml

gc_cordex_011 = GridConfig(rll=default_projection, dx=0.11, dy=0.11, ni=695, nj=680, iref=21, jref=580, xref=145.955, yref=28.525)
gc_cordex_022 = GridConfig(rll=default_projection, dx=0.22, dy=0.22, ni=380, nj=360, iref=21, jref=300, xref=146.01, yref=28.47)
gc_cordex_044 = GridConfig(rll=default_projection, dx=0.44, dy=0.44, ni=212, nj=200, iref=21, jref=160, xref=146.12, yref=28.36)



cordex_arctic_proj = RotatedLatLon(lon1=180, lat1=83.45, lon2=270, lat2=0.)
gc_cordex_Arctic_044 = GridConfig(
    rll=cordex_arctic_proj, dx=0.44, dy=0.44, ni=164, nj=180, iref=21, jref=153, xref=157.12, yref=33.88
)

cordex_na_proj = RotatedLatLon(lon1=-97, lat1=47.5, lon2=-7.0, lat2=0.0)
gc_cordex_NA_044 = GridConfig(
    rll=cordex_na_proj, dx=0.44, dy=0.44, ni=212, nj=200, iref=21, jref=160, xref=146.12, yref=28.36
)

gc_panarctic_05 = gridconfig_from_grid_nml(
    """
    &grid
      Grd_typ_S     = 'LU'   ,
      Grd_dx        =   0.5  ,  Grd_dy          = 0.5,
      Grd_ni        = 212    ,  Grd_nj          = 212,
      Grd_iref      =  35    ,  Grd_jref        =  48,
      Grd_lonr      = 144.00 ,  Grd_latr        = -28.25,
      Grd_xlat1     =  90.   ,  Grd_xlon1       =  60.,
      Grd_xlat2     =   0.   ,  Grd_xlon2       = -30.,
    /
    """
)


gc_GL_and_NENA_01 = gridconfig_from_grid_nml(
    """
 &grid

  Grd_typ_S     = 'LU'   ,
  Grd_ni        =   440  ,   Grd_nj          =   260   ,
  Grd_dx        =    0.1,   Grd_dy          =    0.1 ,
  Grd_iref      =   135  ,   Grd_jref        =   120   ,
  Grd_latr      =    48,   Grd_lonr        =  -84 ,
  Grd_xlat1     =   0,   Grd_xlon1       =  180,
  Grd_xlat2     =    1.0 ,   Grd_xlon2       =   -84,

 /
    """
)


MH_BASINS_PATH = "data/shape/Churchill-Nelson Watershed/Churchill-Nelson Watershed.shp"

GRDC_BASINS_PATH = "data/shape/GIS_dataset_BAFG_GRDC/GRDC_405_basins_from_mouth.shp"


# For selecting basins from the shape file
GRDC_basins_of_interest = [19, 16, 88, 107, 83, 72]

GRDC_basins_of_interest_NA = GRDC_basins_of_interest + [122, 138, 117]

GRDC_basins_of_interest_Panarctic = [19, 25, 5, 7, 138]

GRDC_basins_GL = [117]



# Create the domains for the bc_mh project
bc_mh_011 = gc_cordex_011.subgrid(12, 244, di=404, dj=380)
bc_mh_022 = bc_mh_011.decrease_resolution_keep_free_domain_same(2)
bc_mh_044 = bc_mh_011.decrease_resolution_keep_free_domain_same(4)
