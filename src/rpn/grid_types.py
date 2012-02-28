__author__ = 'huziy'

# A - grid (no grid point at the pole or equator)
GLOBAL_LAT_LON_TYPE = 'A'
#ig1 values
GLOBAL_EXTENT = 0
NH_EXTENT = 1
SH_EXTENT = 2
#ig2 values
START_AT_LOWER_LEFT = 0 # pt(1,1) is at the bottom of the grid
START_AT_UPPER_LEFT = 1 # pt(1,1) is at the top of the grid

#ig3 and ig4 should be 0 for the A-grid

## B-grid (there is grid point at the pole or equator)
#TODO



#G-grid
GAUSSIAN_TYPE = "G"

#L-grid
CYLINDRICAL_EQUIDISTANT = "L"


#E-grid
ROTATED_LAT_LON_TYPE = "E"