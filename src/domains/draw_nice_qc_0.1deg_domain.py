from netCDF4 import Dataset
import numpy as np
from domains.grid_config import GridConfig

import matplotlib.pyplot as plt

params = dict(
        dx=0.1, dy=0.1,
        lon1=180, lat1=0.0,
        lon2=-84, lat2=1.0,
        iref=105, jref=100,
        ni=210, nj=130,
        xref=276.0, yref=48.0
)


PARAMS_GL_01DEG_EXTENDED = dict(

)


gc = GridConfig(**params)


def show_domain(grid_config=None):

    assert isinstance(grid_config, GridConfig)


    pass


def main():
    pass


def main_qc_test():

    nc_path_to_directions = "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Java/DDM/directions_qc_dx0.1deg_5.nc"

    ds = Dataset(nc_path_to_directions)

    lons, lats = [ds.variables[key][:, :] for key in ["lon", "lat"]]

    bsmap = gc.get_basemap(lons=lons, lats=lats)
    x, y = bsmap(lons, lats)
    fig = plt.figure(figsize=(15, 15))

    bsmap.drawcoastlines()
    bsmap.shadedrelief()
    bsmap.drawrivers()
    plt.show()


if __name__ == '__main__':
    main()

