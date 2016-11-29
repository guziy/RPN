import fiona
from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
from pathlib import Path

from shapely.geometry import shape

from application_properties import main_decorator
from crcm5.mh_domains import default_domains
import matplotlib.pyplot as plt
import numpy as np


@main_decorator
def main():
    shp_path = "shp_direction_data_shortnames/directions_bc-mh_0.44deg.shp"
    grid_config = default_domains.bc_mh_044
    bmp = grid_config.get_basemap_for_free_zone(resolution="l")

    assert isinstance(bmp, Basemap)

    assert Path(shp_path).exists()

    gridcells = []
    lake_fractions = []


    with fiona.open(shp_path, "r") as inp:
        for f in inp:
            lkfr = f["properties"]["lkfr"]

            poly = {
                "type": "Polygon",
                # "coordinates": [list(zip(*bmp(*zip(*f["geometry"]["coordinates"][0]))))]
                "coordinates": f["geometry"]["coordinates"]
            }
            poly = shape(poly)

            gridcells.append(PolygonPatch(poly))
            lake_fractions.append(lkfr)

        pcol = PatchCollection(gridcells, cmap="bone_r")
        pcol.set_array(np.array(lake_fractions))


        fig = plt.figure()
        ax = fig.add_subplot(111)
        bmp.ax = ax
        ax.add_collection(pcol)
        bmp.drawcoastlines(ax=ax)
        plt.show()


if __name__ == '__main__':
    main()
