

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pathlib import Path

from application_properties import main_decorator
from crcm5.mh_domains import default_domains
from domains.grid_config import GridConfig
from util import plot_utils

img_folder = "mh"


GRDC_basins_of_interest = [19, 16, 88, 107]


def show_domain(grid_config, halo=None, blending=None, draw_rivers=True, show_GRDC_basins=False):
    assert isinstance(grid_config, GridConfig)
    fig = plt.figure()

    ax = plt.gca()

    halo = 10 if halo is None else halo
    blending = 10 if blending is None else blending

    bmp = grid_config.get_basemap(resolution="i")
    bmp.readshapefile(default_domains.MH_BASINS_PATH[:-4], "basin", color="m", linewidth=2)

    if show_GRDC_basins:
        # Select which basins to show
        bmp.readshapefile(default_domains.GRDC_BASINS_PATH[:-4], "basin", drawbounds=False)

        patches = []

        for info, shape in zip(bmp.basin_info, bmp.basin):
            if info["BASIN_ID"] in GRDC_basins_of_interest:
                patches.append(Polygon(np.array(shape), True))

        ax.add_collection(PatchCollection(patches, facecolor='none', edgecolor='b', linewidths=2., zorder=2))



    lons, lats = grid_config.get_free_zone_corners(halo=halo, blending=blending)

    xx, yy = bmp(lons, lats)
    ax = plt.gca()

    coords = [(xx[0, 0], yy[0, 0]), (xx[0, -1], yy[0, -1]), (xx[-1, -1], yy[-1, -1]), (xx[-1, 0], yy[-1, 0])]
    ax.add_patch(Polygon(coords, facecolor="none"))


    if draw_rivers:
        bmp.drawrivers()

    bmp.drawcoastlines(linewidth=0.3)
    bmp.drawstates(linewidth=0.3)
    bmp.drawcountries(linewidth=0.3)

    ax.set_title(r"${}".format(grid_config.ni) + r"\times" + "{}$ grid cells, resolution {} $^\circ$".format(grid_config.nj, grid_config.dx))


    p = Path(img_folder)
    if not p.exists():
        p.mkdir()


    fig.savefig(str(p.joinpath("mh_dx{}.png".format(grid_config.dx))), bbox_inches="tight", transparent=True)


@main_decorator
def main():
    # Show selected domains, basins, and/or flow directions or flow accumulations


    mh_gc044 = default_domains.gc_cordex_044.subgrid(20, 60, di=130, dj=110)
    mh_gc022 = mh_gc044.double_resolution_keep_free_domain_same()
    mh_gc011 = mh_gc022.double_resolution_keep_free_domain_same()

    test_bc = default_domains.gc_cordex_011.subgrid(0, 260, di=420, dj=380)




    plot_utils.apply_plot_params()
    # show_domain(mh_gc044)
    # show_domain(mh_gc022)
    # show_domain(mh_gc011)


    show_domain(test_bc, draw_rivers=False, show_GRDC_basins=True)

    plt.show()
    pass


if __name__ == '__main__':
    main()
