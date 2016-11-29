

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pathlib import Path

from mpl_toolkits.axisartist import Axes

from application_properties import main_decorator
from crcm5.mh_domains import default_domains
from domains.grid_config import GridConfig
from util import plot_utils

img_folder = "mh"


def show_multiple_domains(label_to_config):
    # TODO: implement
    pass


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
            if info["BASIN_ID"] in default_domains.GRDC_basins_of_interest:
                patches.append(Polygon(np.array(shape), True))

        ax.add_collection(PatchCollection(patches, facecolor='none', edgecolor='b', linewidths=2., zorder=2))


    lons, lats = grid_config.get_free_zone_corners(halo=halo, blending=blending)

    xx, yy = bmp(lons, lats)


    coords = [(xx[0, 0], yy[0, 0]), (xx[0, -1], yy[0, -1]), (xx[-1, -1], yy[-1, -1]), (xx[-1, 0], yy[-1, 0])]
    ax.add_patch(Polygon(coords, facecolor="none"))


    if draw_rivers:
        bmp.drawrivers()

    bmp.drawcoastlines(linewidth=0.3, ax=ax)
    bmp.drawstates(linewidth=0.3, ax=ax)
    bmp.drawcountries(linewidth=0.3, ax=ax)


    p = Path(img_folder)
    if not p.exists():
        p.mkdir()



    ax.set_title(
        r"${}".format(grid_config.ni) + r"\times" + "{}$ grid cells, resolution {} $^\circ$".format(grid_config.nj,
                                                                                                    grid_config.dx))

    fig.savefig(str(p.joinpath("mh_dx{}.png".format(grid_config.dx))), bbox_inches="tight", transparent=True)



def show_bc_mh_domains():

    pass



@main_decorator
def main():
    # Show selected domains, basins, and/or flow directions or flow accumulations


    mh_gc044 = default_domains.gc_cordex_na_044.subgrid(20, 60, di=130, dj=110)
    mh_gc022 = mh_gc044.double_resolution_keep_free_domain_same()

    test_bc_011 = default_domains.gc_cordex_na_011.subgrid(12, 244, di=404, dj=380)
    test_bc_044 = test_bc_011.decrease_resolution_keep_free_domain_same(4)



    print(test_bc_044)

    plot_utils.apply_plot_params()
    # show_domain(mh_gc044)
    # show_domain(mh_gc022)
    # show_domain(mh_gc011)


    print(test_bc_011)

    # fig, ax, bmp = show_domain(default_domains.gc_cordex_011, draw_rivers=False)
    # show_domain(test_bc_011, draw_rivers=False, show_GRDC_basins=True)
    # show_domain(test_bc_044, draw_rivers=False, show_GRDC_basins=True)


    show_domain(default_domains.bc_mh_044)
    show_domain(default_domains.bc_mh_011)

    plt.show()


if __name__ == '__main__':
    main()
