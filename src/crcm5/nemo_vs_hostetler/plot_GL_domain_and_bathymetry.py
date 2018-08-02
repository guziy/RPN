from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.basemap import maskoceans
from rpn.domains.rotated_lat_lon import RotatedLatLon
from rpn.rpn import RPN
import matplotlib.pyplot as plt

from util import plot_utils
import numpy as np
from matplotlib import colors
from matplotlib import patches



def add_rectangle(ax, xx, yy, margin=10, edge_style="solid", **kwargs):

    xll = xx[margin, margin]
    yll = yy[margin, margin]
    xur = xx[-margin, -margin]
    yur = yy[-margin, -margin]


    lw = kwargs.pop("linewidth", 2)

    ax.add_patch(
        patches.Polygon([(xll, yll), (xll, yur), (xur, yur), (xur, yll)], edgecolor="k",
                        facecolor="none", linestyle=edge_style, linewidth=lw, **kwargs)
    )


def main():
    bathymetry_path = ""
    topo_path = "/RECH2/huziy/coupling/coupled-GL-NEMO1h_30min/geophys_452x260_directions_new_452x260_GL+NENA_0.1deg_SAND_CLAY_LDPT_DPTH.fst"




    plot_utils.apply_plot_params()

    with RPN(topo_path) as r:
        assert isinstance(r, RPN)
        topo = r.get_first_record_for_name("ME")
        lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()

        print(lons.shape)

        prj_params = r.get_proj_parameters_for_the_last_read_rec()
        rll = RotatedLatLon(**prj_params)
        bmap = rll.get_basemap_object_for_lons_lats(lons2d=lons, lats2d=lats, resolution="i")


    xx, yy = bmap(lons, lats)

    plt.figure()
    ax = plt.gca()

    lons1 = np.where(lons <= 180, lons, lons - 360)
    topo = maskoceans(lons1, lats, topo)


    topo_clevs = [0, 100, 200, 300, 400, 500, 600, 800, 1000, 1200]
    # bn = BoundaryNorm(topo_clevs, len(topo_clevs) - 1)
    cmap = cm.get_cmap("terrain")

    ocean_color = cmap(0.18)




    cmap, norm = colors.from_levels_and_colors(topo_clevs, cmap(np.linspace(0.3, 1, len(topo_clevs) - 1)))


    add_rectangle(ax, xx, yy, margin=20, edge_style="solid")
    add_rectangle(ax, xx, yy, margin=10, edge_style="dashed")



    im = bmap.pcolormesh(xx, yy, topo, cmap=cmap, norm=norm)
    bmap.colorbar(im, ticks=topo_clevs)
    bmap.drawcoastlines(linewidth=0.3)
    bmap.drawmapboundary(fill_color=ocean_color)
    bmap.drawparallels(np.arange(-90, 90, 10), labels=[1, 0, 0, 1], color="0.3")
    bmap.drawmeridians(np.arange(-180, 190, 10), labels=[1, 0, 0, 1], color="0.3")
    plt.savefig("GL_452x260_0.1deg_domain.png", dpi=300, bbox_inches="tight")

    # plt.show()





if __name__ == '__main__':
    main()