from pathlib import Path

from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import AxesImage
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap, maskoceans
from rpn.rpn import RPN

import matplotlib.pyplot as plt
import numpy as np

from netCDF4 import Dataset

from domains.rotated_lat_lon import RotatedLatLon
from util import plot_utils


def get_hadgem_topo():
    path = "/snow3/huziy/geophysics_files/me_glob_hadgem.rpn"
    with RPN(path) as r:
        topo = r.variables["ME"][:]
        lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()

    return topo.squeeze(), lons, lats



def get_etopo():

    path = "/snow2/teufel/ETOPO1_Ice_c_gmt4.grd"

    with Dataset(path) as ds:
        x = ds.variables["x"][:]
        y = ds.variables["y"][:]
        etopo = ds.variables["z"][:]

    lons, lats = np.meshgrid(x, y)

    return etopo, lons, lats


def get_cmap(clevs):
    cmap_cols = cm.get_cmap("terrain")(np.linspace(0.25, 1, 100))
    cmap = LinearSegmentedColormap.from_list("terrain_cut", cmap_cols, N=len(clevs) - 1)

    return cmap


def show_coast_lines(bmp, resolution="h"):
    path = "data/shp/GSHHS_shp/{}".format(resolution)



    p = Path(path)

    for f in p.iterdir():
        if f.name.endswith(".shp"):
            bmp.readshapefile(str(f)[:-4], "coast")





def get_grid_polygon(rll_obj, rlons, rlats, bmp=None, **kwargs):
    dlon0 = rlons[1] - rlons[0]
    dlat0 = rlats[1] - rlats[0]


    rlon_ll = rlons[0] - dlon0 / 2.0
    rlat_ll = rlats[0] - dlat0 / 2.0


    rlon_ur = rlon_ll + dlon0 * len(rlons)
    rlat_ur = rlat_ll + dlat0 * len(rlats)


    points = []

    for i in range(len(rlons) + 1):
        points.append((rlon_ll + i * dlon0, rlat_ll))


    for j in range(1, len(rlats) + 1):
        points.append((rlon_ur, rlat_ll + j * dlat0))


    for i in range(len(rlons), -1, -1):
        points.append((rlon_ll + i * dlon0, rlat_ur))

    for j in range(len(rlats), 0, -1):
        points.append((rlon_ll, rlat_ll + j * dlat0))


    if bmp is not None:
        points = [bmp(*rll_obj.toGeographicLonLat(*p)) for p in points]
    else:
        points = [rll_obj.toGeographicLonLat(*p) for p in points]

    return Polygon(np.array(points), facecolor="none", **kwargs)




def get_gridcell_polygons(rll_obj, rlons, rlats, step=10, bmp=None, **kwargs):

    """
    Assumes uniform grid
    :param rlons: longitudes of the centers of gridcells
    :param rlats:
    """

    assert isinstance(rll_obj, RotatedLatLon)


    dlon0 = rlons[1] - rlons[0]
    dlat0 = rlats[1] - rlats[0]

    dlon1 = dlon0 * step
    dlat1 = dlat0 * step

    rlon_ll = rlons[0] - dlon0 / 2.0
    rlat_ll = rlats[0] - dlat0 / 2.0

    nx, ny = len(rlons), len(rlats)


    polys = []
    for i in range(nx // step):
        for j in range(ny // step):
            ll = (rlon_ll + i * dlon1, rlat_ll + j * dlat1)
            ul = (ll[0], ll[1] + dlat1)
            ur = (ll[0] + dlon1, ul[1])
            lr = (ur[0], ll[1])

            corners = [ll, ul, ur, lr]

            latlon_proj = [rll_obj.toGeographicLonLat(*p) for p in corners]
            latlon_proj = [(p[0] - 360, p[1]) if p[0] > 180 else p for p in latlon_proj]

            if bmp is not None:
                map_proj = [bmp(*p) for p in latlon_proj]
            else:
                map_proj = latlon_proj

            polys.append(Polygon(np.array(map_proj), edgecolor="k", facecolor="none", **kwargs))




    return polys




def main():

    path = "/snow3/huziy/geophysics_files/geophys_452x260_0.1deg_GL_NENA_and_directions_452x260_GL+NENA_0.1deg_SAND_CLAY_LDPT_DPTH.fst"


    # get rotated latlon projection and grid parameters
    with RPN(path) as r:

        assert isinstance(r, RPN)
        lkfr = r.variables["LKFR"][:].squeeze()
        print(r.variables)

        lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()
        rll_obj = RotatedLatLon(**r.get_proj_parameters_for_the_last_read_rec())

        rlon, rlat = r.get_tictacs_for_the_last_read_record()





    plot_utils.apply_plot_params()

    lons[lons > 180] -= 360

    b = Basemap(projection="ortho", lat_0=50, lon_0=-80, resolution="l")

    #
    # b.drawcoastlines(linewidth=0.5)
    # b.shadedrelief()
    # b.warpimage()

    b.drawmapboundary(fill_color="#94e3e8")

    # aximg = b.etopo()
    # assert isinstance(aximg, AxesImage)
    # aximg.set_cmap("bwr")



    plot_topo = True

    if plot_topo:
        topo, lonst, latst = get_etopo()
        lonst[lonst >= 180] -= 360
        xxt, yyt = b(lonst, latst)
        oc_mask = maskoceans(lonst, latst, topo)

        clevs = [-500,0,100,200,300,400,500,600,700,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3500,4000,4500,5000]
        cmap = get_cmap(clevs)
        topo[topo < 0] = -600
        topo[oc_mask.mask] = -600

        cs = b.contourf(xxt, yyt, topo, cmap=cmap, levels=clevs)
        # b.colorbar(cs)
        b.drawcoastlines(linewidth=0.1)


    # show_coast_lines(b, resolution="h")


    b.drawmeridians(np.arange(-180, 180, 30), linewidth=0.5, latmax=90)
    b.drawparallels(np.arange(-90, 120, 30), linewidth=0.5)



    ax = plt.gca()
    for gc in get_gridcell_polygons(rll_obj, rlon, rlat, step=10, bmp=b, linewidth=0.1):
        ax.add_patch(gc)



    ax.add_patch(get_grid_polygon(rll_obj, rlon, rlat, bmp=b, linewidth=0.1, edgecolor="k"))



    plt.savefig("GL_NENA_ortho0.1deg.png", bbox_inches="tight", dpi=400)




if __name__ == '__main__':
    main()