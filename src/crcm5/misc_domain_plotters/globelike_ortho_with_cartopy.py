from mpl_toolkits.basemap import maskoceans
from rpn.domains.rotated_lat_lon import RotatedLatLon
from rpn.rpn import RPN

from util import plot_utils
import matplotlib.pyplot as plt

from crcm5.misc_domain_plotters.globelike_ortho import get_gridcell_polygons, get_cmap, get_etopo

import cartopy
import cartopy.crs as ccrs

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



    crs_map = ccrs.Orthographic(central_latitude=50, central_longitude=-80)
    ax = plt.axes(projection=crs_map)
    # b = Basemap(projection="ortho", lat_0=50, lon_0=-80)

    #
    # b.drawcoastlines(linewidth=0.5)
    # b.shadedrelief()
    # b.warpimage()

    # b.drawmapboundary(fill_color="#94e3e8")

    # aximg = b.etopo()
    # assert isinstance(aximg, AxesImage)
    # aximg.set_cmap("bwr")

    topo, lonst, latst = get_etopo()

    lonst[lonst >= 180] -= 360


    oc_mask = maskoceans(lonst, latst, topo)


    clevs = [-500,0,100,200,300,400,500,600,700,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3500,4000,4500,5000]
    cmap = get_cmap(clevs)
    topo[topo < 0] = -600
    topo[oc_mask.mask] = -600

    cs = ax.contourf(lonst, latst, topo, cmap=cmap, levels=clevs, transform=ccrs.Geodetic())
    ax.colorbar(cs)


    # show_coast_lines(b, resolution="h")


    ax = plt.gca()
    for gc in get_gridcell_polygons(rll_obj, rlon, rlat, step=20):
        ax.add_patch(gc)



    plt.savefig("carrtopy_GL_NENA_ortho0.1deg.png", bbox_inches="tight", dpi=400)

if __name__ == '__main__':
    main()