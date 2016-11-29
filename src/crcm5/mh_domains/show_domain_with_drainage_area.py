import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from pathlib import Path

from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset

from application_properties import main_decorator
from crcm5.mh_domains import default_domains
from crcm5.mh_domains.utils.region_and_mask import get_rectangular_region_from_mask_and_margin

from domains.grid_config import GridConfig

img_folder = "mh"


def show_domain(grid_config, halo=None, blending=None, draw_rivers=True, grdc_basins_of_interest=None,
                directions_file=None, imgfile_prefix="bc-mh", include_buffer=True, ax=None, basin_border_width=1.5,
                path_to_shape_with_focus_polygons=None, nc_varname_to_show="accumulation_area"):
    assert isinstance(grid_config, GridConfig)

    is_subplot = ax is not None

    fig = None
    if not is_subplot:
        fig = plt.figure()
        ax = plt.gca()

    halo = 10 if halo is None else halo
    blending = 10 if blending is None else blending

    if include_buffer:
        bmp = grid_config.get_basemap(resolution="l")
    else:
        bmp = grid_config.get_basemap_for_free_zone(resolution="l")

    margin = halo + blending

    if directions_file is not None:
        with Dataset(directions_file) as ds:
            lons2d, lats2d, faa = [ds.variables[k][:] for k in ["lon", "lat", nc_varname_to_show]]

        # Focus over the selected watersheds
        data_mask = None
        mask_margin = 5

        if path_to_shape_with_focus_polygons is not None:
            bmp, data_mask = grid_config.get_basemap_using_shape_with_polygons_of_interest(
                lons2d[margin:-margin, margin:-margin],
                lats2d[margin:-margin, margin:-margin],
                shp_path=path_to_shape_with_focus_polygons,
                mask_margin=mask_margin)

            bmp.readshapefile(path_to_shape_with_focus_polygons[:-4], "basins", linewidth=2, color="m")

        xxx, yyy = bmp(lons2d[margin:-margin, margin:-margin], lats2d[margin:-margin, margin:-margin])

        data = faa[margin:-margin, margin:-margin]
        if data_mask is not None:

            # subset the data for plotting with imshow (not required for contourf)
            imin, imax, jmin, jmax = get_rectangular_region_from_mask_and_margin(data_mask > 0.5, margin_points=mask_margin)
            data = np.ma.masked_where(data_mask < 0.5, data)
            data = data[imin:imax + 1, jmin:jmax + 1]

        if nc_varname_to_show == "lake_fraction":
            im = bmp.imshow(data.T, cmap="bone_r", interpolation="nearest")
        else:
            im = bmp.contourf(xxx, yyy, data, cmap="bone_r", norm=LogNorm())

        bmp.colorbar(im)

    # bmp.readshapefile(default_domains.MH_BASINS_PATH[:-4], "basin", color="m", linewidth=basin_border_width)

    if grdc_basins_of_interest is not None:
        # Select which basins to show
        bmp.readshapefile(default_domains.GRDC_BASINS_PATH[:-4], "basin", drawbounds=False)

        patches = []

        for info, shape in zip(bmp.basin_info, bmp.basin):
            if info["BASIN_ID"] in grdc_basins_of_interest:
                patches.append(Polygon(np.array(shape), True))

        ax.add_collection(
            PatchCollection(patches, facecolor='none', edgecolor='r', linewidths=basin_border_width, zorder=2))

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

    nx = grid_config.ni if include_buffer else (grid_config.ni - 2 * grid_config.halo - 2 * grid_config.blendig)
    ny = grid_config.nj if include_buffer else (grid_config.nj - 2 * grid_config.halo - 2 * grid_config.blendig)

    ax.set_title(r"${}".format(nx) + r"\times" + "{}$ grid cells, $\Delta x$ = {}$^\circ$".format(ny, grid_config.dx))

    if not is_subplot:
        fig.savefig(str(p.joinpath("{}_dx{}.png".format(imgfile_prefix, grid_config.dx))), bbox_inches="tight",
                    transparent=True, dpi=600)

        plt.close(fig)


def show_all_domains():

    transparent = True
    from util import plot_utils
    plot_utils.apply_plot_params(width_cm=17, height_cm=6.5, font_size=6)

    img_folder_path = Path(img_folder)

    fig1 = plt.figure()
    gs = GridSpec(1, 2, wspace=0.0)

    ax = fig1.add_subplot(gs[0, 0])
    show_domain(default_domains.bc_mh_011,
                grdc_basins_of_interest=default_domains.GRDC_basins_of_interest,
                draw_rivers=False,
                directions_file="/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Java/DDM/directions_bc-mh_0.11deg.nc",
                include_buffer=False, ax=ax)

    ax = fig1.add_subplot(gs[0, 1])
    show_domain(default_domains.bc_mh_044,
                grdc_basins_of_interest=default_domains.GRDC_basins_of_interest,
                draw_rivers=False,
                directions_file="/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Java/DDM/directions_bc-mh_0.44deg.nc",
                include_buffer=False, ax=ax)

    fig1.savefig(str(img_folder_path.joinpath("bc_mh_011_and_044.png")), bbox_inches="tight", transparent=transparent,
                 dpi=600)
    plt.close(fig1)

    fig2 = plt.figure()
    gs = GridSpec(1, 2, wspace=0.05)

    ax = fig2.add_subplot(gs[0, 0])

    show_domain(default_domains.gc_panarctic_05,
                grdc_basins_of_interest=default_domains.GRDC_basins_of_interest_Panarctic,
                draw_rivers=False,
                directions_file="/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Java/DDM/directions_arctic_0.5deg_Bernardo.nc",
                imgfile_prefix="PanArctic_0.5deg" + "_transparent" if transparent else "",
                include_buffer=False, ax=ax)

    ax = fig2.add_subplot(gs[0, 1])
    show_domain(default_domains.gc_cordex_NA_044,
                grdc_basins_of_interest=default_domains.GRDC_basins_of_interest_NA,
                draw_rivers=False,
                directions_file="/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Java/DDM/directions_na_0.44deg_CORDEX.nc",
                imgfile_prefix="CORDEX_NA_0.44deg" + "_transparent" if transparent else "",
                include_buffer=False, ax=ax)

    fig2.savefig(str(img_folder_path.joinpath("NA_and_Arctic_044.png")), bbox_inches="tight", transparent=transparent,
                 dpi=600)
    plt.close(fig2)

    # Great Lakes largest domain
    plot_utils.apply_plot_params(width_cm=6.5, height_cm=6.5, font_size=6)
    show_domain(default_domains.gc_GL_and_NENA_01,
                grdc_basins_of_interest=default_domains.GRDC_basins_GL,
                draw_rivers=False,
                imgfile_prefix="GL_NENA_01",
                directions_file="/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Java/DDM/directions_440x260_GL+NENA_0.1deg.nc",
                include_buffer=False)

@main_decorator
def test_lake_fraction_calculation():
    show_domain(default_domains.bc_mh_044, include_buffer=False, imgfile_prefix="mh-focus-zone-lkfr_test",
                path_to_shape_with_focus_polygons=default_domains.MH_BASINS_PATH,
                directions_file="/Users/san/Java/ddm/directions_bc-mh_0.44deg.nc", nc_varname_to_show="lake_fraction")



@main_decorator
def main():
    # default_domains.bc_mh_044.export_to_shape_ogr(shp_folder="mh/shapes/", shp_filename="dx_044deg")
    # default_domains.bc_mh_044.export_to_shape_native_grid(shp_folder="mh/shapes/", shp_filename="dx_044deg")
    #
    # default_domains.bc_mh_011.decrease_resolution_keep_free_domain_same(2).export_to_shape_ogr(shp_folder="mh/shapes/", shp_filename="dx_022deg")
    #
    # default_domains.bc_mh_011.export_to_shape_ogr(shp_folder="mh/shapes/", shp_filename="dx_011deg")

    # show_domain(default_domains.bc_mh_011.decrease_resolution_keep_free_domain_same(2),
    #             grdc_basins_of_interest=default_domains.GRDC_basins_of_interest,
    #             draw_rivers=False,
    #             directions_file=None,
    #             imgfile_prefix="bc-mh_0.22deg",
    #             include_buffer=False)

    show_all_domains()


    # show_domain(default_domains.bc_mh_044, include_buffer=False, imgfile_prefix="mh-focus-zone-lkfr",
    #             path_to_shape_with_focus_polygons=default_domains.MH_BASINS_PATH,
    #             directions_file="/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Java/DDM/directions_bc-mh_0.44deg.nc")
    #
    # show_domain(default_domains.bc_mh_022, include_buffer=False, imgfile_prefix="mh-focus-zone-lkfr",
    #             path_to_shape_with_focus_polygons=default_domains.MH_BASINS_PATH,
    #             directions_file="/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Java/DDM/directions_bc-mh_0.22deg.nc")
    #
    # show_domain(default_domains.bc_mh_011, include_buffer=False, imgfile_prefix="mh-focus-zone-lkfr",
    #             path_to_shape_with_focus_polygons=default_domains.MH_BASINS_PATH,
    #             directions_file="/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Java/DDM/directions_bc-mh_0.11deg.nc")


if __name__ == '__main__':
    # main()
    test_lake_fraction_calculation()
