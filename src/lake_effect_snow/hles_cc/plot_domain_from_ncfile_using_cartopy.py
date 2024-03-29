import cartopy
from collections import OrderedDict
from pathlib import Path

from cartopy.mpl.geoaxes import GeoAxes
from matplotlib import cm, colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import KDTree

from application_properties import main_decorator
from crcm5.nemo_vs_hostetler.main_for_lake_effect_snow import get_mask_of_points_near_lakes
from crcm5.nemo_vs_hostetler.plot_GL_domain_and_bathymetry import add_rectangle
from lake_effect_snow.hles_cc import common_params
from lake_effect_snow.hles_cc.plot_cc_2d_all_variables_for_all_periods import get_gl_mask
import xarray
import matplotlib.pyplot as plt

import numpy as np

from rpn.domains.rotated_lat_lon import RotatedLatLon

import logging

from lake_effect_snow.lake_effect_snowfall_entry import get_zone_around_lakes_mask
from util.geo import lat_lon

logger = logging.getLogger(__name__)


def plot_domain_and_interest_region(ax: Axes, topo_nc_file_path: Path, focus_region_lonlat_nc_file: Path=None):
    """
    :param focus_region_lonlat_nc_file: Path to the file containing focus region lons and lats
    :param region_mask_lats: latitudes corresponding to the region mask
    :param region_mask_lons:
    :param ax:
    :param topo_nc_file_path:
    :param region_mask:

    Note: below is the expected structure of the input netcdf file

    $ ncdump -h geophys_452x260_me.nc
    netcdf geophys_452x260_me {
    dimensions:
        x = 452 ;
        y = 260 ;
    variables:
        float ME(x, y) ;
        float lon(x, y) ;
        float lat(x, y) ;
        int proj_params ;
            proj_params:grid_type = "E" ;
            proj_params:lat1 = 0. ;
            proj_params:lon1 = 180. ;
            proj_params:lat2 = 1. ;
            proj_params:lon2 = 276. ;
    }
    """

    # read the model topography from the file
    with xarray.open_dataset(topo_nc_file_path) as topo_ds:
        topo_lons, topo_lats, topo = [topo_ds[k].values for k in ["lon", "lat", "ME"]]

        prj_params = topo_ds["proj_params"]

        rll = RotatedLatLon(lon1=prj_params.lon1, lat1=prj_params.lat1, lon2=prj_params.lon2, lat2=prj_params.lat2)

        rot_pole_cpy = rll.get_cartopy_projection_obj()

    ax.set_visible(False)
    ax = ax.figure.add_axes(ax.get_position().bounds, projection=rot_pole_cpy)
    # ax.coastlines()

    gl_mask = get_gl_mask(topo_nc_file_path)
    # define the ~200km near lake zone
    ktree = KDTree(list(zip(*lat_lon.lon_lat_to_cartesian(topo_lons.flatten(), topo_lats.flatten()))))
    region_mask = get_zone_around_lakes_mask(lons=topo_lons, lats=topo_lats,
                                             lake_mask=gl_mask,
                                             ktree=ktree,
                                             dist_km=common_params.NEAR_GL_HLES_ZONE_SIZE_KM)

    # region_mask = get_mask_of_points_near_lakes(gl_mask, npoints_radius=20)
    topo_lons[topo_lons > 180] -= 360



    xll, yll = rot_pole_cpy.transform_point(topo_lons[0, 0], topo_lats[0, 0], cartopy.crs.PlateCarree())
    xur, yur = rot_pole_cpy.transform_point(topo_lons[-1, -1], topo_lats[-1, -1], cartopy.crs.PlateCarree())
    map_extent = [xll, xur, yll, yur]
    print("Map extent: ", map_extent)

    topo_clevs = [0, 100, 200, 300, 400, 500, 600, 800, 1000, 1200]
    # bn = BoundaryNorm(topo_clevs, len(topo_clevs) - 1)
    cmap = cm.get_cmap("terrain")
    ocean_color = cmap(0.18)
    cmap, norm = colors.from_levels_and_colors(topo_clevs, cmap(np.linspace(0.3, 1, len(topo_clevs) - 1)))

    xyz_coords = rot_pole_cpy.transform_points(cartopy.crs.PlateCarree(), topo_lons, topo_lats)
    xx = xyz_coords[:, :, 0]
    yy = xyz_coords[:, :, 1]

    add_rectangle(ax, xx, yy, margin=20, edge_style="solid", zorder=10, linewidth=0.5)
    add_rectangle(ax, xx, yy, margin=10, edge_style="dashed", zorder=10, linewidth=0.5)

    # plot a rectangle for the focus region
    logging.warning("Focus region file: %s", focus_region_lonlat_nc_file)
    if focus_region_lonlat_nc_file is not None:
        with xarray.open_dataset(focus_region_lonlat_nc_file) as fr:
            focus_lons, focus_lats = fr["lon"].data, fr["lat"].data

            focus_lons[focus_lons > 180] -= 360

            logger.warning(f"focus region coords, lon: %f, ..., %f, lats: %f, ..., %f",
                        focus_lons.min(), focus_lons.max(),
                        focus_lats.min(), focus_lats.max())
            logger.warning(f"focus_reg shape = (%d, %d)", *focus_lons.shape)

            xyz_coords = rot_pole_cpy.transform_points(cartopy.crs.PlateCarree(), focus_lons, focus_lats)
            xxf, yyf = xyz_coords[..., 0], xyz_coords[..., 1]

            add_rectangle(ax, xxf, yyf, edge_style="solid",
                          margin=0,
                          edgecolor="magenta",
                          zorder=10,
                          linewidth=1)

    cs = ax.pcolormesh(topo_lons[:, :], topo_lats[:, :], topo[:, :], transform=cartopy.crs.PlateCarree(),
                     cmap=cmap, norm=norm)

    to_plot = np.ma.masked_where(region_mask < 0.5, region_mask)
    ax.scatter(topo_lons, topo_lats, to_plot * 0.01, c="cyan", transform=cartopy.crs.PlateCarree(),
               alpha=0.5)

    # Add geographic features
    line_color = "k"
    ax.add_feature(common_params.LAKES_50m, facecolor=cartopy.feature.COLORS["water"], edgecolor=line_color, linewidth=0.5)
    ax.add_feature(common_params.OCEAN_50m, facecolor=cartopy.feature.COLORS["water"], edgecolor=line_color, linewidth=0.5)
    ax.add_feature(common_params.COASTLINE_50m, facecolor="none", edgecolor=line_color, linewidth=0.5)
    ax.add_feature(common_params.RIVERS_50m, facecolor="none", edgecolor=line_color, linewidth=0.5)
    ax.set_extent(map_extent, crs=rot_pole_cpy)

    # improve colorbar
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_vertical(size="5%", pad=0.1, axes_class=plt.Axes, pack_start=True)
    ax.figure.add_axes(ax_cb)
    cb = plt.colorbar(cs, cax=ax_cb, orientation="horizontal")
    cb.ax.set_xticklabels(topo_clevs, rotation=45)
    return ax

@main_decorator
def test():
    data_root = common_params.data_root

    label_to_datapath = OrderedDict([
        (common_params.crcm_nemo_cur_label, data_root / "lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_1989-2010_1989-2010"),
        (common_params.crcm_nemo_fut_label, data_root / "lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_2079-2100_2079-2100"),
    ])


    topo_ncfile = data_root / "geophys_452x260_me.nc"


    plt.figure()
    ax = plt.gca()
    plot_domain_and_interest_region(ax, topo_ncfile)



    plt.show()



if __name__ == '__main__':
    test()