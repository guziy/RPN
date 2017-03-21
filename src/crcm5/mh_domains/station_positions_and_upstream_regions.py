from pathlib import Path

from matplotlib.axes import Axes, GridSpec
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.basemap import Basemap

from application_properties import main_decorator
from crcm5.mh_domains import constants
from crcm5.mh_domains import default_domains
from crcm5.mh_domains import stfl_stations
from data.cell_manager import CellManager
from domains.grid_config import GridConfig

from netCDF4 import Dataset

from util import plot_utils

import matplotlib.pyplot as plt
from collections import OrderedDict



img_folder = Path("mh/engage_report/station_positions_and_upstream/")

if not img_folder.exists():
    img_folder.mkdir(parents=True)


def plot_station_positions(directions_file_path: Path, stations: list, ax: Axes, grid_config: GridConfig=default_domains.bc_mh_044,
                           save_upstream_boundaries_to_shp=False):


    with Dataset(str(directions_file_path)) as ds:
        flow_dirs = ds.variables["flow_direction_value"][:]
        flow_acc_area = ds.variables["accumulation_area"][:]
        lons_2d, lats_2d = [ds.variables[k][:] for k in ["lon", "lat"]]



    basemap, reg_of_interest = grid_config.get_basemap_using_shape_with_polygons_of_interest(lons_2d, lats_2d,
                                                                                             shp_path=default_domains.MH_BASINS_PATH,
                                                                                             resolution="i")


    cell_manager = CellManager(flow_dirs, lons2d=lons_2d, lats2d=lats_2d, accumulation_area_km2=flow_acc_area)
    station_to_model_point = cell_manager.get_model_points_for_stations(station_list=stations, nneighbours=8)

    #####
    xx, yy = basemap(lons_2d, lats_2d)
    upstream_edges = cell_manager.get_upstream_polygons_for_points(
        model_point_list=list(station_to_model_point.values()), xx=xx, yy=yy)

    upstream_edges_latlon = cell_manager.get_upstream_polygons_for_points(
        model_point_list=list(station_to_model_point.values()), xx=lons_2d, yy=lats_2d)




    plot_utils.draw_upstream_area_bounds(ax, upstream_edges=upstream_edges, color="r", linewidth=0.6)

    if save_upstream_boundaries_to_shp:
        plot_utils.save_to_shape_file(upstream_edges_latlon, folder_path="mh/engage_report/upstream_stations_areas/mh_{}".format(grid_config.dx), in_proj=None)


    basemap.drawrivers(linewidth=0.2)
    basemap.drawstates(linewidth=0.1)
    basemap.drawcountries(linewidth=0.1)
    basemap.drawcoastlines(linewidth=0.2)


    pos_ids, lons_pos, lats_pos = [], [], []
    pos_labels = []
    legend_lines = []
    for i, (s, mp) in enumerate(sorted(station_to_model_point.items(), key=lambda p: p[0].latitude, reverse=True), start=1):
        pos_ids.append(s.id)
        pos_labels.append(i)
        lons_pos.append(mp.longitude)
        lats_pos.append(mp.latitude)

        legend_lines.append("{}: {}".format(i, s.id))

    xm, ym = basemap(lons_pos, lats_pos)
    ax.scatter(xm, ym, c="g", s=20)
    for txt, x1, y1, pos_label in zip(pos_ids, xm, ym, pos_labels):
        ax.annotate(pos_label, xy=(x1, y1))



    at = AnchoredText("\n".join(legend_lines), prop=dict(size=8), frameon=True, loc=1)

    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)



@main_decorator
def main():

    grid_config_to_dir_file = OrderedDict([
        #(default_domains.bc_mh_044, Path("/RESCUE/skynet3_rech1/huziy/directions_for_ManitobaHydro/directions_mh_0.44deg.nc")),
        (default_domains.bc_mh_022, Path("/RESCUE/skynet3_rech1/huziy/directions_for_ManitobaHydro/directions_mh_0.22deg.nc")),
        # (default_domains.bc_mh_011, Path("/RESCUE/skynet3_rech1/huziy/directions_for_ManitobaHydro/directions_bc-mh_0.11deg_new.nc")),
    ])


    stations = stfl_stations.load_stations_from_csv(selected_ids=None)

    print(stations)

    gs = GridSpec(1, len(grid_config_to_dir_file))

    plot_utils.apply_plot_params(width_cm=25, height_cm=20, font_size=8)
    fig = plt.figure()

    for col, (grid_config, dir_path) in enumerate(grid_config_to_dir_file.items()):
        ax = fig.add_subplot(gs[0, col])
        plot_station_positions(directions_file_path=dir_path, stations=stations, ax=ax, grid_config=grid_config)



    img_file = img_folder / "{}_{}.png".format("mh", "_".join([str(gc.dx) for gc in grid_config_to_dir_file]))
    fig.savefig(str(img_file), bbox_inches="tight", dpi=300)



if __name__ == '__main__':
    main()