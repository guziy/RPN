from matplotlib import cm
import matplotlib
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.patches import Rectangle
from crcm5.model_point import ModelPoint
from data import cehq_station
from data.cehq_station import Station
from data.cell_manager import CellManager
import common_plot_params as cpp

__author__ = 'huziy'
import numpy as np


def plot_positions_of_station_list(ax, stations, model_points, basemap, cell_manager):
    from util import direction_and_value

    delta = 1.0 / float(len(model_points))
    darkness = 0

    cmap = cm.get_cmap(name="spectral", lut = len(model_points))


    artists = []
    labels = []

    if None not in stations:
        st_to_mp = dict(zip(stations, model_points))
        stlist_sorted = list(sorted(stations, key=lambda st: st.drainage_km2, reverse=True))
        mplist_sorted = [st_to_mp[s] for s in stlist_sorted]
    else:
        mp_to_st = dict(zip(model_points, stations))
        mplist_sorted = list(sorted(model_points, key=lambda mp: mp.latitude, reverse=True))
        stlist_sorted = [mp_to_st[mp] for mp in mplist_sorted]


    for the_station, the_model_point in zip(stlist_sorted, mplist_sorted):
        assert the_station is None or isinstance(the_station, Station)
        assert isinstance(the_model_point, ModelPoint)

        assert isinstance(cell_manager, CellManager)

        x, y = basemap(cell_manager.lons2d, cell_manager.lats2d)


        #plot the arrows for upstream cells
        ups_mask = cell_manager.get_mask_of_cells_connected_with_by_indices(the_model_point.ix, the_model_point.jy)

        x1d_start = x[ups_mask == 1]
        y1d_start = y[ups_mask == 1]
        fld1d = cell_manager.flow_directions[ups_mask == 1]
        i_upstream, j_upstream = np.where(ups_mask == 1)

        if the_station is not None:
            x_station, y_station = basemap(the_station.longitude, the_station.latitude)
            point_info = "{0}".format(the_station.id)
        else:
            x_station, y_station = basemap(the_model_point.longitude, the_model_point.latitude)
            point_info = "({0}, {1})".format(the_model_point.ix, the_model_point.jy)

        labels.append(point_info)

        basemap.scatter(x_station, y_station, c="r", s=10, ax=ax, linewidths=0.5, zorder=2)

        ishift, jshift = direction_and_value.flowdir_values_to_shift(fld1d)

        sub_i_upstream_next = i_upstream + ishift
        sub_j_upstream_next = j_upstream + jshift

        u = x[sub_i_upstream_next, sub_j_upstream_next] - x1d_start
        v = y[sub_i_upstream_next, sub_j_upstream_next] - y1d_start

        u2d = np.ma.masked_all_like(x)
        v2d = np.ma.masked_all_like(y)

        u2d[i_upstream, j_upstream] = u
        v2d[i_upstream, j_upstream] = v

        #basemap.quiver(x, y, u2d, v2d, angles="xy", scale_units="xy", scale=1, ax=ax)

        img = basemap.pcolormesh(x, y, np.ma.masked_where(ups_mask < 0.5, ups_mask) * darkness,
                                 cmap= cmap,
                                 ax=ax, vmax=1, vmin=0)


        p = Rectangle((0, 0), 1, 1, fc=cmap(darkness))
        artists.append(p)


        #ax.text(x_station, y_station, point_info, bbox=dict(facecolor="white"))
        darkness += delta

    ax.legend(artists, labels, handleheight = 2, ncol = 3, loc=4)
    basemap.drawcoastlines(linewidth=cpp.COASTLINE_WIDTH)
    basemap.drawrivers(cpp.COASTLINE_WIDTH)


def main():
    model_file = ""

    selected_ids = [

    ]

    #Get the list of stations to plot
    stations = cehq_station.read_station_data(
        start_date=None, end_date=None, selected_ids=selected_ids
    )


if __name__ == "__main__":
    main()