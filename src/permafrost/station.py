import itertools
import re
from matplotlib import gridspec
from mpl_toolkits.basemap import Basemap
from scipy.spatial.kdtree import KDTree
import application_properties
from . import active_layer_thickness
from . import draw_regions
from util import plot_utils
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np
import codecs
import matplotlib.pyplot as plt
class Station:
    def __init__(self, name = "", lon = None, lat = None, mean_alt_m = None):
        self.name = name
        self.mean_alt_m = mean_alt_m
        self.lon = lon
        self.lat = lat


def get_station_list(path = "data/permafrost_obs/Sites_permafrost.txt",
                     record_length = 20, header_length = 20):
    """
    record_length - number of lines that correspond to each station
    1/20 - Name
    2/20 - Latitude
    3/20 - Longitude
    4-20/20 - data
    """
    f = codecs.open(path, "r", encoding="utf8")
    lines = f.readlines()
    f.close()
    #skip header
    lines = lines[header_length:]
    end_garbage = len(lines) % record_length
    if end_garbage > 0:
        lines = lines[:-end_garbage]


    lines = [x.strip() for x in lines]
    stations = []
    for i in range(0, len(lines), record_length):
        current = lines[i:(i + record_length)]

        current = filter(lambda x: x not in ("", "N.A.", "N.A", "NA"), current)
        current = filter(lambda x: x[0] not in ("<", ">"), current)

        current = [x[1:] if x[0] in ("~",) else x for x in current]
        current = [x[:-1] if x[-1] in ("*", ) else x for x in current]

        deg, min, sec = list(map(float, re.findall("\d+", current[1])))
        lat = deg + min / 60.0 + sec / 3600.0

        deg, min, sec = list(map(float, re.findall("\d+", current[2])))
        lon = deg + min / 60.0 + sec / 3600.0

        vals = list(map( float, current[3:]))
        s = Station(name=current[0], lon=-lon, lat=lat, mean_alt_m=np.mean(vals) / 100.0) #convert cm to m
        stations.append(s)
        print(s.name, ", mean = {0}".format(s.mean_alt_m))
        print("lat = {0}; lon = {1}".format(lat,-lon))
    return stations



def compare_alt():
    #obs
    stations = get_station_list()

    #model data
    alts_model = [ active_layer_thickness.get_alt_for_year(the_year)
        for the_year in range(1991, 2001)
    ]
    alt_mean = np.mean(alts_model, axis=0)
    b, lons2d, lats2d = draw_regions.get_basemap_and_coords()

    permafrost_kinds = draw_regions.get_permafrost_mask(lons2d, lats2d)
    permafrost_kinds_flat = permafrost_kinds.flatten()

    lons2d[lons2d > 180] -= 360

    #find corresponding indices on the model grid
    x, y, z = lat_lon.lon_lat_to_cartesian(lons2d.flatten(), lats2d.flatten())
    kdtree = KDTree(list(zip(x, y, z)))

    alt_mean_flat = alt_mean.flatten()
    h_mod = []
    h_obs = []

    station_lons = []
    station_lats = []
    for the_station in stations:
        x0, y0, z0 = lat_lon.lon_lat_to_cartesian(the_station.lon, the_station.lat)
        d, i = kdtree.query([x0, y0, z0])
        if permafrost_kinds_flat[i] not in (1,2):
            continue
        print(d, i)
        print(the_station.mean_alt_m, alt_mean_flat[i])
        h_mod.append(alt_mean_flat[i])
        h_obs.append(the_station.mean_alt_m)
        station_lons.append(the_station.lon)
        station_lats.append(the_station.lat)

    plot_utils.apply_plot_params(width_pt=None, height_cm=20, width_cm=16, font_size=12)
    fig = plt.figure()
    gs = gridspec.GridSpec(2,1)
    ax = fig.add_subplot(gs[0,0])
    ax.plot(h_obs, h_mod, 'o')
    ax.set_xlabel("Obs.")
    ax.set_ylabel("Mod.")
    upper_lim = max(np.max(h_mod), np.max(h_obs))
    ax.set_xlim(0, upper_lim + 0.1 * upper_lim)
    ax.set_ylim(0, upper_lim + 0.1 * upper_lim)

    ax = fig.add_subplot(gs[1,0])


    min_lon, max_lon = min(station_lons), max(station_lons)
    min_lat, max_lat = min(station_lats), max(station_lats)

    dx = (max_lon - min_lon) * 0.1
    dy = (max_lat - min_lat) * 0.6
    min_lon -= dx
    max_lon += dx
    min_lat -= dy
    max_lat += dy



    lon1 = -97
    lat1 = 47.50
    lon2 = -7
    lat2 = 0
    b_zoom = Basemap(projection="omerc", resolution="l",
               llcrnrlon=min_lon, llcrnrlat=min_lat,
                urcrnrlon=max_lon, urcrnrlat=max_lat,
                lat_1=lat1, lon_1=lon1, lat_2=lat2, lon_2=lon2, no_rot=True
        )
    s_x, s_y = b_zoom(station_lons, station_lats)
    b_zoom.scatter(s_x, s_y, c = "r", ax = ax, marker = "*", s = 30, linewidths = 0.1, zorder = 2)
    b_zoom.drawcoastlines(ax = ax, linewidth = 0.5)
    fig.savefig("pf_validate.png")

    pass


def main():
    compare_alt()
    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print("Hello world")
  