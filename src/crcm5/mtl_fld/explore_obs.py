from datetime import datetime

from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap

from data import cehq_station
from data.cehq_station import Station



import matplotlib.pyplot as plt

from util import plot_utils

from crcm5.mtl_fld import commons


def plot_station_positions(stations, img_file="stfl_station_positions.png",
                           shp_paths=None, min_lon=-180., min_lat=-90., max_lon=180., max_lat=90.):
    lons = []
    lats = []
    st_ids = []

    for s in stations:
        assert isinstance(s, Station)

        lons.append(s.longitude)
        lats.append(s.latitude)
        st_ids.append(s.id)





    b = Basemap(llcrnrlon=min_lon, llcrnrlat=min_lat, urcrnrlon=max_lon, urcrnrlat=max_lat, resolution="h", area_thresh=0)

    x, y = b(lons, lats)


    fig = plt.figure()
    nrows = 1
    ncols = 1
    gs = GridSpec(nrows, ncols)

    ax = fig.add_subplot(gs[0, 0])
    b.scatter(x, y, c="b", s=20)
    b.drawcoastlines()
    b.drawrivers(color="b")


    if shp_paths is not None:
        for shp_path in shp_paths:
            b.readshapefile(shp_path[:-4], "basin_info", color="m", linewidth=0.5)



    for xi, yi, st_id in zip(x, y, st_ids):
        ax.annotate(st_id, (xi, yi))

    fig.savefig(img_file, bbox_inches="tight", dpi=400)



def main():

    data_dir = "data/cehq_data/MontrealFlood2017_station_data/streamflow/daily"


    basin_shapes = [
        "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/data/shp/mtl_flood_2017_basins/02JKL_SDA_Ottawa.shp",
        #"/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/data/shp/mtl_flood_2017_basins/02MOP_SDA_St_Lawrence.shp",
        # "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/data/shp/mtl_flood_2017_basins/02N_SDA_St_Maurice.shp"
    ]


    selected_ids = [""]

    excluded_ids = ["120201"]

    start_date = datetime(1980, 1, 1)
    end_date = datetime(2017, 6, 1)

    stations = cehq_station.read_station_data(start_date=start_date, end_date=end_date, folder=data_dir,
                                              min_number_of_complete_years=0, only_natural=None)

    for s in stations:

        assert isinstance(s, Station)

        print(s)
        print(s.get_list_of_complete_years())




    stations = [s for s in stations if s.id not in excluded_ids]







    plot_utils.apply_plot_params(font_size=10)
    img_file = "stfl_station_positions.png"
    img_file = commons.img_folder / img_file

    plot_station_positions(stations, img_file=str(img_file), shp_paths=basin_shapes,
                           min_lon=-81.5, max_lon=-70, min_lat=43.5, max_lat=49)


    pass


if __name__ == '__main__':
    main()