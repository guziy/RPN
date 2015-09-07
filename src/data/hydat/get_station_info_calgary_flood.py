from mpl_toolkits.basemap import Basemap
from crcm5.analyse_hdf.compare_streamflow_with_obs import _plot_station_position
from data import cehq_station
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from data.cell_manager import CellManager

__author__ = 'san'

PROVINCE = "AB"  # Alberta


def get_station_objects(db_path="/home/san/Downloads/Hydat.sqlite"):
    selected_ids = ["05BB001", "05BH005", "05BH004", "05BM004"]

    stations = cehq_station.load_from_hydat_db(natural=None, province=PROVINCE, path=db_path, selected_ids=selected_ids)
    for s in stations:
        print(s)
    return stations


def get_cell_manager_from_directions_file(path="/home/san/Downloads/directions_WestCaUs_dx0.11deg.nc", margin=20):
    ds = Dataset(path)


    dirs = ds.variables["flow_direction_value"]
    lons = ds.variables["lon"]
    lats = ds.variables["lat"]
    acc_area = ds.variables["accumulation_area"]

    nc_vars = [dirs, lons, lats, acc_area]
    nc_data = []
    for i, v in enumerate(nc_vars):
        if margin is not None and margin > 0:
            nc_data.append(v[margin:-margin, margin:-margin])
        else:
            nc_data.append(v[:])

        print(type(nc_vars[i]))

    return CellManager(nc_data[0], lons2d=nc_data[1], lats2d=nc_data[2], accumulation_area_km2=nc_data[3])



def main():
    import application_properties
    application_properties.set_current_directory()
    stations = get_station_objects()
    """:type : list[data.cehq_station.Station] """


    fig = plt.figure()
    ax = fig.add_subplot(111)
    bmp = Basemap()


    ds = Dataset("calg_flood_stations_upstream_masks.nc", "w")
    """
    :type : netCDF4.Dataset
    """
    cell_manager = get_cell_manager_from_directions_file()


    station_to_modelpoint = cell_manager.get_model_points_for_stations(station_list=stations, drainaige_area_reldiff_limit=0.15)


    for i, s in enumerate(stations):
        the_mask = _plot_station_position(ax=ax, the_station=s, basemap=bmp, cell_manager=cell_manager,
                                          the_model_point=station_to_modelpoint[s])

        # save to file
        if i == 0:
            # create dimensions
            ds.createDimension("x", the_mask.shape[0])
            ds.createDimension("y", the_mask.shape[1])

        v = ds.createVariable("station_{}".format(s.id), "i4", ("x", "y"))
        v[:] = the_mask

    ds.close()




if __name__ == '__main__':
    main()