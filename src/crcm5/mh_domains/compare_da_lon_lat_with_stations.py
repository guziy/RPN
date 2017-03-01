from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter, LogLocator
from netCDF4 import Dataset

from application_properties import main_decorator
from crcm5.mh_domains import stfl_stations
from crcm5.model_point import ModelPoint
from data.cehq_station import Station
from data.cell_manager import CellManager
import matplotlib.pyplot as plt

from util import plot_utils

img_folder = Path("mh/engage_report/station_data_plots")


@main_decorator
def main(directions_file_path: Path):
    """
    compare drainage areas, longitudes and latitudes from the stations and model
    """
    stations = stfl_stations.load_stations_from_csv()
    lake_fraction=None

    with Dataset(str(directions_file_path)) as ds:
        flow_dirs = ds.variables["flow_direction_value"][:]
        flow_acc_area = ds.variables["accumulation_area"][:]
        lons_2d, lats_2d = [ds.variables[k][:] for k in ["lon", "lat"]]

        # lake_fraction = ds.variables["lake_fraction"][:]


    cell_manager = CellManager(flow_dirs, lons2d=lons_2d, lats2d=lats_2d, accumulation_area_km2=flow_acc_area)

    station_to_mod_point = cell_manager.get_model_points_for_stations(station_list=stations, lake_fraction=lake_fraction,
                                                                      nneighbours=8)


    lons_m, lats_m, da_m = [], [], []
    lons_o, lats_o, da_o = [], [], []


    for s, mp in station_to_mod_point.items():
        assert isinstance(s, Station)
        assert isinstance(mp, ModelPoint)

        # obs
        lons_o.append(s.longitude if s.longitude < 180 else s.longitude - 360)
        lats_o.append(s.latitude)
        da_o.append(s.drainage_km2)

        # model
        lons_m.append(mp.longitude if mp.longitude < 180 else mp.longitude - 360)
        lats_m.append(mp.latitude)
        da_m.append(mp.accumulation_area)


        print("m  | s ({})".format(s.id))
        print("{} | {}".format(mp.longitude, s.longitude))
        print("{} | {}".format(mp.latitude, s.latitude))
        print("{} | {}".format(mp.accumulation_area, s.drainage_km2))


    axes_list = []
    plot_utils.apply_plot_params(width_cm=25, height_cm=10, font_size=8)
    fig = plt.figure()
    gs = GridSpec(1, 3)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("Longitude")
    ax.scatter(lons_o, lons_m)
    axes_list.append(ax)
    ax.set_ylabel("Model")


    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Latitude")
    ax.scatter(lats_o, lats_m)
    axes_list.append(ax)
    ax.set_xlabel("Obs")

    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("Drainage area (km$^2$)")
    ax.scatter(da_o, da_m)
    sf = ScalarFormatter(useMathText=True)
    sf.set_powerlimits((-2, 3))

    ax.set_xscale("log")
    ax.set_yscale("log")


    axes_list.append(ax)



    # plot the 1-1 line
    for ax in axes_list:
        assert isinstance(ax, Axes)

        ax.plot(ax.get_xlim(), ax.get_xlim(), "--", color="gray")
        ax.grid()


    img_file = img_folder.joinpath("lon_lat_da_scatter_{}_{}.png".format(directions_file_path.name,
                                                                         "-".join(sorted(s.id for s in station_to_mod_point))))
    fig.savefig(str(img_file), bbox_inches="tight")


if __name__ == '__main__':

    main(directions_file_path=Path("/RESCUE/skynet3_rech1/huziy/directions_for_ManitobaHydro/directions_mh_0.11deg.nc"))
    main(directions_file_path=Path("/RESCUE/skynet3_rech1/huziy/directions_for_ManitobaHydro/directions_mh_0.22deg.nc"))
    main(directions_file_path=Path("/RESCUE/skynet3_rech1/huziy/directions_for_ManitobaHydro/directions_mh_0.44deg.nc"))