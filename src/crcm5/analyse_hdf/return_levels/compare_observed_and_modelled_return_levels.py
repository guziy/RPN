from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from data import cehq_station
from crcm5 import infovar
from data.cehq_station import Station
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis
from data.cell_manager import CellManager
from crcm5.analyse_hdf.return_levels import extreme_commons

__author__ = 'huziy'


def main(hdf_folder="/home/huziy/skynet3_rech1/hdf_store", start_year=1980, end_year=2010):
    # Station ids to get from the CEHQ database
    ids_with_lakes_upstream = [
        "104001", "093806", "093801", "081002", "081007", "080718"
    ]

    selected_ids = ids_with_lakes_upstream

    filedir = Path(hdf_folder)
    sim_name_to_file_path = OrderedDict([
        ("CRCM5-LI", filedir.joinpath("quebec_0.1_crcm5-hcd-r.hdf5").as_posix()),

        ("CRCM5-L", filedir.joinpath("quebec_0.1_crcm5-hcd-rl.hdf5").as_posix()),

        ("CRCM5-NL", filedir.joinpath("quebec_0.1_crcm5-r.hdf5").as_posix()),
    ])


    # Get the list of stations to do the comparison with
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    stations = cehq_station.read_station_data(
        start_date=start_date, end_date=end_date, selected_ids=selected_ids
    )

    # Get geophysical fields from one of the model simulations
    path0 = list(sim_name_to_file_path.values())[0]
    lons2d, lats2d, basemap = analysis.get_basemap_from_hdf(file_path=path0)
    flow_directions = analysis.get_array_from_file(path=path0, var_name=infovar.HDF_FLOW_DIRECTIONS_NAME)
    lake_fraction = analysis.get_array_from_file(path=path0, var_name=infovar.HDF_LAKE_FRACTION_NAME)

    accumulation_area_km2 = analysis.get_array_from_file(path=path0, var_name=infovar.HDF_ACCUMULATION_AREA_NAME)
    area_m2 = analysis.get_array_from_file(path=path0, var_name=infovar.HDF_CELL_AREA_NAME_M2)

    # Try to read cell areas im meters if it is not Ok then try in km2
    if area_m2 is not None:
        cell_area_km2 = area_m2 * 1.0e-6
    else:
        cell_area_km2 = analysis.get_array_from_file(path=path0, var_name=infovar.HDF_CELL_AREA_NAME_KM2)

    # Create a cell manager if it is not provided
    cell_manager = CellManager(flow_directions, accumulation_area_km2=accumulation_area_km2,
                               lons2d=lons2d, lats2d=lats2d)

    # Get the list of the corresponding model points
    station_to_modelpoint = cell_manager.get_model_points_for_stations(
        station_list=stations,
        lake_fraction=lake_fraction,
        drainaige_area_reldiff_limit=0.1)


    print("Initial list of stations:")
    for s in stations:
        print("{0}".format(s))
        assert isinstance(s, Station)

        print(len([y for y in s.get_list_of_complete_years() if start_year <= y <= end_year]))
        print(s.get_list_of_complete_years())

        extreme_commons.get_annual_extrema(ts_times=s.dates, ts_vals=s.values)










if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()

    main()
