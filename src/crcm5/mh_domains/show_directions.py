from application_properties import main_decorator
from hydrosheds.plot_directions import plot_directions
from crcm5.mh_domains import default_domains


@main_decorator
def main():
    directions_file_path = "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Java/DDM/directions_bc-mh_0.44deg.nc"
    plot_directions(nc_path_to_directions=directions_file_path,
                    grid_config=default_domains.bc_mh_044,
                    shape_path_to_focus_polygons=default_domains.MH_BASINS_PATH)








if __name__ == '__main__':
    main()