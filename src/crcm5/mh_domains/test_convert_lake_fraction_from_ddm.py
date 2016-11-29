from collections import OrderedDict
from netCDF4 import Dataset
from pathlib import Path

from application_properties import main_decorator
from crcm5.mh_domains import default_domains
from crcm5.mh_domains.convert_directions_to_shape_file import MyNdArray, field_name_to_shp_name
from domains.grid_config import GridConfig


@main_decorator
def main():
    directions_dir = Path(".")

    selected_vars = ["lake_fraction"]


    # Create the directory for the shapes
    shp_dir = directions_dir.joinpath("shp_direction_data_shortnames")
    if not shp_dir.is_dir():
        shp_dir.mkdir()

    grid_config_to_dirfile = OrderedDict([
        (default_domains.bc_mh_044, directions_dir.joinpath("directions_bc-mh_0.44deg.nc")),
        # (default_domains.bc_mh_011, directions_dir.joinpath("directions_bc-mh_0.11deg.nc")),
    ])


    for gc, dir_file in grid_config_to_dirfile.items():

        assert isinstance(gc, GridConfig)
        print(gc.get_basemap().proj4string)

        out_shp_filename = "{}.shp".format(dir_file.name[:-3])

        assert isinstance(gc, GridConfig)

        fields_to_add = OrderedDict()
        with Dataset(str(dir_file)) as ds:
            for vname, var in ds.variables.items():

                if not vname in selected_vars:
                    continue

                fields_to_add[vname] = var[:].view(MyNdArray)
                fields_to_add[vname].type_of_shp_field = "float"
                print("{} ranges from {} to {}".format(vname, fields_to_add[vname].min(), fields_to_add[vname].max()))

        # Export the cells to a shapefile
        fields_to_add = {field_name_to_shp_name[k]: v for k, v in fields_to_add.items()}

        gc.export_to_shape_fiona(shp_folder=shp_dir, shp_filename=out_shp_filename, shape_fields=fields_to_add)

if __name__ == '__main__':
    main()
