from collections import OrderedDict
from pathlib import Path

from crcm5.mh_domains import default_domains
from domains.grid_config import GridConfig
from netCDF4 import Dataset

import numpy as np


class MyNdArray(np.ndarray):
    pass


field_name_to_shp_name = {
    "flow_direction_value": "fldir",
    "lake_outlet": "lkout",
    "cell_area": "carea",
    "lake_fraction": "lkfr",
    "lon": "lon",
    "lat": "lat",
    "accumulation_area": "accarea",
    "channel_length": "chalen",
    "slope": "slope",
    "sand": "sand",
    "clay": "clay",
    "dpth_bdrck": "bdrckdp"

}


def main():
    # directions_dir = Path("/HOME/huziy/skynet3_rech1/directions_for_ManitobaHydro")
    directions_dir = Path("/HOME/huziy/skynet3_rech1/directions_for_ManitobaHydro/20170310/dist")

    # Create the directory for the shapes
    shp_dir = directions_dir.joinpath("shp_direction_data_shortnames")
    if not shp_dir.is_dir():
        shp_dir.mkdir()


    grid_config_to_dirfile = OrderedDict([
        # (default_domains.bc_mh_044, directions_dir.joinpath("directions_mh_0.44deg.nc")),
        # (default_domains.bc_mh_011, directions_dir.joinpath("directions_mh_0.11deg.nc")),
        # (default_domains.bc_mh_022, directions_dir.joinpath("directions_mh_0.22deg.nc")),
        # (default_domains.bc_mh_011, directions_dir.joinpath("directions_bc-mh_0.11deg_new.nc")),
        # (default_domains.bc_mh_022, directions_dir.joinpath("directions_bc-mh_0.22deg_new.nc")),
        # (default_domains.bc_mh_044, directions_dir.joinpath("directions_bc-mh_0.44deg_new.nc")),

        (default_domains.bc_mh_011, directions_dir.joinpath("all_mh_0.11.nc")),
        (default_domains.bc_mh_022, directions_dir.joinpath("all_mh_0.22.nc")),
        (default_domains.bc_mh_044, directions_dir.joinpath("all_mh_0.44.nc")),

    ])

    for gc, dir_file in grid_config_to_dirfile.items():

        out_shp_filename = "{}.shp".format(dir_file.name[:-3])

        assert isinstance(gc, GridConfig)


        fields_to_add = OrderedDict()
        with Dataset(str(dir_file)) as ds:
            for vname, var in ds.variables.items():

                if vname not in field_name_to_shp_name:
                    continue

                fields_to_add[vname] = var[:].view(MyNdArray)
                fields_to_add[vname].type_of_shp_field = "float"
                print("{} ranges from {} to {}".format(vname, fields_to_add[vname].min(), fields_to_add[vname].max()))

        # Export the cells to a shapefile
        fields_to_add = {field_name_to_shp_name[k.lower()]: v for k, v in fields_to_add.items()}

        gc.export_to_shape_fiona(shp_folder=shp_dir, shp_filename=out_shp_filename, shape_fields=fields_to_add)


if __name__ == '__main__':
    main()
