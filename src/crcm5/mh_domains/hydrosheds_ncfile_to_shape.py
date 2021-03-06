from collections import OrderedDict
from pathlib import Path

from fiona.crs import from_epsg, from_string
import fiona
from shapely.geometry import Polygon
from shapely.geometry import mapping

from application_properties import main_decorator
from netCDF4 import Dataset
import numpy as np

@main_decorator
def main():

    """
    assumes lon and lat variables are 1d and are inside of the netcdf file
    """

    # flow accumulation index
    # in_path = "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Java/DDM/data/netcdf/NA/na_acc_30s.nc"
    in_path = "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/hydrosheds_corrected_mh/na_acc_30s.nc"
    nc_varname = "flow_accumulation"
    shp_field_name = "acc"
    out_dir = "mh/engage_report/hydrosheds_acc_index_30s_test"

    # flow direction
    # in_path = "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Java/DDM/data/netcdf/NA/na_dir_30s.nc"
    # in_path = "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/hydrosheds_corrected_mh/na_dir_30s.nc"
    # nc_varname = "flow_direction"
    # shp_field_name = "dir"
    # out_dir = "mh/engage_report/hydrosheds_dir_30s_test"

    # read the input data
    with Dataset(in_path) as ds:
        lons, lats = [ds.variables[k][:] for k in ["lon", "lat"]]
        acc_ind = ds.variables[nc_varname][:]

        if hasattr(acc_ind, "mask"):
            acc_ind = np.ma.getdata(acc_ind)




    # write the resulting shape file
    proj = from_epsg(4326)

    out_folder = Path(out_dir)
    shp_filename = ""
    if not out_folder.exists():
        out_folder.mkdir()

    schema = {
        "geometry": "Polygon",
        "properties": OrderedDict(
            [("i", "int"), ("j", "int"), ("lon", "float"), ("lat", "float"), (shp_field_name, "int")]
        )
    }


    nx, ny = len(lons), len(lats)
    dx = abs(lons[-1] - lons[0]) / (nx - 1)
    dy = abs(lats[-1] - lats[0]) / (ny - 1)

    print("dx={}, dy={}".format(dx, dy))



    with fiona.open(str(out_folder/shp_filename), mode="w", driver="ESRI Shapefile", crs=proj, schema=schema) as output:

        for i, lon in enumerate(lons):
            if (i + 1) % 100 == 0:
                print("{} / {}".format(i + 1, nx))

            # approximate region of interest (Churchill-Nelson)
            if not (-100 <= lon <= -95):
                continue


            for j, lat in enumerate(lats):


                if not (55 <= lat <= 60):
                    continue

                p00 = (lon - dx / 2.0, lat - dy / 2.0)
                p01 = (lon - dx / 2.0, lat + dy / 2.0)
                p11 = (lon + dx / 2.0, lat + dy / 2.0)
                p10 = (lon + dx / 2.0, lat - dy / 2.0)


                poly = Polygon(shell=[p00, p01, p11, p10, p00])
                props = OrderedDict([("i", i + 1), ("j", j + 1), ("lon", lon), ("lat", lat), ("acc", int(acc_ind[j, i]))])



                output.write({"geometry": mapping(poly), "properties": props})






if __name__ == '__main__':
    main()