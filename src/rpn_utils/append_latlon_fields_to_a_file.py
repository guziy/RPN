

import sys
from pathlib import Path

from rpn.rpn import RPN


def main(path_param=None):


    if path_param is None:
        if len(sys.argv) == 1:
            path = "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/misc_fields/gridcell_areas/test.rpn"
        else:
            path = sys.argv[1]

    else:
        path = path_param


    with RPN(path, mode="a") as r:
        assert isinstance(r, RPN)

        # ignore coordinate records
        vname = [v for v in r.get_list_of_varnames() if v not in [">>", "^^", "HY"]][0]

        # get any field just to get the metadata and coord indicators
        field = r.get_first_record_for_name(vname)

        lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()



        info = r.get_current_info()

        print(info)

        info["varname"] = "LON"
        r.write_2d_field_clean(lons, properties=info)

        info["varname"] = "LAT"
        r.write_2d_field_clean(lons, properties=info)



def append_lat_lon_to_all_files_in_folder(folder="/HOME/huziy/skynet3_rech1/NEI_geophysics/misc_fields/gridcell_areas"):
    folder_p = Path(folder)

    for f in folder_p.iterdir():
        main(path_param=str(f))



if __name__ == '__main__':
    append_lat_lon_to_all_files_in_folder()
    # main()