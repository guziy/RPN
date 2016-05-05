# rotate wind vectors from the geographical, lat, lon to the projection coordinates
from collections import OrderedDict

from mpl_toolkits.basemap import Basemap
from pathlib import Path

from rpn import level_kinds
from rpn.rpn import RPN
from rpn.rpn_multi import MultiRPN

from application_properties import main_decorator


import os

from crcm5.nemo_vs_hostetler import nemo_hl_util

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, datetime, date2num


def rotate_vecs_from_geo_to_rotpole(uu_geo, vv_geo, lons, lats, bmp=None):


    urot, vrot = bmp.rotate_vector(uu_geo, vv_geo, lons=lons, lats=lats)
    return urot, vrot






def test():
    plt.figure()
    b = Basemap()
    u = np.array([1, ])
    v = np.array([1, ])
    lon, lat = np.array([-90, ]), np.array([45, ])
    xx, yy = b(lon, lat)
    print(xx.shape)
    b.quiver(xx, yy, u, v, color="r")

    urot, vrot = b.rotate_vector(u, v, lon, lat)
    b.quiver(xx, yy, urot, vrot, color="g")

    b.drawcoastlines()



    # Plot the same in rotpole projection

    HL_LABEL = "CRCM5_HL"
    NEMO_LABEL = "CRCM5_NEMO"

    sim_label_to_path = OrderedDict(
        [(HL_LABEL, "/RESCUE/skynet3_rech1/huziy/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl_oneway/Samples"),
         (NEMO_LABEL, "/HOME/huziy/skynet3_rech1/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl/Samples")]
    )

    # get a coord file ...
    coord_file = ""
    found_coord_file = False
    for mdir in os.listdir(sim_label_to_path[HL_LABEL]):

        mdir_path = os.path.join(sim_label_to_path[HL_LABEL], mdir)
        if not os.path.isdir(mdir_path):
            continue

        for fn in os.listdir(mdir_path):
            print(fn)
            if fn[:2] not in ["pm", "dm", "pp", "dp"]:
                continue

            coord_file = os.path.join(mdir_path, fn)
            found_coord_file = True

        if found_coord_file:
            break

    bmp, lons, lats = nemo_hl_util.get_basemap_obj_and_coords_from_rpn_file(path=coord_file)


    plt.figure()
    urot, vrot = bmp.rotate_vector(u, v, lon, lat)
    xx, yy = bmp(lon, lat)

    bmp.quiver(xx, yy, urot, vrot, color="b")
    bmp.drawcoastlines()
    plt.show()




@main_decorator
def main():
    start_year = 1979
    end_year = 1981

    HL_LABEL = "CRCM5_HL"
    NEMO_LABEL = "CRCM5_NEMO"

    file_prefix = "dm"
    level = 1
    level_type = level_kinds.HYBRID

    wind_comp_names = ["UU", "VV"]


    sim_label_to_path = OrderedDict(
        [(HL_LABEL, "/RESCUE/skynet3_rech1/huziy/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl_oneway/Samples"),
         (NEMO_LABEL, "/HOME/huziy/skynet3_rech1/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl/Samples")]
    )

    # get a coord file ...
    coord_file = ""
    found_coord_file = False
    for mdir in os.listdir(sim_label_to_path[HL_LABEL]):

        mdir_path = os.path.join(sim_label_to_path[HL_LABEL], mdir)
        if not os.path.isdir(mdir_path):
            continue

        for fn in os.listdir(mdir_path):
            print(fn)
            if fn[:2] not in ["pm", "dm", "pp", "dp"]:
                continue

            coord_file = os.path.join(mdir_path, fn)
            found_coord_file = True

        if found_coord_file:
            break

    bmp, lons, lats = nemo_hl_util.get_basemap_obj_and_coords_from_rpn_file(path=coord_file)
    xx, yy = bmp(lons, lats)




    # loop through all files rotate vaectors and save to netcdf
    for sim_label, samples_dir in sim_label_to_path.items():

        samples = Path(samples_dir)
        po = samples.parent

        monthdate_to_path_list = nemo_hl_util.get_monthyeardate_to_paths_map(file_prefix=file_prefix,
                                                                             start_year=start_year, end_year=end_year,
                                                                             samples_dir_path=samples)

        # Netcdf output file to put rotated winds
        po /= "rotated_wind_{}.nc".format(sim_label)

        with Dataset(str(po), "w") as ds:

            ds.createDimension("time", None)
            ds.createDimension("lon", lons.shape[0])
            ds.createDimension("lat", lons.shape[1])

            # create the schema of the output file
            vname_to_ncvar = {}
            for vname in wind_comp_names:
                vname_to_ncvar[vname] = ds.createVariable(vname, "f4", dimensions=("time", "lon", "lat"))
                vname_to_ncvar[vname].units = "knots"

            lons_var = ds.createVariable("lon", "f4", dimensions=("lon", "lat"))
            lats_var = ds.createVariable("lat", "f4", dimensions=("lon", "lat"))
            time_var = ds.createVariable("time", "i8", dimensions=("time",))
            time_var.units = "hours since {:%Y-%m-%d %H:%M:%S}".format(datetime(start_year, 1, 1))

            lons_var[:] = lons
            lats_var[:] = lats


            # use sorted dates
            record_count = 0


            for month_date in sorted(monthdate_to_path_list):
                # select only dm files
                mr = MultiRPN(path=monthdate_to_path_list[month_date])


                vname_to_fields = {}
                for vname in wind_comp_names:
                    vname_to_fields[vname] = mr.get_all_time_records_for_name_and_level(varname=vname, level=level, level_kind=level_type)

                for ti, t in enumerate(sorted(vname_to_fields[wind_comp_names[0]])):
                    time_var[record_count] = date2num(t, time_var.units)

                    uu = vname_to_fields[wind_comp_names[0]][t]
                    vv = vname_to_fields[wind_comp_names[1]][t]

                    uu_rot, vv_rot = rotate_vecs_from_geo_to_rotpole(uu, vv, lons, lats, bmp=bmp)


                    # in knots not in m/s
                    vname_to_ncvar[wind_comp_names[0]][record_count, :, :] = uu_rot
                    vname_to_ncvar[wind_comp_names[1]][record_count, :, :] = vv_rot
                    record_count += 1



if __name__ == '__main__':
    main()
    #  test()
