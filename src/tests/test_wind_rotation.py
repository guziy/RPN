from collections import OrderedDict

from mpl_toolkits.basemap import Basemap
from rpn import level_kinds
import os

from crcm5.nemo_vs_hostetler import nemo_hl_util
import matplotlib.pyplot as plt
from netCDF4 import Dataset

from crcm5.nemo_vs_hostetler.rotate_wind_vectors import rotate_vecs_from_geo_to_rotpole


def main():
    start_year = 1979
    end_year = 1981

    HL_LABEL = "CRCM5_HL"
    NEMO_LABEL = "CRCM5_NEMO"

    dx = 0.1
    dy = 0.1

    file_prefix = "pm"
    PR_level = -1
    PR_level_type = level_kinds.ARBITRARY

    tprecip_vname = "PR"
    sprecip_vname = "SN"

    TT_level = 1
    TT_level_type = level_kinds.HYBRID

    sim_label_to_path = OrderedDict(
        [(HL_LABEL, "/RESCUE/skynet3_rech1/huziy/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl_oneway/Samples"),
         (NEMO_LABEL, "/HOME/huziy/skynet3_rech1/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl/Samples")]
    )

    # get a coord file ... (use pm* files, since they contain NEM1 variable)
    # Should be NEMO_LABEL, since the hostetler case does not calculate NEM? vars
    coord_file = ""
    found_coord_file = False
    for mdir in os.listdir(sim_label_to_path[NEMO_LABEL]):

        mdir_path = os.path.join(sim_label_to_path[NEMO_LABEL], mdir)
        if not os.path.isdir(mdir_path):
            continue

        for fn in os.listdir(mdir_path):

            if fn[:2] not in ["pm", ]:
                continue

            if fn[-9:-1] == "0" * 8:
                continue

            coord_file = os.path.join(mdir_path, fn)
            found_coord_file = True

        if found_coord_file:
            break

    bmp, lons, lats = nemo_hl_util.get_basemap_obj_and_coords_from_rpn_file(path=coord_file)
    xx, yy = bmp(lons, lats)


    stride = 3

    #
    rot_path = "/HOME/huziy/skynet3_rech1/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl/rotated_wind_CRCM5_NEMO.nc"
    fig = plt.figure()

    with Dataset(rot_path) as ds:
        ncvars = ds.variables
        uu_rot, vv_rot = ncvars["UU"][10, ...], ncvars["VV"][10, ...]

    plt.title("rotated in the file")
    bmp.quiver(xx[::stride, ::stride], yy[::stride, ::stride], uu_rot[::stride, ::stride], vv_rot[::stride, ::stride], scale=1000, color="r")


    lons[lons > 180] -= 360

    not_rot_path = "/HOME/huziy/skynet3_rech1/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl/not_rotated_wind_CRCM5_NEMO.nc"
    with Dataset(not_rot_path) as ds:
        ncvars = ds.variables
        uu, vv = ncvars["UU"][10, ...], ncvars["VV"][10, ...]

        uu_rot1, vv_rot1 = rotate_vecs_from_geo_to_rotpole(uu, vv, lons, lats, bmp=bmp)


    plt.title("not rotated in the file")
    bmp.quiver(xx[::stride, ::stride], yy[::stride, ::stride], uu_rot1[::stride, ::stride], vv_rot1[::stride, ::stride],
               scale=1000)
    bmp.drawcoastlines()





    plt.figure()
    b = Basemap(lon_0=0)
    xx1, yy1 = b(lons, lats)

    uu_rot2, vv_rot2 = b.rotate_vector(uu, vv, lons, lats)
    b.quiver(xx1[::stride, ::stride], yy1[::stride, ::stride], uu_rot2[::stride, ::stride], vv_rot2[::stride, ::stride],
               scale=1000)

    b.drawcoastlines()

    plt.show()


if __name__ == '__main__':
    main()
