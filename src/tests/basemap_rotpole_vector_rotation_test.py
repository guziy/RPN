from collections import OrderedDict

import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

import os

from crcm5.nemo_vs_hostetler import nemo_hl_util


def main():



    # HL_LABEL = "CRCM5_HL"
    # NEMO_LABEL = "CRCM5_NEMO"
    #
    #
    # sim_label_to_path = OrderedDict(
    #     [(HL_LABEL, "/RESCUE/skynet3_rech1/huziy/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl_oneway/Samples"),
    #      (NEMO_LABEL, "/HOME/huziy/skynet3_rech1/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl/Samples")]
    # )
    #
    #
    # # get a coord file ... (use pm* files, since they contain NEM1 variable)
    # # Should be NEMO_LABEL, since the hostetler case does not calculate NEM? vars
    # coord_file = ""
    # found_coord_file = False
    # for mdir in os.listdir(sim_label_to_path[NEMO_LABEL]):
    #
    #     mdir_path = os.path.join(sim_label_to_path[NEMO_LABEL], mdir)
    #     if not os.path.isdir(mdir_path):
    #         continue
    #
    #     for fn in os.listdir(mdir_path):
    #
    #         if fn[:2] not in ["pm", ]:
    #             continue
    #
    #         if fn[-9:-1] == "0" * 8:
    #             continue
    #
    #         coord_file = os.path.join(mdir_path, fn)
    #         found_coord_file = True
    #
    #     if found_coord_file:
    #         break
    #
    #
    #
    # bmp, lons, lats = nemo_hl_util.get_basemap_obj_and_coords_from_rpn_file(path=coord_file)
    # xx, yy = bmp(lons, lats)

    u = np.array([0.0, ])
    v = np.array([0.5, ])

    lon = -84
    lat = 45

    lon = np.array([lon, ])
    lat = np.array([lat, ])

    b = Basemap(lon_0=0)

    urot, vrot = b.rotate_vector(u, v, lon, lat)
    xx, yy = b(lon, lat)

    b.quiver(xx, yy, urot, vrot, color="r")

    b.quiver(xx, yy, u, v, color="g")



    b.drawcoastlines()

    plt.show()


if __name__ == '__main__':
    main()