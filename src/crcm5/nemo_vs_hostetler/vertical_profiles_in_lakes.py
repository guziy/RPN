from collections import OrderedDict
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon

from rpn import level_kinds

from crcm5.nemo_vs_hostetler import nemo_hl_util
from crcm5.nemo_vs_hostetler import commons
from crcm5.nemo_vs_hostetler.nemo_hl_util import IndexRectangle


def get_temperature_profile_HL(start_year=-np.Inf, end_year=np.Inf, samples_dir="", spatial_mask=None):
    """
    get water temperature profile from Hostetler outputs
    TT(time, z)
    :param start_year:
    :param end_year:
    :param samples_dir:
    """

    pass


def get_temperature_profile_NEMO(start_year=-np.Inf, end_year=np.Inf, samples_dir="", spatial_mask=None):
    """
    get water temperature profile from NEMO outputs
    TT(time, z)
    :param start_year:
    :param end_year:
    :param samples_dir:
    """
    pass



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

    # The average profile will be calculated over the selection
    selection = IndexRectangle(ill=50, jll=60, ni=10, nj=10)

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

    lake_mask = np.greater(commons.get_nemo_lake_mask_from_rpn(coord_file, vname="NEM1"), 0)



    fig = plt.figure()
    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    bmp.drawcoastlines(linewidth=0.3, ax=ax)

    xll, yll = xx[selection.ill, selection.jll], yy[selection.ill, selection.jll]
    xur, yur = xx[selection.get_ur_corner()], yy[selection.get_ur_corner()]

    # ax.add_patch(Polygon(np.asarray([(xll, yll), (xll, yur), (xur, yur), (xur, yll)])))

    # selection_mask = np.zeros_like(lake_mask, dtype=bool)
    # selection_mask[selection.get_2d_slice()] = True
    # bmp.pcolormesh(xx, yy, selection_mask, ax=ax)


    hl_profile = get_temperature_profile_HL(start_year=start_year, end_year=end_year, samples_dir=sim_label_to_path[HL_LABEL])
    nemo_profile = get_temperature_profile_HL(start_year=start_year, end_year=end_year, samples_dir=sim_label_to_path[HL_LABEL])


    plt.show()









if __name__ == '__main__':
    main()