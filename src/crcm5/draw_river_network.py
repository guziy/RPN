from matplotlib.ticker import FuncFormatter, FixedFormatter, ScalarFormatter
from .model_data import Crcm5ModelDataManager

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt
from util import direction_and_value
from matplotlib import cm, colors


fixedFormatter = ScalarFormatter()


def get_label(curText, pos):
    v = float(curText)
    print(type(curText), curText, pos)
    return fixedFormatter.format_data_short(np.exp(v))
    pass


def main():
    # base_data_path =  "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_wo_lakes_and_wo_lakeroff"

    #base_data_path = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_86x86_0.5deg_wo_lakes_and_wo_lakeroff"
    base_data_path = "/home/huziy/skynet3_rech1/from_guillimin/quebec_260x260_wo_lakes_and_with_lakeroff"
    base_data_manager = Crcm5ModelDataManager(samples_folder_path=base_data_path,
                                              all_files_in_samples_folder=True, need_cell_manager=True)

    acc_areas = base_data_manager.accumulation_area_km2
    fdrs = base_data_manager.flow_directions
    #cell_manager = base_data_manager.cell_manager

    lon = base_data_manager.lons2D
    lat = base_data_manager.lats2D
    basemap = base_data_manager.get_omerc_basemap(resolution="i")

    x, y = basemap(lon, lat)

    fdrs[fdrs <= 0] = 0.5

    vals = []
    di = []
    dj = []

    vals.extend(direction_and_value.values)
    di.extend(direction_and_value.iShifts)
    dj.extend(direction_and_value.jShifts)

    vals.append(0.5)
    di.append(0)
    dj.append(0)

    vals = np.array(vals)
    di = np.array(di)
    dj = np.array(dj)


    # get indices of the corresponding shifts
    kmat = np.log2(fdrs).astype(int)

    dimat = di[kmat]
    djmat = dj[kmat]

    i = list(range(x.shape[0]))
    j = list(range(x.shape[1]))

    jmat, imat = np.meshgrid(j, i)

    imat_end = imat + dimat
    jmat_end = jmat + djmat

    nx, ny = x.shape
    interest_region = ((imat_end != imat) | (jmat_end != jmat)) & (imat_end >= 0)
    interest_region = interest_region & (imat_end < nx)
    interest_region = interest_region & (jmat_end >= 0) & (jmat_end < ny)

    i1d_start = imat[interest_region]
    j1d_start = jmat[interest_region]
    i1d_end = imat_end[interest_region]
    j1d_end = jmat_end[interest_region]

    assert len(i1d_start) == len(i1d_end)

    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(1, 1, 1)

    max_acc = np.max(acc_areas)
    lwidths = acc_areas / max_acc * 20

    basemap.drawcoastlines(linewidth=0.25)
    cMap = cm.get_cmap("RdYlBu")
    lines = []
    # basemap.drawrivers(color="b")
    # basemap.drawmapboundary(fill_color="aqua")

    im = basemap.etopo()
    # im = basemap.shadedrelief(zorder = 2)
    # plt.colorbar(im)
    # basemap.fillcontinents(lake_color="aqua")

    x1, x2, y1, y2 = x[i1d_start, j1d_start], x[i1d_end, j1d_end], y[i1d_start, j1d_start], y[i1d_end, j1d_end]
    # basemap.quiver(x1,y1,x2-x1, y2-y1, scale_units="xy", angles = "xy", scale = 1,
    #         color = "k", zorder=2, headwidth = 1, headlength = 0.2)
    basemap.scatter(x, y, color="k", s=1, linewidths=0)

    plt.tight_layout()
    plt.savefig("domain1.pdf")

    pass


if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()
    from util import plot_utils

    plot_utils.apply_plot_params(width_cm=15, height_cm=15)
    main()
    print("Hello world")