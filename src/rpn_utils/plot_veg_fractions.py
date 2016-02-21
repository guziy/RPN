import os
from collections import OrderedDict

from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap, maskoceans
from scipy.spatial.ckdtree import cKDTree
from crcm5.analyse_hdf import common_plot_params
from domains.rotated_lat_lon import RotatedLatLon
from util import plot_utils
from util.geo import lat_lon

__author__ = 'huziy'

from rpn.rpn import RPN
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm




y2c_level_to_title = {
    2: "Broadleaf", 1: "Needleleaf", 3: "Crops", 4: "Grass"
}

vegkey_to_title = {
    "needleleaftrees": "Needle-leaf trees",
    "broadleaftrees": "Broadleaf trees",
    "crops": "Crops",
    "grass": "Grass"
}

VGCLASS = [
    0, 0, 0, 1, 2,
    1, 2, 2, 2, 4,
    2, 4, 4, 4, 3,
    3, 3, 3, 3, 3,
    5, 4, 4, 6, 12,
    4]


def level_to_veg_class(lev):
    """
    get the vegetation class for each of 26 types of vegetation used in gem
    return codes:
    0 - ignore the field
    1 - needle leaf trees
    2 - broad leaf trees
    3 - crops
    4 - grass
    5 - urban
    6 - bare ground
    :param lev: level in the geophysical fields file
    """
    return VGCLASS[int(lev) - 1]


def plot_depth_to_bedrock(basemap,
                          lons1, lats1, field1, label1,
                          lons2, lats2, field2, label2,
                          base_folder="/skynet3_rech1/huziy/veg_fractions/"
                          ):
    xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons2.flatten(), lats2.flatten())
    xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons1.flatten(), lats1.flatten())

    ktree = cKDTree(list(zip(xs, ys, zs)))
    dists, inds = ktree.query(list(zip(xt, yt, zt)))

    levels = np.arange(0, 5.5, 0.5)

    field2_interp = field2.flatten()[inds].reshape(field1.shape)

    field1 = np.ma.masked_where(field1 < 0, field1)
    field2_interp = np.ma.masked_where(field2_interp < 0, field2_interp)

    vmin = min(field1.min(), field2_interp.min())
    vmax = max(field1.max(), field2_interp.max())
    cmap = cm.get_cmap("BuPu", len(levels) - 1)
    bn = BoundaryNorm(levels, len(levels) - 1)
    x, y = basemap(lons1, lats1)
    imname = "depth_to_bedrock_{0}-{1}.jpeg".format(label2, label1)
    impath = os.path.join(base_folder, imname)
    fig = plt.figure(figsize=(6, 2.5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])
    ax = fig.add_subplot(gs[0, 0])
    basemap.pcolormesh(x, y, field1, vmin=vmin, vmax=vmax, cmap=cmap, norm=bn)
    basemap.drawcoastlines(linewidth=common_plot_params.COASTLINE_WIDTH)
    ax.set_title(label1)

    ax = fig.add_subplot(gs[0, 1])
    im = basemap.pcolormesh(x, y, field2_interp, vmin=vmin, vmax=vmax, cmap=cmap, norm=bn)
    basemap.drawcoastlines(linewidth=common_plot_params.COASTLINE_WIDTH)
    ax.set_title(label2)

    plt.colorbar(im, cax=fig.add_subplot(gs[0, 2]))
    fig.tight_layout()
    fig.savefig(impath, dpi=common_plot_params.FIG_SAVE_DPI)


def plot_difference(basemap,
                    lons1, lats1, data1, label1,
                    lons2, lats2, data2, label2,
                    base_folder="/skynet3_rech1/huziy/veg_fractions/"
                    ):
    xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons2.flatten(), lats2.flatten())
    xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons1.flatten(), lats1.flatten())

    ktree = cKDTree(list(zip(xs, ys, zs)))
    dists, inds = ktree.query(list(zip(xt, yt, zt)))

    # Calculate differences
    diff_dict = {}
    for key, the_field in data2.items():
        diff_dict[key] = the_field.flatten()[inds].reshape(data1[key].shape) - data1[key]

    x, y = basemap(lons1, lats1)
    imname = "sand_clay_diff_{0}-{1}.jpeg".format(label2, label1)
    impath = os.path.join(base_folder, imname)
    plot_sand_and_clay_diff(x, y, basemap, diff_dict["SAND"], diff_dict["CLAY"],
                            out_image=impath)

    del diff_dict["SAND"], diff_dict["CLAY"]
    imname = "veg_fract_diff_{0}-{1}.jpeg".format(label2, label1)
    impath = os.path.join(base_folder, imname)
    plot_veg_fractions_diff(x, y, basemap, diff_dict,
                            out_image=impath)


def plot_sand_and_clay_diff(x, y, basemap, sand, clay, out_image=""):
    fig = plt.figure(figsize=(6, 2.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])

    delta = 1
    step = 0.2
    clevels = list(np.arange(-delta, 0, step)) + list(np.arange(step, delta + step, step))
    bn = BoundaryNorm(clevels, len(clevels) - 1)

    cmap = cm.get_cmap("seismic", len(clevels) - 1)

    # plot sand
    ax = fig.add_subplot(gs[0, 0])
    cs = basemap.pcolormesh(x, y, sand / 100.0, vmin=-1, vmax=1, ax=ax,
                            cmap=cmap, norm=bn)
    basemap.drawcoastlines(linewidth=common_plot_params.COASTLINE_WIDTH)
    ax.set_title("Sand")

    # plot clay
    ax = fig.add_subplot(gs[0, 1])
    cs = basemap.pcolormesh(x, y, clay / 100.0, vmin=-1, vmax=1, ax=ax, cmap=cmap,
                            norm=bn)
    basemap.drawcoastlines(linewidth=common_plot_params.COASTLINE_WIDTH)
    ax.set_title("Clay")

    ax = fig.add_subplot(gs[0, 2])
    plt.colorbar(cs, cax=ax, ticks=clevels)
    fig.savefig(out_image, bbox_inches="tight", dpi=common_plot_params.FIG_SAVE_DPI)


def plot_veg_fractions_diff(x, y, basemap, veg_data, out_image=""):
    fig = plt.figure(figsize=(6, 5))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])

    delta = 1
    step = 0.2
    clevels = list(np.arange(-delta, 0, step)) + list(np.arange(step, delta + step, step))
    bn = BoundaryNorm(clevels, len(clevels) - 1)
    cmap = cm.get_cmap("RdBu_r", len(clevels) - 1)

    cs = None
    index = 0
    for title, data in veg_data.items():
        row = index // 2
        col = index % 2
        ax = fig.add_subplot(gs[row, col])
        cs = basemap.pcolormesh(x, y, data, vmin=-1, vmax=1, ax=ax,
                                cmap=cmap, norm=bn)

        basemap.drawcoastlines(linewidth=common_plot_params.COASTLINE_WIDTH)
        ax.set_title(title)
        index += 1
    plt.colorbar(cs, cax=fig.add_subplot(gs[:, 2]), ticks=clevels)
    fig.savefig(out_image, bbox_inches="tight", dpi=common_plot_params.FIG_SAVE_DPI)


def plot_sand_and_clay(x, y, basemap, sand, clay, out_image=""):
    fig = plt.figure(figsize=(6, 2.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
    clevels = np.arange(0, 1.1, 0.1)

    cmap = cm.get_cmap("jet", len(clevels) - 1)

    # plot sand
    ax = fig.add_subplot(gs[0, 0])
    cs = basemap.pcolormesh(x, y, sand / 100.0, vmin=0, vmax=1, ax=ax, cmap=cmap)
    basemap.drawcoastlines(linewidth=common_plot_params.COASTLINE_WIDTH)
    ax.set_title("Sand")

    # plot clay
    ax = fig.add_subplot(gs[0, 1])
    cs = basemap.pcolormesh(x, y, clay / 100.0, vmin=0, vmax=1, ax=ax, cmap=cmap)
    basemap.drawcoastlines(linewidth=common_plot_params.COASTLINE_WIDTH)
    ax.set_title("Clay")

    ax = fig.add_subplot(gs[0, 2])
    plt.colorbar(cs, cax=ax, ticks=clevels)
    fig.savefig(out_image, bbox_inches="tight", dpi=common_plot_params.FIG_SAVE_DPI)


def plot_veg_fractions(x, y, basemap, veg_data, out_image=""):
    fig = plt.figure(figsize=(6, 5))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])

    clevels = np.arange(0, 1.1, 0.1)
    cmap = cm.get_cmap("jet", len(clevels) - 1)

    cs = None


    main_veg_types = OrderedDict()

    # calculate the fractions of 4 main vegetation types
    for level, data in veg_data.items():
        cl = level_to_veg_class(level)

        # ignore urban, desert and bare soil
        if cl in [0, 5, 6]:
            continue


        # needle leaf
        if cl == 1:
            key = "needleleaftrees"

        # broad leaf
        if cl == 2:
            key = "broadleaftrees"

        # crops
        if cl == 3:
            key = "crops"

        # grass
        if cl == 4:
            key = "grass"

        # mixed needleleaf and broadleaf trees categories
        if cl == 12:
            key = "needleleaftrees"
            main_veg_types[key] = main_veg_types.get(key, np.zeros_like(data)) + data * 0.5

            key = "broadleaftrees"
            main_veg_types[key] = main_veg_types.get(key, np.zeros_like(data)) + data * 0.5
        else:
            main_veg_types[key] = main_veg_types.get(key, np.zeros_like(data)) + data


    veg_keys = ("needleleaftrees", "broadleaftrees", "crops", "grass")
    for cl, veg_key in enumerate(veg_keys):

        data = main_veg_types[veg_key]
        print(data.min(), data.max())
        data[data > 1] = 1
        title = vegkey_to_title[veg_key]
        row = cl // 2
        col = cl % 2
        ax = fig.add_subplot(gs[row, col])
        # cs = basemap.pcolormesh(x, y, data, vmin=0, vmax=1, ax=ax, cmap=cmap)
        cs = basemap.contourf(x, y, data, levels=clevels, cmap=cmap)
        basemap.drawcoastlines(linewidth=common_plot_params.COASTLINE_WIDTH)
        ax.set_title(title)

    plt.colorbar(cs, cax=fig.add_subplot(gs[:, 2]), ticks=clevels)
    fig.savefig(out_image, bbox_inches="tight", dpi=common_plot_params.FIG_SAVE_DPI)


def main(base_folder="/skynet3_rech1/huziy/veg_fractions/",
         fname="pm1983120100_00000000p", canopy_name="Y2C", label="USGS",
         depth_to_bedrock_name="8L"
         ):
    data_path = os.path.join(base_folder, fname)
    r = RPN(data_path)

    veg_fractions = r.get_2D_field_on_all_levels(name=canopy_name)
    print(list(veg_fractions.keys()))
    sand = r.get_first_record_for_name("SAND")
    clay = r.get_first_record_for_name("CLAY")

    dpth_to_bedrock = r.get_first_record_for_name(depth_to_bedrock_name)

    proj_params = r.get_proj_parameters_for_the_last_read_rec()

    lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()
    print(lons.shape)

    rll = RotatedLatLon(lon1=proj_params["lon1"], lat1=proj_params["lat1"],
                        lon2=proj_params["lon2"], lat2=proj_params["lat2"])

    lon0, lat0 = rll.get_true_pole_coords_in_rotated_system()
    plon, _ = rll.get_north_pole_coords()

    b = Basemap(projection="rotpole", llcrnrlon=lons[0, 0], llcrnrlat=lats[0, 0],
                urcrnrlon=lons[-1, -1], urcrnrlat=lats[-1, -1], lon_0=lon0 - 180,
                o_lon_p=lon0, o_lat_p=lat0)

    lons[lons > 180] -= 360
    for lev in list(veg_fractions.keys()):
        veg_fractions[lev] = maskoceans(lons, lats, veg_fractions[lev], inlands=False)

    sand = maskoceans(lons, lats, sand)
    clay = maskoceans(lons, lats, clay)
    dpth_to_bedrock = maskoceans(lons, lats, dpth_to_bedrock)

    x, y = b(lons, lats)
    plot_veg_fractions(x, y, b, veg_fractions, out_image=os.path.join(base_folder,
                                                                      "veg_fractions_{0}.jpeg".format(label)))
    plot_sand_and_clay(x, y, b, sand, clay, out_image=os.path.join(base_folder,
                                                                   "sand_clay_{0}.jpeg".format(label)))

    # set relation between vegetation frsction fields and names
    veg_fract_dict = {}
    for lev, the_field in veg_fractions.items():
        lev = int(lev)
        if lev not in y2c_level_to_title:
            continue
        veg_fract_dict[y2c_level_to_title[lev]] = the_field

    data = {
        "SAND": sand, "CLAY": clay, "BDRCK_DEPTH": dpth_to_bedrock
    }
    data.update(veg_fract_dict)

    return b, lons, lats, data, label


def plot_only_vegetation_fractions(
        data_path="/RESCUE/skynet3_rech1/huziy/geof_lake_infl_exp/geophys_Quebec_0.1deg_260x260_with_dd_v6_with_ITFS",
        canopy_name="VF", label="QC_10km"):
    r = RPN(data_path)

    veg_fractions = r.get_2D_field_on_all_levels(name=canopy_name)
    print(list(veg_fractions.keys()))

    proj_params = r.get_proj_parameters_for_the_last_read_rec()

    lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()
    print(lons.shape)

    rll = RotatedLatLon(lon1=proj_params["lon1"], lat1=proj_params["lat1"],
                        lon2=proj_params["lon2"], lat2=proj_params["lat2"])

    lon0, lat0 = rll.get_true_pole_coords_in_rotated_system()
    plon, _ = rll.get_north_pole_coords()

    b = Basemap(projection="rotpole", llcrnrlon=lons[0, 0], llcrnrlat=lats[0, 0],
                urcrnrlon=lons[-1, -1], urcrnrlat=lats[-1, -1], lon_0=lon0 - 180,
                o_lon_p=lon0, o_lat_p=lat0)

    lons[lons > 180] -= 360
    for lev in list(veg_fractions.keys()):
        veg_fractions[lev] = maskoceans(lons, lats, veg_fractions[lev], inlands=False)

    x, y = b(lons, lats)
    plot_veg_fractions(x, y, b, veg_fractions, out_image=os.path.join(os.path.dirname(data_path),
                                                                      "veg_fractions_{0}.png".format(label)))


if __name__ == "__main__":
    # main()
    import application_properties


    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=20, height_cm=17)

    application_properties.set_current_directory()
    plot_only_vegetation_fractions()
    # main(base_folder="/home/huziy/skynet3_rech1/hdf_store", fname="pm1979010100_00000000p",
    #     label="veg_qc_10km",
    #     canopy_name="VF")
