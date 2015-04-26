from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.figure import Figure
from matplotlib.ticker import LinearLocator, MultipleLocator, LogLocator, MaxNLocator, ScalarFormatter, FuncFormatter
from mpl_toolkits.basemap import maskoceans, Basemap
import os
from crcm5 import infovar
from crcm5.analyse_hdf import common_plot_params
from data.cell_manager import CellManager

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt
from . import do_analysis_using_pytables as analysis
from . import common_plot_params as cpp
from matplotlib import cm

images_folder = "/home/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/images_for_lake-river_paper"


def _plot_lake_fraction(ax, basemap, x, y, field, title="", cmap=None):
    ax.set_title(title)
    if cmap is not None:
        bn = BoundaryNorm(np.arange(0, 1.1, 0.1), cmap.N)
        im = basemap.pcolormesh(x, y, field, ax=ax, cmap=cmap, norm=bn, zorder=5)
    else:
        im = basemap.pcolormesh(x, y, field, ax=ax)

    basemap.colorbar(im)


def _plot_sand_clay_percentages(ax, basemap, x, y, field, title="", cmap=None):
    ax.set_title(title)
    if cmap is not None:
        bn = BoundaryNorm(np.arange(0, 110, 10), cmap.N)
        im = basemap.pcolormesh(x, y, field, ax=ax, cmap=cmap, norm=bn)
    else:
        im = basemap.pcolormesh(x, y, field, ax=ax)

    basemap.colorbar(im)


def _plot_slope(ax, basemap, x, y, field, title="", cmap=None):
    ax.set_title(title)
    if cmap is not None:
        the_norm = LogNorm(vmin=1.0e-5, vmax=1)
        im = basemap.pcolormesh(x, y, field, ax=ax, cmap=cmap, norm=the_norm)
    else:
        im = basemap.pcolormesh(x, y, field, ax=ax)

    basemap.colorbar(im)


def _plot_accumulation_area(ax, basemap, x, y, field, title="", cmap=None):
    ax.set_title(title)
    if cmap is not None:
        field = np.ma.masked_where(field <= 0, field)
        norm = LogNorm(vmin=10, vmax=1e6)
        im = basemap.pcolormesh(x, y, field, ax=ax, cmap=cmap, norm=norm)
        basemap.colorbar(im)
    else:
        im = basemap.pcolormesh(x, y, field, ax=ax)
        basemap.colorbar(im, format="%.1f")



def _plot_soil_hydraulic_conductivity(ax, basemap, x, y, field, title="", cmap=None):
    ax.set_title(title)
    if cmap is not None:
        levels = np.linspace(field.min(), field.max(), cmap.N + 1)
        levels = np.round(levels, decimals=6)
        bn = BoundaryNorm(levels, cmap.N)
        im = basemap.pcolormesh(x, y, field, ax=ax, cmap=cmap, norm = bn)
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits([-2, 3])


        cb = basemap.colorbar(im, ticks=levels, format=fmt)
        cax = cb.ax
        cax.yaxis.get_offset_text().set_position((-3, 5))



    else:
        im = basemap.pcolormesh(x, y, field, ax=ax)
        basemap.colorbar(im, format="%.1f")


def _plot_field(ax, basemap, x, y, field, title="", cmap=None):
    ax.set_title(title)
    if cmap is not None:
        im = basemap.pcolormesh(x, y, field, ax=ax, cmap=cmap)
        basemap.colorbar(im, ticks=LinearLocator(numticks=cmap.N + 1), format="%.1f")
    else:
        im = basemap.pcolormesh(x, y, field, ax=ax)
        basemap.colorbar(im, format="%.1f")
    basemap.drawcoastlines(ax=ax, linewidth=cpp.COASTLINE_WIDTH)


def plot_histograms(path="/home/huziy/skynet3_rech1/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup_ecoclimap.hdf"):
    fig = plt.figure()
    assert isinstance(fig, Figure)
    gs = gridspec.GridSpec(3, 3)

    lons2d, lats2d, basemap = analysis.get_basemap_from_hdf(file_path=path)

    # slope
    ch_slope = analysis.get_array_from_file(path=path, var_name="slope")
    ch_slope = maskoceans(lons2d, lats2d, ch_slope)
    ch_slope = np.ma.masked_where(ch_slope.mask | (ch_slope < 0), ch_slope)
    ax = fig.add_subplot(gs[0, 0])
    assert isinstance(ax, Axes)
    ch_slope_flat = ch_slope[~ch_slope.mask]
    the_hist, positions = np.histogram(ch_slope_flat, bins=25, range=[0, np.percentile(ch_slope_flat, 90)])
    the_hist = the_hist.astype(float)
    the_hist /= the_hist.sum()
    barwidth = (positions[1] - positions[0]) * 0.9
    ax.bar(positions[:-1], the_hist, color="0.75", linewidth=0, width=barwidth)
    ax.set_title(r"$\alpha$")
    ax.grid()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # drainage density
    dd = analysis.get_array_from_file(path=path, var_name="drainage_density_inv_meters")
    dd *= 1000  # convert to km^-1
    ax = fig.add_subplot(gs[0, 1])
    assert isinstance(ax, Axes)
    dd_flat = dd[~ch_slope.mask]
    the_hist, positions = np.histogram(dd_flat, bins=25, range=[0, np.percentile(dd_flat, 90)])
    the_hist = the_hist.astype(np.float)
    the_hist /= the_hist.sum()
    print(the_hist.max(), the_hist.min())
    barwidth = (positions[1] - positions[0]) * 0.9
    ax.bar(positions[:-1], the_hist, color="0.75", linewidth=0, width=barwidth)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_title(r"$DD {\rm \left( km^{-1} \right)}$")
    ax.grid()


    # vertical soil hydraulic conductivity
    vshc = analysis.get_array_from_file(path=path, var_name=infovar.HDF_VERT_SOIL_HYDR_COND_NAME)
    if vshc is not None:
        # get only on the first layer
        vshc = vshc[0, :, :]
        ax = fig.add_subplot(gs[1, 0])
        assert isinstance(ax, Axes)
        vshc_flat = vshc[~ch_slope.mask]
        the_hist, positions = np.histogram(vshc_flat, bins=25, range=[0, np.percentile(vshc_flat, 90)])
        the_hist = the_hist.astype(np.float)
        the_hist /= the_hist.sum()
        print(the_hist.max(), the_hist.min())
        barwidth = (positions[1] - positions[0]) * 0.9
        ax.bar(positions[:-1], the_hist, color="0.75", linewidth=0, width=barwidth)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # set a scalar formatter
        sfmt = ScalarFormatter(useMathText=True)
        sfmt.set_powerlimits([-2, 2])
        ax.xaxis.set_major_formatter(sfmt)
        ax.set_title(r"$ K_{\rm V} {\rm (m/s)}$")
        ax.grid()

        # Kv * slope * DD
        ax = fig.add_subplot(gs[1, 1])
        assert isinstance(ax, Axes)

        interflow_h = 0.2  # Soulis et al 2000
        # 1e-3 is to convert drainage density to m^-1
        the_prod = dd_flat * 1e-3 * vshc_flat * ch_slope_flat * 48 * interflow_h

        print("product median: {0}".format(np.median(the_prod)))
        print("product maximum: {0}".format(the_prod.max()))
        print("product 90-quantile: {0}".format(np.percentile(the_prod, 90)))

        the_hist, positions = np.histogram(the_prod, bins=25, range=[0, np.percentile(the_prod, 90)])
        the_hist = the_hist.astype(np.float)
        the_hist /= the_hist.sum()
        print(the_hist.max(), the_hist.min())
        barwidth = (positions[1] - positions[0]) * 0.9
        ax.bar(positions[:-1], the_hist, color="0.75", linewidth=0, width=barwidth)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # set a scalar formatter
        sfmt = ScalarFormatter(useMathText=True)
        sfmt.set_powerlimits([-2, 2])
        ax.xaxis.set_major_formatter(sfmt)
        ax.set_title(r"$ \beta_{\rm max}\cdot K_{\rm v} \cdot \alpha \cdot DD \cdot H {\rm (m/s)}$ ")
        ax.grid()

        # read flow directions
        flow_directions = analysis.get_array_from_file(path=path, var_name=infovar.HDF_FLOW_DIRECTIONS_NAME)
        # read cell areas
        # cell_areas = analysis.get_array_from_file(path=path, var_name=infovar.HDF_CELL_AREA_NAME)
        cell_manager = CellManager(flow_directions)
        acc_index = cell_manager.get_accumulation_index()
        acc_index_flat = acc_index[acc_index > 1]
        print("acc_index: min={0}; max={1}; median={2}; 90-quantile={3}".format(
            acc_index_flat.min(), acc_index_flat.max(), np.median(acc_index_flat), np.percentile(acc_index_flat, 90)))

        # plot the range of the accumulation index
        ax = fig.add_subplot(gs[0, 2])
        assert isinstance(ax, Axes)
        the_hist, positions = np.histogram(acc_index_flat, bins=25, range=[0, np.percentile(acc_index_flat, 90)])
        the_hist = the_hist.astype(np.float)
        the_hist /= the_hist.sum()
        print(the_hist.max(), the_hist.min())
        barwidth = (positions[1] - positions[0]) * 0.9
        ax.bar(positions[:-1], the_hist, color="0.75", linewidth=0, width=barwidth)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # set a scalar formatter
        sfmt = ScalarFormatter(useMathText=True)
        sfmt.set_powerlimits([-2, 2])
        ax.xaxis.set_major_formatter(sfmt)
        ax.set_title(r"Accum. index")
        ax.grid()





    # lake fraction


    # sand

    # clay


    fig_path = os.path.join(images_folder, "static_fields_histograms.jpeg")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=cpp.FIG_SAVE_DPI, bbox_inches="tight")


def main():
    fig = plt.figure()
    assert isinstance(fig, Figure)
    gs = gridspec.GridSpec(3, 3, wspace=0.4)

    # plot the control

    path1 = "/home/huziy/skynet3_rech1/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_do_not_discard_small.hdf"
    # path = "/home/huziy/skynet3_rech1/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup_ecoclimap.hdf"
    # path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup_ecoclimap_era075.hdf"

    path = "/home/huziy/skynet3_rech1/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup_ITFS.hdf5"

    slope = analysis.get_array_from_file(path=path, var_name="slope")
    itf_slope = analysis.get_array_from_file(path=path, var_name="interflow_slope")
    # slope = np.ma.masked_where(slope <= 0, slope)

    cell_areas = analysis.get_array_from_file(path=path, var_name=infovar.HDF_CELL_AREA_NAME_M2)
    label = "crcm5-hcd-rl-intfl".upper()
    fig.suptitle(label)
    soil_layer_depths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                         1.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    soil_layer_depths = np.array(soil_layer_depths)

    lons, lats, basemap = analysis.get_basemap_from_hdf(file_path=path)
    x, y = basemap(lons, lats)

    # print basemap.proj4string

    slope = maskoceans(lons, lats, slope)
    slope = np.ma.masked_less(slope, 0)

    # create the colormap object
    cmap = cm.get_cmap("spectral_r", lut=10)

    # lake fraction
    ax = fig.add_subplot(gs[0, 0])
    assert isinstance(ax, Axes)
    all_axes = [ax]
    lake_fraction = analysis.get_array_from_file(path=path, var_name="lake_fraction")
    lake_fraction = np.ma.masked_where(slope.mask & (lake_fraction <= 0), lake_fraction)

    _plot_lake_fraction(ax, basemap, x, y, lake_fraction, title="a) Lake fraction", cmap=cmap)

    where_lakes = lake_fraction > 0
    where_glob_lakes = lake_fraction >= 0.6
    where_land = (lake_fraction > 0) | ~slope.mask
    percetage_lakes = np.sum(np.sum(where_glob_lakes.astype(float))) / np.sum(where_lakes.astype(float)) * 100
    print("percentage of global lakes: {0}".format(percetage_lakes))

    #plt.show()

    # slope
    ax = fig.add_subplot(gs[2, 0])
    _plot_slope(ax, basemap, x, y, slope, title="g) River slope", cmap=cmap)
    all_axes.append(ax)

    # slope
    ax = fig.add_subplot(gs[2, 1])
    _plot_slope(ax, basemap, x, y, itf_slope, title="h) Interflow slope", cmap=cmap)
    all_axes.append(ax)


    # depth to bedrock
    ax = fig.add_subplot(gs[0, 1])
    all_axes.append(ax)
    depth_to_bedrock = analysis.get_array_from_file(path=path, var_name="depth_to_bedrock")
    depth_to_bedrock = np.ma.masked_where(slope.mask, depth_to_bedrock)
    _plot_field(ax, basemap, x, y, depth_to_bedrock, title="b) Depth to bedrock (m)", cmap=cmap)



    # sand (calculate mean sand content in the soil above bedrock)
    ax = fig.add_subplot(gs[1, 0])
    sand = analysis.get_array_from_file(path=path, var_name="sand")

    print("sand variable shape: {0} ".format(",".join([str(length) for length in sand.shape])))
    print("layer depths shape: ", soil_layer_depths.shape)
    sand[sand < 0] = 0.0
    sand_height = np.tensordot(soil_layer_depths, sand, axes=(0, 0))
    #sand_height[depth_to_bedrock > 0] /= depth_to_bedrock[depth_to_bedrock > 0]
    #sand_height[depth_to_bedrock <= 0] = 0.0
    sand_height /= soil_layer_depths.sum()
    sand_height = np.ma.masked_where(slope.mask, sand_height)
    _plot_sand_clay_percentages(ax, basemap, x, y, sand_height, title="d) Sand", cmap=cmap)
    all_axes.append(ax)

    # clay
    ax = fig.add_subplot(gs[1, 1])
    clay = analysis.get_array_from_file(path=path, var_name="clay")
    print("clay variable shape: {0} ".format(",".join([str(length) for length in clay.shape])))
    clay[clay < 0] = 0.0
    clay_height = np.tensordot(soil_layer_depths, clay, axes=(0, 0))
    #clay_height[depth_to_bedrock > 0] /= depth_to_bedrock[depth_to_bedrock > 0]
    #clay_height[depth_to_bedrock <= 0] = 0.0
    clay_height = np.ma.masked_where(slope.mask, clay_height)
    clay_height /= soil_layer_depths.sum()
    _plot_sand_clay_percentages(ax, basemap, x, y, clay_height, title="e) Clay", cmap=cmap)
    all_axes.append(ax)

    # drainage density
    ax = fig.add_subplot(gs[2, 2])
    all_axes.append(ax)
    drainage_density = analysis.get_array_from_file(path=path, var_name="drainage_density_inv_meters")
    drainage_density = np.ma.masked_where(slope.mask, drainage_density)
    _plot_field(ax, basemap, x, y, drainage_density * 1000.0, title="i) DD (${\\rm km^{-1}}$)", cmap=cmap)


    # drainage area
    ax = fig.add_subplot(gs[0, 2])
    all_axes.append(ax)
    field = analysis.get_array_from_file(path=path, var_name="accumulation_area_km2")
    field = np.ma.masked_where(slope.mask, field)
    _plot_accumulation_area(ax, basemap, x, y, field, title="c) Drainage area (${\\rm km^{2}}$)", cmap=cmap)


    # vertical hydraulic conductivity
    ax = fig.add_subplot(gs[1, 2])
    all_axes.append(ax)
    field = analysis.get_array_from_file(path=path1, var_name=infovar.HDF_VERT_SOIL_HYDR_COND_NAME)
    field = np.ma.masked_where(slope.mask, field[0, :, :])
    print(field.shape)
    _plot_soil_hydraulic_conductivity(ax, basemap, x, y, field, title="f) Kv, m/s", cmap=cmap)

    #soil anisotropy ratio
    #ax = fig.add_subplot(gs[2, 2])
    #all_axes.append(ax)
    #field = analysis.get_array_from_file(path=path, var_name=infovar.HDF_SOIL_ANISOTROPY_RATIO_NAME)
    #field = np.ma.masked_where(slope.mask, field)
    #_plot_field(ax, basemap, x, y, field, title="i) Soil anisotropy", cmap=cmap)


    for the_ax in all_axes:
        basemap.drawcoastlines(linewidth=common_plot_params.COASTLINE_WIDTH, ax=the_ax)

    figpath = os.path.join(images_folder, "static_fields.jpeg")
    fig.savefig(figpath, dpi=cpp.FIG_SAVE_DPI, bbox_inches="tight")


if __name__ == "__main__":
    #plot_histograms(path="/home/huziy/skynet3_rech1/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_do_not_discard_small.hdf")
    main()

    print(Basemap(projection="npstere", lon_0=-115, boundinglat=60).proj4string)
    print("Hello world")
