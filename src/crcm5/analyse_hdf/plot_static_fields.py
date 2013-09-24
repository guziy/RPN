from brewer2mpl import brewer2mpl
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.figure import Figure
from matplotlib.ticker import LinearLocator, MultipleLocator, LogLocator
from mpl_toolkits.basemap import maskoceans
import os
from crcm5 import infovar
from crcm5.analyse_hdf import common_plot_params

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt
import do_analysis_using_pytables as analysis
import common_plot_params as cpp


images_folder = "/home/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/images_for_lake-river_paper"

def _plot_lake_fraction(ax, basemap, x, y, field, title = "", cmap = None):
    ax.set_title(title)
    if cmap is not None:
        bn = BoundaryNorm(np.arange(0,1.1, 0.1), cmap.N)
        im = basemap.pcolormesh(x, y, field, ax = ax, cmap=cmap, norm = bn, zorder = 5)
        basemap.colorbar(im)
    else:
        im = basemap.pcolormesh(x, y, field, ax = ax)
        basemap.colorbar(im)
    basemap.drawcoastlines(ax = ax)

def _plot_sand_clay_percentages(ax, basemap, x, y, field, title = "", cmap = None):
    ax.set_title(title)
    if cmap is not None:
        bn = BoundaryNorm(np.arange(0,110, 10), cmap.N)
        im = basemap.pcolormesh(x, y, field, ax = ax, cmap=cmap, norm = bn)
        basemap.colorbar(im)
    else:
        im = basemap.pcolormesh(x, y, field, ax = ax)
        basemap.colorbar(im)
    basemap.drawcoastlines(ax = ax)


def _plot_slope(ax, basemap, x, y, field, title = "", cmap = None):
    ax.set_title(title)
    if cmap is not None:
        the_norm = LogNorm(vmin=1.0e-5, vmax = 1)
        im = basemap.pcolormesh(x, y, field, ax = ax, cmap=cmap, norm = the_norm)
        basemap.colorbar(im)
    else:
        im = basemap.pcolormesh(x, y, field, ax = ax)
        basemap.colorbar(im)
    basemap.drawcoastlines(ax = ax)



def _plot_accumulation_area(ax, basemap, x, y, field, title = "", cmap = None):
    ax.set_title(title)
    if cmap is not None:
        field = np.ma.masked_where(field <= 0, field)
        norm = LogNorm(vmin=10, vmax=1e6)
        im = basemap.pcolormesh(x, y, field, ax = ax, cmap=cmap, norm = norm)
        basemap.colorbar(im)
    else:
        im = basemap.pcolormesh(x, y, field, ax = ax)
        basemap.colorbar(im, format = "%.1f")
    basemap.drawcoastlines(ax = ax)


def _plot_field(ax, basemap, x, y, field, title = "", cmap = None):
    ax.set_title(title)
    if cmap is not None:
        im = basemap.pcolormesh(x, y, field, ax = ax, cmap=cmap)
        basemap.colorbar(im, ticks = LinearLocator(numticks=cmap.N + 1), format="%.1f")
    else:
        im = basemap.pcolormesh(x, y, field, ax = ax)
        basemap.colorbar(im, format = "%.1f")
    basemap.drawcoastlines(ax = ax)

def main():
    fig = plt.figure()
    assert isinstance(fig, Figure)
    gs = gridspec.GridSpec(3, 3, wspace=0.4)

    #plot the control

    path = "/home/huziy/skynet3_rech1/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup3.hdf"
    slope = analysis.get_array_from_file(path=path, var_name="slope")
    #slope = np.ma.masked_where(slope <= 0, slope)


    label = "crcm5-hcd-rl-intfl".upper()
    fig.suptitle(label)
    soil_layer_depths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                         1.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    soil_layer_depths = np.array(soil_layer_depths)

    lons, lats, basemap = analysis.get_basemap_from_hdf(file_path=path)
    x, y = basemap(lons, lats)

    slope = maskoceans(lons, lats, slope)
    slope = np.ma.masked_less(slope, 0)





    #create the colormap object
    cmap = brewer2mpl.get_map("spectral", "diverging", 9, reverse = True).get_mpl_colormap(N = 10)

    #lake fraction
    ax = fig.add_subplot(gs[0, 0])
    assert isinstance(ax, Axes)
    all_axes = [ax]
    lake_fraction = analysis.get_array_from_file(path=path, var_name="lake_fraction")
    lake_fraction = np.ma.masked_where(lake_fraction <= 0, lake_fraction)

    _plot_lake_fraction(ax, basemap, x, y, lake_fraction, title="Lake fraction", cmap=cmap)


    #slope
    ax = fig.add_subplot(gs[2, 0])
    _plot_slope(ax, basemap, x, y, slope, title="Slope", cmap=cmap)



    #depth to bedrock
    ax = fig.add_subplot(gs[0,1])
    all_axes.append(ax)
    depth_to_bedrock = analysis.get_array_from_file(path=path, var_name="depth_to_bedrock")
    depth_to_bedrock = np.ma.masked_where(slope.mask, depth_to_bedrock)
    _plot_field(ax, basemap, x, y, depth_to_bedrock, title="Depth to bedrock (m)", cmap = cmap)




    #sand (calculate mean sand content in the soil above bedrock)
    ax = fig.add_subplot(gs[1, 0])
    sand = analysis.get_array_from_file(path=path, var_name="sand")
    print "sand variable shape: {0} ".format(",".join([str(length) for length in sand.shape]))
    print "layer depths shape: ", soil_layer_depths.shape
    sand[sand < 0] = 0.0
    sand_height = np.tensordot(soil_layer_depths, sand, axes=(0, 0))
    #sand_height[depth_to_bedrock > 0] /= depth_to_bedrock[depth_to_bedrock > 0]
    #sand_height[depth_to_bedrock <= 0] = 0.0
    sand_height /= soil_layer_depths.sum()
    sand_height = np.ma.masked_where(slope.mask, sand_height)
    _plot_sand_clay_percentages(ax, basemap, x, y, sand_height, title="Sand", cmap=cmap)

    #clay
    ax = fig.add_subplot(gs[1, 1])
    clay = analysis.get_array_from_file(path=path, var_name="clay")
    print "clay variable shape: {0} ".format(",".join([str(length) for length in clay.shape]))
    clay[clay < 0] = 0.0
    clay_height = np.tensordot(soil_layer_depths, clay, axes=(0,0))
    #clay_height[depth_to_bedrock > 0] /= depth_to_bedrock[depth_to_bedrock > 0]
    #clay_height[depth_to_bedrock <= 0] = 0.0
    clay_height = np.ma.masked_where(slope.mask, clay_height)
    clay_height /= soil_layer_depths.sum()
    _plot_sand_clay_percentages(ax, basemap, x, y, clay_height, title="Clay", cmap=cmap)


    #drainage density
    ax = fig.add_subplot(gs[2,1])
    all_axes.append(ax)
    drainage_density = analysis.get_array_from_file(path=path, var_name="drainage_density_inv_meters")
    drainage_density = np.ma.masked_where(slope.mask, drainage_density)
    _plot_field(ax, basemap, x, y, drainage_density * 1000.0, title="Drainage density (${\\rm km^{-1}}$)", cmap = cmap)


    #drainage area
    ax = fig.add_subplot(gs[0,2])
    all_axes.append(ax)
    field = analysis.get_array_from_file(path=path, var_name="accumulation_area_km2")
    field = np.ma.masked_where(slope.mask, field)
    _plot_accumulation_area(ax, basemap, x, y, field, title="Drainage area (${\\rm km^{2}}$)", cmap = cmap)



    for the_ax in all_axes:
        basemap.drawcoastlines(linewidth=common_plot_params.COASTLINE_WIDTH, ax = the_ax)

    figPath = os.path.join(images_folder, "static_fields.jpeg")
    fig.savefig(figPath, dpi=cpp.FIG_SAVE_DPI, bbox_inches = "tight")

if __name__ == "__main__":
    main()
    print "Hello world"
  