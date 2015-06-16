from datetime import datetime
import os
from docutils.nodes import thead

from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm
from matplotlib.dates import MonthLocator, num2date
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, FuncFormatter, ScalarFormatter
from osgeo.ogr import Geometry
from pandas.tseries.converter import _daily_finder
from rpn.rpn import RPN
from matplotlib import cm

from data.cell_manager import CellManager
from util.geo.basemap_info import BasemapInfo
from matplotlib.text import Text

__author__ = 'huziy'

# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crcm5.analyse_hdf.run_config import RunConfig
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis

BASIN_BOUNDARIES_FILE = "data/shape/contour_bv_MRCC/Bassins_MRCC_utm18.shp"

from osgeo import ogr, osr
import numpy as np
from util import plot_utils


class DataToPlot(object):
    def __init__(self, daily_dates=None,
                 simlabel_to_cc_fields=None,
                 basin_name_to_out_indices=None):
        self.daily_dates = daily_dates
        self.simlabel_to_cc_fields = simlabel_to_cc_fields
        self.basin_name_to_out_indices = basin_name_to_out_indices
        self.base_label = ""
        self.modif_label = ""

    def set_base_and_modif_labels(self, lbase, lmodif):
        self.base_label = lbase
        self.modif_label = lmodif

    def _calculate_timeseries(self, field_txy, basin_name_to_mask=None):
        """
        if basin_name_to_mask is None, timeseries at outlet points are returned
        :param field_txy:
        :param basin_name_to_mask:
        """
        result_ts = {}
        for bname, inds in self.basin_name_to_out_indices.items():
            i, j = inds
            if basin_name_to_mask is None:
                result_ts[bname] = field_txy[:, i, j].copy()
            else:
                i_list, j_list = np.where(basin_name_to_mask[bname])
                result_ts[bname] = field_txy[:, i_list, j_list].mean(axis=1)
        return result_ts

    def get_diff_timeseries(self, basin_name_to_mask=None):
        """
        if basin_name_to_mask is None, timeseries at outlet points are returned
        """
        assert self.modif_label is not None and self.base_label is not None, \
            "Simulation labels should be set before doing calculations"

        diff_field = self.simlabel_to_cc_fields[self.modif_label] - self.simlabel_to_cc_fields[self.base_label]
        return self._calculate_timeseries(diff_field, basin_name_to_mask=basin_name_to_mask)

    def get_cc_timeseries(self, sim_label=None, basin_name_to_mask=None):
        """
        if basin_name_to_mask is None, timeseries at outlet points are returned
        :param sim_label:
        :param basin_name_to_mask:
        :return: a dictionary of {basin name: timeseries}
        """
        return self._calculate_timeseries(self.simlabel_to_cc_fields[sim_label],
                                          basin_name_to_mask=basin_name_to_mask)


def read_cc_and_cc_diff(base_configs,
                        modif_configs,
                        name_to_indices=None,
                        varname=None):
    """
    get cc for 2 different configurations and the difference
    work with the area means if basin_name_to_basin_mask is not None
    :param base_configs:
    :param modif_configs:
    :param name_to_indices:
    :param varname:
    :return:
    """
    base_c, base_f = base_configs
    modif_c, modif_f = modif_configs

    # Sasha: custom end year
    end_year_future = base_f.end_year
    end_year_current = base_c.end_year

    level = 0
    daily_dates, data_clim_base_c = analysis.get_daily_climatology(path_to_hdf_file=base_c.data_path,
                                                                   var_name=varname, level=level,
                                                                   start_year=base_c.start_year,
                                                                   end_year=end_year_current)

    _, data_clim_base_f = analysis.get_daily_climatology(path_to_hdf_file=base_f.data_path,
                                                         var_name=varname, level=level,
                                                         start_year=base_f.start_year,
                                                         end_year=end_year_future)

    _, data_clim_modif_c = analysis.get_daily_climatology(path_to_hdf_file=modif_c.data_path,
                                                          var_name=varname, level=level,
                                                          start_year=modif_c.start_year,
                                                          end_year=end_year_current)

    _, data_clim_modif_f = analysis.get_daily_climatology(path_to_hdf_file=modif_f.data_path,
                                                          var_name=varname, level=level,
                                                          start_year=modif_f.start_year,
                                                          end_year=end_year_future)

    delta_modif = data_clim_modif_f - data_clim_modif_c
    delta_base = data_clim_base_f - data_clim_base_c

    sim_label_to_cc_fields = {
        base_c.label: delta_base,
        modif_c.label: delta_modif
    }

    data_to_plot = DataToPlot(daily_dates=daily_dates,
                              simlabel_to_cc_fields=sim_label_to_cc_fields,
                              basin_name_to_out_indices=name_to_indices)

    data_to_plot.set_base_and_modif_labels(base_c.label, modif_c.label)

    return data_to_plot


def calculate_and_plot_climate_change_hydrographs(data_to_plot,
                                                  name_to_out_indices=None,
                                                  months=None, varname=None,
                                                  basin_name_to_basin_mask=None,
                                                  img_path=""):
    assert isinstance(data_to_plot, DataToPlot)

    if months is None:
        months = list(range(1, 13))

    # sort the basins to be from north to south
    items = list(sorted(list(name_to_out_indices.items()), key=lambda item: item[1][1], reverse=True))


    # Get corresponding timeseries, average over basins if required
    delta = data_to_plot.get_diff_timeseries(basin_name_to_mask=basin_name_to_basin_mask)
    cc_base = data_to_plot.get_cc_timeseries(sim_label=data_to_plot.base_label)
    cc_modif = data_to_plot.get_cc_timeseries(sim_label=data_to_plot.modif_label)
    daily_dates = data_to_plot.daily_dates

    # find the maximum value for plots
    vmax = 0
    vmin = 0

    diff_vmax = 0
    diff_vmin = 0
    for name, _ in items:
        vmax = max((vmax, cc_base[name].max(), cc_modif[name].max(), delta[name].max()))
        vmin = min((vmin, cc_base[name].min(), cc_modif[name].min(), delta[name].min()))

        diff_vmax = max(diff_vmax, delta[name].max())
        diff_vmin = min(diff_vmin, delta[name].min())

    plot_utils.apply_plot_params(font_size=12, width_pt=None, width_cm=25, height_cm=12)
    ncols = 3
    nrows = len(items) // ncols + int(len(items) % ncols != 0)
    fig = plt.figure()

    sfmt = ScalarFormatter(useMathText=True, useOffset=False)
    sfmt.set_powerlimits((-2, 2))

    gs = GridSpec(nrows, ncols)
    subplot_count = 0

    line_base, line_modif, line_diff = None, None, None
    ax_last = None
    for name, (i, j) in items:
        print(name)
        print(i, j, np.max(delta[name]))
        row = subplot_count // ncols
        col = subplot_count % ncols

        ax = fig.add_subplot(gs[row, col])
        ax_last = ax
        bbox_props = dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=1, alpha=0.5)
        ax.annotate(name, (0.9, 0.1), xycoords="axes fraction", bbox=bbox_props, zorder=10,
                    alpha=0.5, horizontalalignment="right", verticalalignment="bottom")

        # line_base = ax.plot(daily_dates, cc_base[name].copy(), "b", label="{}".format(data_to_plot.base_label), lw=2)
        #
        #
        # line_modif = ax.plot(daily_dates, cc_modif[name].copy(),
        #                      color="r", label="{}".format(data_to_plot.modif_label),
        #                      zorder=5, lw=2)

        # Plot monthly
        monthly_dates = [datetime(2001, m, 1) for m in range(1, 13)]
        monthly_base = [np.mean([v for d, v in zip(daily_dates, cc_base[name]) if d.month == m]) for m in range(1, 13)]
        monthly_modif = [np.mean([v for d, v in zip(daily_dates, cc_modif[name]) if d.month == m]) for m in
                         range(1, 13)]

        line_base = ax.plot(monthly_dates, monthly_base, "b", label="{}".format(data_to_plot.base_label), lw=2)
        line_modif = ax.plot(monthly_dates, monthly_modif, color="r", label="{}".format(data_to_plot.modif_label),
                             zorder=5, lw=2)

        for tl in ax.get_yticklabels():
            tl.set_color("b")

        assert isinstance(ax, Axes)
        coef_text = ax.yaxis.get_offset_text()
        assert isinstance(coef_text, Text)
        coef_text.set_color("b")

        ax_twin = ax.twinx()
        assert isinstance(ax_twin, Axes)
        ax_twin.yaxis.get_offset_text().set_color("g")

        for tl in ax_twin.get_yticklabels():
            tl.set_color('g')

        # line_diff = ax_twin.plot(daily_dates, delta[name].copy(), "g--",
        #                          label="({})-({})".format(data_to_plot.modif_label, data_to_plot.base_label),
        #                          lw=3)


        # Plot monthly
        monthly_dates = [datetime(2001, m, 1) for m in range(1, 13)]
        monthly_delta = [np.mean([v for d, v in zip(daily_dates, delta[name]) if d.month == m]) for m in range(1, 13)]
        line_diff = ax_twin.plot(monthly_dates, monthly_delta, "g--",
                                 label="({})-({})".format(data_to_plot.modif_label, data_to_plot.base_label),
                                 lw=3)

        ax_twin.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
        # ax_twin.set_ylim(bottom=diff_vmin, top=diff_vmax)
        # ax_twin.invert_yaxis()
        ax_twin.yaxis.set_major_formatter(sfmt)


        # ax.grid()
        subplot_count += 1
        ax.yaxis.set_major_formatter(sfmt)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
        # ax.set_ylim(bottom=vmin, top=vmax * 1.2)

        ax.xaxis.set_major_formatter(FuncFormatter(format_day_tick_labels))

        # ax.xaxis.set_minor_locator(DayLocator(interval=5))
        ax.xaxis.set_major_locator(MonthLocator())
        ax.grid()

    the_labels = (data_to_plot.base_label, data_to_plot.modif_label)
    ax_last.legend((line_base[0], line_modif[0], line_diff[0]),
                   (the_labels[0], the_labels[1], "{} vs {}".format(*the_labels[::-1])),
                   bbox_to_anchor=(1, -0.2), loc="upper right", borderaxespad=0, ncol=len(the_labels) + 1)

    plt.tight_layout()
    print("Saving the plot to {}".format(img_path))
    fig.savefig(img_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def format_day_tick_labels(d, pos=None):
    d = num2date(d)
    return d.strftime("%b")[0]


def is_part_of_points_in(basin, x_list, y_list, the_part=0.5):
    assert isinstance(basin, Geometry)
    n_total = len(x_list)
    n_inside = 0
    for x, y in zip(x_list, y_list):
        n_inside += int(basin.Contains(ogr.CreateGeometryFromWkt("POINT ({} {})".format(x, y))))
        if n_inside > 5:
            return True

    return float(n_inside) / float(n_total) > the_part


def get_basin_to_outlet_indices_map(shape_file=BASIN_BOUNDARIES_FILE, bmp_info=None,
                                    directions=None, accumulation_areas=None,
                                    lake_fraction_field=None):
    assert isinstance(bmp_info, BasemapInfo)

    driver = ogr.GetDriverByName("ESRI Shapefile")
    print(driver)
    ds = driver.Open(shape_file, 0)

    assert isinstance(ds, ogr.DataSource)
    layer = ds.GetLayer()

    assert isinstance(layer, ogr.Layer)
    print(layer.GetFeatureCount())

    latlong_proj = osr.SpatialReference()
    latlong_proj.ImportFromEPSG(4326)

    utm_proj = layer.GetSpatialRef()

    # create Coordinate Transformation
    coord_transform = osr.CoordinateTransformation(latlong_proj, utm_proj)

    utm_coords = coord_transform.TransformPoints(list(zip(bmp_info.lons.flatten(), bmp_info.lats.flatten())))
    utm_coords = np.asarray(utm_coords)
    x_utm = utm_coords[:, 0].reshape(bmp_info.lons.shape)
    y_utm = utm_coords[:, 1].reshape(bmp_info.lons.shape)

    basin_mask = np.zeros_like(bmp_info.lons)
    cell_manager = CellManager(directions, accumulation_area_km2=accumulation_areas,
                               lons2d=bmp_info.lons, lats2d=bmp_info.lats)

    index = 1
    basins = []
    basin_names = []
    basin_name_to_mask = {}
    for feature in layer:
        assert isinstance(feature, ogr.Feature)
        # print feature["FID"]

        geom = feature.GetGeometryRef()
        assert isinstance(geom, ogr.Geometry)


        basins.append(ogr.CreateGeometryFromWkb(geom.ExportToWkb()))
        basin_names.append(feature["abr"])

    accumulation_areas_temp = accumulation_areas[:, :]
    lons_out, lats_out = [], []
    basin_names_out = []
    name_to_ij_out = {}

    min_basin_area = min(b.GetArea() * 1.0e-6 for b in basins)

    while len(basins):
        fm = np.max(accumulation_areas_temp)

        i, j = np.where(fm == accumulation_areas_temp)
        i, j = i[0], j[0]
        p = ogr.CreateGeometryFromWkt("POINT ({} {})".format(x_utm[i, j], y_utm[i, j]))
        b_selected = None
        name_selected = None
        for name, b in zip(basin_names, basins):

            assert isinstance(b, ogr.Geometry)
            assert isinstance(p, ogr.Geometry)
            if b.Contains(p.Buffer(2000 * 2 ** 0.5)):
                # Check if there is an upstream cell from the same basin
                the_mask = cell_manager.get_mask_of_cells_connected_with_by_indices(i, j)

                # Save the mask of the basin for future use
                basin_name_to_mask[name] = the_mask

                # if is_part_of_points_in(b, x_utm[the_mask == 1], y_utm[the_mask == 1]):
                # continue


                b_selected = b
                name_selected = name
                # basin_names_out.append(name)

                lons_out.append(bmp_info.lons[i, j])
                lats_out.append(bmp_info.lats[i, j])
                name_to_ij_out[name] = (i, j)

                basin_mask[the_mask == 1] = index
                index += 1

                break

        if b_selected is not None:
            basins.remove(b_selected)
            basin_names.remove(name_selected)
            outlet_index_in_basin = 1
            current_basin_name = name_selected
            while current_basin_name in basin_names_out:
                current_basin_name = name_selected + str(outlet_index_in_basin)
                outlet_index_in_basin += 1

            basin_names_out.append(current_basin_name)
            print(len(basins), basin_names_out)

        accumulation_areas_temp[i, j] = -1

    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=20, height_cm=12)
    gs = GridSpec(1, 2, width_ratios=[1.0, 0.5], wspace=0.01)
    fig = plt.figure()

    ax = fig.add_subplot(gs[0, 0])
    xx, yy = bmp_info.get_proj_xy()
    # im = bmp.pcolormesh(xx, yy, basin_mask.reshape(xx.shape))
    bmp_info.basemap.drawcoastlines(linewidth=0.5, ax=ax)
    bmp_info.basemap.drawrivers(zorder=5, color="0.5", ax=ax)
    # bmp.colorbar(im)


    xs, ys = bmp_info.basemap(lons_out, lats_out)
    # bmp.scatter(xs, ys, c="r", s=40, zorder=2)

    cmap = cm.get_cmap("rainbow", index - 1)
    bn = BoundaryNorm(list(range(index + 1)), index - 1)

    basin_mask = np.ma.masked_where(basin_mask < 0.5, basin_mask)
    bmp_info.basemap.pcolormesh(xx, yy, basin_mask, norm=bn, cmap=cmap, ax=ax)

    for name, xa, ya, lona, lata in zip(basin_names_out, xs, ys, lons_out, lats_out):

        text_offset = (-20, 20) if name not in ["GEO", ] else (30, 20)

        if name in ["ARN"]:
            text_offset = (-10, 30)

        if name in ["FEU"]:
            text_offset = (5, 50)

        if name in ["CAN"]:
            text_offset = (-75, 50)

        if name in ["MEL"]:
            text_offset = (20, 40)

        if name in ["PYR"]:
            text_offset = (60, 60)

        if name in ["BAL", ]:
            text_offset = (50, 30)

        if name in ["BEL"]:
            text_offset = (-20, -10)

        if name in ["RDO", "STM", "SAG", ]:
            text_offset = (50, -50)

        if name in ["BOM", ]:
            text_offset = (20, -20)

        if name in ["MOI", ]:
            text_offset = (30, -20)

        if name in ["ROM", ]:
            text_offset = (40, -20)

        if name in ["RDO", ]:
            text_offset = (30, -30)

        if name in ["CHU", "NAT"]:
            text_offset = (40, 40)

        if name in ["MAN", ]:
            text_offset = (55, -45)

        ax.annotate(name, xy=(xa, ya), xytext=text_offset,
                    textcoords='offset points', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white'),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                    font_properties=FontProperties(size=8), zorder=20)

    bmp_info.basemap.readshapefile(".".join(BASIN_BOUNDARIES_FILE.split(".")[:-1]).replace("utm18", "latlon"), "basin",
                                   linewidth=1.2, ax=ax)



    # Plot zonally averaged lake fraction
    ax = fig.add_subplot(gs[0, 1])
    ydata = range(lake_fraction_field.shape[1])
    ax.plot(lake_fraction_field.mean(axis=0) * 100, ydata, lw=2)

    ax.fill_betweenx(ydata, lake_fraction_field.mean(axis=0) * 100, alpha=0.5)

    ax.set_xlabel("Lake fraction (%)")
    ax.set_ylim(min(ydata), max(ydata))
    ax.xaxis.set_tick_params(direction='out', width=1)
    ax.yaxis.set_tick_params(direction='out', width=1)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    for tl in ax.yaxis.get_ticklabels():
        tl.set_visible(False)

    fig.savefig("qc_basin_outlets_points.png", bbox_inches="tight")
    # plt.show()
    plt.close(fig)

    return name_to_ij_out, basin_name_to_mask


def get_image_path(base_c, base_f, modif_c, modif_f, varname):
    """
    Generate image path based on the compared experimants and periods
    :param base_c:
    :param base_f:
    :param modif_c:
    :param modif_f:
    :param varname:
    :return:
    """
    from pathlib import Path

    the_labels = (modif_c.label, base_c.label)

    img_folder = Path("cc_paper").joinpath("{}_vs_{}".format(*the_labels))

    # Generate the path to the image
    if not img_folder.exists():
        img_folder.mkdir(parents=True)

    img_name = "plot1d_{}-{}_{}-{}_{}.png".format(base_f.start_year, base_f.end_year,
                                                  base_c.start_year, base_c.end_year,
                                                  varname)


    # check that the future period is the same in the base and modified configurations
    assert modif_f.start_year == base_f.start_year
    assert modif_f.end_year == base_f.end_year
    return str(img_folder.joinpath(img_name))


def main_interflow():
    base_current_path = \
        "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"
    base_label = "CRCM5-L"

    modif_current_path = \
        "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-intfl-cc-canesm2-1980-2010.hdf5"
    modif_label = "CRCM5-LI"

    start_year_c = 1980
    end_year_c = 2010

    future_shift_years = 75

    params = dict(
        data_path=base_current_path, start_year=start_year_c, end_year=end_year_c, label=base_label)

    geo_data_file = "/skynet3_rech1/huziy/hdf_store/pm1979010100_00000000p"

    base_config_c = RunConfig(**params)
    base_config_f = base_config_c.get_shifted_config(future_shift_years)

    base_configs = [base_config_c, base_config_f]

    params.update(dict(
        data_path=modif_current_path, label=modif_label))

    modif_config_c = RunConfig(**params)
    modif_config_f = modif_config_c.get_shifted_config(future_shift_years)
    modif_configs = [modif_config_c, modif_config_f]

    r_obj = RPN(geo_data_file)
    facc = r_obj.get_first_record_for_name("FAA")
    fldr = r_obj.get_first_record_for_name("FLDR")
    lake_fraction = r_obj.get_first_record_for_name_and_level("ML")
    # mask ocean points
    lake_fraction = np.ma.masked_where((fldr <= 0) | (fldr > 128), lake_fraction)


    bmp_info = analysis.get_basemap_info_from_hdf(file_path=base_current_path)

    basin_name_to_out_indices_map, basin_name_to_basin_mask = get_basin_to_outlet_indices_map(bmp_info=bmp_info,
                                                                                              accumulation_areas=facc,
                                                                                              directions=fldr,
                                                                                              lake_fraction_field=lake_fraction)

    varname = "STFL"

    basin_name_to_basin_mask = basin_name_to_basin_mask if varname not in ["STFA", "STFL"] else None

    data_to_plot = read_cc_and_cc_diff(base_configs, modif_configs,
                                       name_to_indices=basin_name_to_out_indices_map, varname=varname)

    img_path = get_image_path(base_config_c, base_config_f, modif_config_c, modif_config_f, varname)

    # select lake rich basins
    sel_basins = ["ARN", "PYR", "LGR", "RDO", "SAG", "WAS"]
    basin_name_to_out_indices_map = {k: v for k, v in basin_name_to_out_indices_map.items() if k in sel_basins}

    calculate_and_plot_climate_change_hydrographs(data_to_plot,
                                                  name_to_out_indices=basin_name_to_out_indices_map,
                                                  months=list(range(1, 13)), varname=varname,
                                                  img_path=img_path, basin_name_to_basin_mask=basin_name_to_basin_mask)


def main():
    base_current_path = \
        "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-r-cc-canesm2-1980-2010.hdf5"
    base_label = "CRCM5-NL"

    modif_current_path = \
        "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"
    modif_label = "CRCM5-L"

    start_year_c = 1980
    end_year_c = 2010

    varname = "STFL"

    future_shift_years = 75

    params = dict(
        data_path=base_current_path, start_year=start_year_c, end_year=end_year_c, label=base_label)

    geo_data_file = "/skynet3_rech1/huziy/hdf_store/pm1979010100_00000000p"

    base_config_c = RunConfig(**params)
    base_config_f = base_config_c.get_shifted_config(future_shift_years)

    base_configs = [base_config_c, base_config_f]

    params.update(dict(
        data_path=modif_current_path, label=modif_label))

    modif_config_c = RunConfig(**params)
    modif_config_f = modif_config_c.get_shifted_config(future_shift_years)
    modif_configs = [modif_config_c, modif_config_f]

    r_obj = RPN(geo_data_file)
    facc = r_obj.get_first_record_for_name("FAA")
    fldr = r_obj.get_first_record_for_name("FLDR")
    lake_fraction = r_obj.get_first_record_for_name_and_level("ML")
    lake_fraction = np.ma.masked_where((fldr <= 0) | (fldr > 128), lake_fraction)


    bmp_info = analysis.get_basemap_info_from_hdf(file_path=base_current_path, resolution="f")

    basin_name_to_out_indices_map, basin_name_to_basin_mask = get_basin_to_outlet_indices_map(bmp_info=bmp_info,
                                                                                              accumulation_areas=facc,
                                                                                              directions=fldr,
                                                                                              lake_fraction_field=lake_fraction)

    # TODO: Remove the throwing exception (added only temporarily)
    if True:
        raise Exception("I was put here in order to plot basins and stop")




    data_to_plot = read_cc_and_cc_diff(base_configs, modif_configs,
                                       name_to_indices=basin_name_to_out_indices_map, varname=varname)

    basin_name_to_basin_mask = basin_name_to_basin_mask if varname not in ["STFA", "STFL"] else None


    img_path = get_image_path(base_config_c, base_config_f, modif_config_c, modif_config_f, varname)


    # select lake rich basins
    sel_basins = ["ARN", "PYR", "LGR", "RDO", "SAG", "WAS"]
    basin_name_to_out_indices_map = {k: v for k, v in basin_name_to_out_indices_map.items() if k in sel_basins}

    calculate_and_plot_climate_change_hydrographs(data_to_plot,
                                                  name_to_out_indices=basin_name_to_out_indices_map,
                                                  months=list(range(1, 13)), varname=varname,
                                                  img_path=img_path)


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()

    main()
    # main_interflow()
