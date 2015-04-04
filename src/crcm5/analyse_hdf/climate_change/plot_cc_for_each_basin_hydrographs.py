import os
from matplotlib.axes import Axes

from matplotlib.colors import BoundaryNorm

from matplotlib.dates import DateFormatter, MonthLocator, DayLocator, num2date
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, FuncFormatter, LinearLocator, ScalarFormatter
from osgeo.ogr import Geometry
from rpn.rpn import RPN
from matplotlib import cm

from data.cell_manager import CellManager

__author__ = 'huziy'

# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crcm5.analyse_hdf.run_config import RunConfig
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis

BASIN_BOUNDARIES_FILE = "data/shape/contour_bv_MRCC/Bassins_MRCC_utm18.shp"

from osgeo import ogr, osr
import numpy as np
from util import plot_utils


def calculate_and_plot_climate_change_hydrographs(base_configs,
                                                  modif_configs,
                                                  name_to_indices=None,
                                                  months=None, varname=None,
                                                  basin_name_to_basin_mask=None):
    base_c, base_f = base_configs
    modif_c, modif_f = modif_configs

    if months is None:
        months = list(range(1, 13))

    assert isinstance(base_c, RunConfig)

    level = 0
    daily_dates, stfa_clim_base_c = analysis.get_daily_climatology(path_to_hdf_file=base_c.data_path,
                                                                   var_name=varname, level=level,
                                                                   start_year=base_c.start_year,
                                                                   end_year=base_c.end_year)

    _, stfa_clim_base_f = analysis.get_daily_climatology(path_to_hdf_file=base_f.data_path,
                                                         var_name=varname, level=level,
                                                         start_year=base_f.start_year,
                                                         end_year=base_f.end_year)

    _, stfa_clim_modif_c = analysis.get_daily_climatology(path_to_hdf_file=modif_c.data_path,
                                                          var_name=varname, level=level,
                                                          start_year=modif_c.start_year,
                                                          end_year=modif_c.end_year)

    _, stfa_clim_modif_f = analysis.get_daily_climatology(path_to_hdf_file=modif_f.data_path,
                                                          var_name=varname, level=level,
                                                          start_year=modif_f.start_year,
                                                          end_year=modif_f.end_year)

    delta_modif = stfa_clim_modif_f - stfa_clim_modif_c

    delta_modif = np.asarray([delta_modif[ti, :, :] for ti, t in enumerate(daily_dates)
                              if t.month in months])

    delta_base = stfa_clim_base_f - stfa_clim_base_c
    delta_base = np.asarray([delta_base[ti, :, :] for ti, t in enumerate(daily_dates)
                             if t.month in months])



    daily_dates = [d for d in daily_dates if d.month in months]

    delta = delta_modif - delta_base


    items = list(sorted(list(name_to_indices.items()), key=lambda item: item[1][1], reverse=True))


    # find the maximum value for plots
    vmax = 0
    for name, (i, j) in items:
        vmax = max((vmax, delta_base[:, i, j].max(), delta_modif[:, i, j].max(), delta[:, i, j].max()))



    plot_utils.apply_plot_params(font_size=16, width_pt=None, width_cm=35, height_cm=45)
    ncols = 4
    nrows = len(items) // ncols + int(len(items) % ncols != 0)
    fig = plt.figure()

    sfmt = ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((-2, 2))


    gs = GridSpec(nrows, ncols)
    subplot_count = 0

    line_base, line_modif, line_diff = None, None, None
    ax_last = None
    for name, (i, j) in items:
        print(name)
        print(i, j, np.max(delta))
        row = subplot_count // ncols
        col = subplot_count % ncols

        ax = fig.add_subplot(gs[row, col])
        ax_last = ax
        bbox_props = dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=1)
        ax.annotate(name, (0.8, 0.5), xycoords="axes fraction", bbox=bbox_props, zorder=10, alpha=0.5)

        line_base = ax.plot(daily_dates, delta_base[:, i, j], "b", label="{}".format(base_c.label), lw=2)
        line_modif = ax.plot(daily_dates, delta_modif[:, i, j], "r", label="{}".format(modif_c.label), lw=2)

        for tl in ax.get_yticklabels():
            tl.set_color("r")

        ax_twin = ax.twinx()
        assert isinstance(ax_twin, Axes)

        for tl in ax_twin.get_yticklabels():
            tl.set_color('g')

        line_diff = ax_twin.plot(daily_dates, delta[:, i, j], "g",
                                 label="({})-({})".format(modif_c.label, base_c.label),
                                 lw=3)

        ax_twin.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ymin, ymax = ax_twin.get_ylim()
        ax_twin.set_ylim(bottom=ymin, top=vmax * 1.2)
        # ax_twin.invert_yaxis()
        ax_twin.yaxis.set_major_formatter(sfmt)


        # ax.grid()
        subplot_count += 1
        ax.yaxis.set_major_formatter(sfmt)


        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=ymin, top=vmax * 1.2)

        if row == nrows - 1:
            ax.xaxis.set_major_formatter(FuncFormatter(format_day_tick_labels))
        else:
            for tl in ax.get_xticklabels():
                tl.set_visible(False)

        # ax.xaxis.set_minor_locator(DayLocator(interval=5))
        ax.xaxis.set_major_locator(MonthLocator(bymonth=list(range(1, 13, 2))))

    ax_last.legend((line_base[0], line_modif[0], line_diff[0]),
                   (base_c.label, modif_c.label, "{} vs {}".format(modif_c.label, base_c.label)),
                   bbox_to_anchor=(1.2, 1), loc=2)
    # Save the image to the file
    img_folder = os.path.join("cc_paper", "{}_vs_{}".format(modif_c.label, base_c.label))
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)

    img_name = "{}-{}_{}-{}_{}.pdf".format(base_f.start_year, base_f.end_year,
                                           base_c.start_year, base_c.end_year,
                                           varname)

    img_path = os.path.join(img_folder, img_name)
    plt.tight_layout()
    fig.savefig(img_path, bbox_inches="tight")


def format_day_tick_labels(d, pos=None):
    d = num2date(d)

    if d.day == 1 or pos == 0:
        if pos == 0:
            return "\n{:%b}".format(d)

        return "{:%d}\n{:%b}".format(d, d)
    else:
        if (d.day - 1) % 31 != 0:
            return ""

        return "{:%d}".format(d)


def is_part_of_points_in(basin, x_list, y_list, the_part=0.5):
    assert isinstance(basin, Geometry)
    n_total = len(x_list)
    n_inside = 0
    for x, y in zip(x_list, y_list):
        n_inside += int(basin.Contains(ogr.CreateGeometryFromWkt("POINT ({} {})".format(x, y))))
        if n_inside > 5:
            return True

    return float(n_inside) / float(n_total) > the_part


def get_basin_to_outlet_indices_map(shape_file=BASIN_BOUNDARIES_FILE, lons=None, lats=None, bmp=None,
                                    directions=None, accumulation_areas=None):
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

    utm_coords = coord_transform.TransformPoints(list(zip(lons.flatten(), lats.flatten())))
    utm_coords = np.asarray(utm_coords)
    x_utm = utm_coords[:, 0].reshape(lons.shape)
    y_utm = utm_coords[:, 1].reshape(lons.shape)

    basin_mask = np.zeros_like(lons)
    cell_manager = CellManager(directions, accumulation_area_km2=accumulation_areas, lons2d=lons, lats2d=lats)

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

                lons_out.append(lons[i, j])
                lats_out.append(lats[i, j])
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

    fig = plt.figure()
    xx, yy = bmp(lons, lats)
    # im = bmp.pcolormesh(xx, yy, basin_mask.reshape(xx.shape))
    bmp.drawcoastlines(linewidth=0.5)
    bmp.drawrivers(zorder=5, color="w")
    # bmp.colorbar(im)


    xs, ys = bmp(lons_out, lats_out)
    bmp.scatter(xs, ys, c="r", s=40, zorder=2)

    cmap = cm.get_cmap("rainbow", index - 1)
    bn = BoundaryNorm(list(range(index)), index - 1)

    bmp.pcolormesh(xx, yy, basin_mask, norm=bn, cmap=cmap)

    for name, xa, ya, lona, lata in zip(basin_names_out, xs, ys, lons_out, lats_out):
        text_offset = (-20, 20) if name not in ["WAS", "GRB", "GEO", "BAL", "MEL", "NAT"] else (20, 20)
        plt.annotate(name, xy=(xa, ya), xytext=text_offset,
                     textcoords='offset points', ha='right', va='bottom', font_properties=FontProperties(size=10),
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow'),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    bmp.readshapefile(BASIN_BOUNDARIES_FILE.split(".")[0].replace("utm18", "latlon"), "basin")

    fig.savefig("qc_basin_outlets_points.pdf", bbox_inches="tight")
    # plt.show()

    return name_to_ij_out, basin_name_to_mask


def main_interflow():
    base_current_path = \
        "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"
    base_label = "CRCM5-L2"

    modif_current_path = \
        "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-intfl-cc-canesm2-1980-2010.hdf5"
    modif_label = "CRCM5-L2I"

    start_year_c = 1980
    end_year_c = 2010

    future_shift_years = 75

    params = dict(
        data_path=base_current_path, start_year=start_year_c, end_year=end_year_c, label=base_label
    )

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

    lons, lats, bmp = analysis.get_basemap_from_hdf(file_path=base_current_path)

    basin_name_to_indexes_map, basin_name_to_basin_mask = get_basin_to_outlet_indices_map(lons=lons, lats=lats, bmp=bmp,
                                                                                          accumulation_areas=facc,
                                                                                          directions=fldr)

    varname = "STFA"
    calculate_and_plot_climate_change_hydrographs(base_configs, modif_configs,
                                                  name_to_indices=basin_name_to_indexes_map,
                                                  months=list(range(1, 13)), varname=varname)

    # plt.show()
    # b = Basemap()
    # b.readshapefile(BASIN_BOUNDARIES_FILE, "basin")
    # print b.basin_info
    #
    # b.drawcoastlines()
    #
    # plt.show()
    #

    pass


def main():
    base_current_path = \
        "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-r-cc-canesm2-1980-2010.hdf5"
    base_label = "CRCM5-NL"

    modif_current_path = \
        "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"
    modif_label = "CRCM5-L2"

    start_year_c = 1980
    end_year_c = 2010

    varname = "PR"

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

    lons, lats, bmp = analysis.get_basemap_from_hdf(file_path=base_current_path)

    basin_name_to_indexes_map = get_basin_to_outlet_indices_map(lons=lons, lats=lats, bmp=bmp,
                                                                accumulation_areas=facc,
                                                                directions=fldr)

    calculate_and_plot_climate_change_hydrographs(base_configs, modif_configs, varname=varname,
                                                  name_to_indices=basin_name_to_indexes_map)

    print(base_config_f)
    # bmp.readshapefile(BASIN_BOUNDARIES_FILE.split(".")[0].replace("utm18", "latlon"), "basin")

    # plt.show()
    # b = Basemap()
    # b.readshapefile(BASIN_BOUNDARIES_FILE, "basin")
    # print b.basin_info
    #
    # b.drawcoastlines()
    #
    # plt.show()
    #

    pass


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()

    # main()
    main_interflow()