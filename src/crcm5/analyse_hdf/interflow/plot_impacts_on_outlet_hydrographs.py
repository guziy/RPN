from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.dates import num2date, MonthLocator
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from osgeo import ogr
from osgeo import osr
import numpy as np
from pathlib import Path
from rpn.rpn import RPN
from crcm5.analyse_hdf.run_config import RunConfig
from data.cell_manager import CellManager
import matplotlib.pyplot as plt
import crcm5.analyse_hdf.do_analysis_using_pytables as analysis
from util import plot_utils

__author__ = 'huziy'

BASIN_BOUNDARIES_FILE = "data/shape/contour_bv_MRCC/Bassins_MRCC_utm18.shp"
GEO_DATA_FILE = "/skynet3_rech1/huziy/hdf_store/pm1979010100_00000000p"

IMG_FOLDER = Path("impact_of_interflow/hydrographs")


def get_basin_to_outlet_indices_map(shape_file=BASIN_BOUNDARIES_FILE, lons=None, lats=None,
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
                the_mask = cell_manager.get_mask_of_upstream_cells_connected_with_by_indices(i, j)

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

    return name_to_ij_out, basin_name_to_mask


def plot_hydrographs():
    plot_utils.apply_plot_params(font_size=14, width_pt=None, width_cm=20, height_cm=20)
    start_year = 1980
    end_year = 2010

    varname = "STFL"

    base_config = RunConfig(start_year=start_year, end_year=end_year,
                            data_path="/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5",
                            label="NI")

    modif_config = RunConfig(start_year=start_year, end_year=end_year,
                             data_path="/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5",
                             label="WI")


    r_obj = RPN(GEO_DATA_FILE)
    facc = r_obj.get_first_record_for_name("FAA")
    fldr = r_obj.get_first_record_for_name("FLDR")

    lons, lats, bmp = analysis.get_basemap_from_hdf(file_path=base_config.data_path)

    basin_name_to_out_indices_map, basin_name_to_basin_mask = get_basin_to_outlet_indices_map(lons=lons, lats=lats,
                                                                                              accumulation_areas=facc,
                                                                                              directions=fldr)
    # Calculate the daily mean fields
    dates, stf_base = analysis.get_daily_climatology_for_rconf(base_config, var_name=varname, level=0)
    _, stf_modif = analysis.get_daily_climatology_for_rconf(modif_config, var_name=varname, level=0)

    for bname, (i_out, j_out) in basin_name_to_out_indices_map.items():
        print(bname, i_out, j_out)
        fig = plt.figure()

        gs = GridSpec(2, 1, height_ratios=[1, 0.5], hspace=0.1)

        ax = fig.add_subplot(gs[0, 0])
        ax.plot(dates, stf_base[:, i_out, j_out], "b", lw=2, label=base_config.label)
        ax.plot(dates, stf_modif[:, i_out, j_out], "r", lw=2, label=modif_config.label)
        ax.set_title(bname)
        format_axis(ax)

        # Hide the tick labels from the x-axis of the upper plot
        for tl in ax.xaxis.get_ticklabels():
            tl.set_visible(False)


        ax = fig.add_subplot(gs[1, 0])
        ax.plot(dates, stf_modif[:, i_out, j_out] - stf_base[:, i_out, j_out], "k", lw=2,
                label="{}-{}".format(modif_config.label, base_config.label))
        format_axis(ax)


        fig.savefig(str(IMG_FOLDER.joinpath("{}_{}-{}.png".format(bname, start_year, end_year))))
        plt.close(fig)


def format_axis(ax):
    ax.legend()
    ax.grid()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda tickval, pos: num2date(tickval).strftime("%b")[0]))
    ax.xaxis.set_major_locator(MonthLocator())



def main():
    plot_hydrographs()


if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()

    # Create the image folder if necessary
    if not IMG_FOLDER.is_dir():
        IMG_FOLDER.mkdir(parents=True)


    main()
