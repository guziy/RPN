from netCDF4 import Dataset
from pathlib import Path
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib import cm
from rpn.rpn import RPN
from data.cell_manager import CellManager
from domains.rotated_lat_lon import RotatedLatLon
from datetime import datetime, timedelta
from util import plot_utils
from matplotlib.dates import num2date, MonthLocator
import netCDF4 as nc

__author__ = 'huziy'

path_to_basin_shape = "/skynet3_exec2/aganji/glacier_katja/fraizer/fraizer.shp"

from osgeo import ogr

import numpy as np
import pandas as pd
img_folder = Path("offline_route")



def get_basemap_glaciers_nw_america():
    r = RPN("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/Depth_to_bedrock_WestNA_0.25")
    r.get_first_record_for_name("8L")
    proj_params = r.get_proj_parameters_for_the_last_read_rec()
    lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()
    bsmp = RotatedLatLon(**proj_params).get_basemap_object_for_lons_lats(lons2d=lons, lats2d=lats)
    return bsmp


def get_outlet_indices(basin_polygon, acc_areas, lon2d, lat2d):
    assert isinstance(basin_polygon, ogr.Geometry)
    acc = acc_areas.copy()

    lon2d[lon2d > 180] -= 360

    max_area = acc.max()
    while max_area > 0:
        i, j = np.where(acc == max_area)
        i, j = i[0], j[0]
        point = ogr.CreateGeometryFromWkt("POINT({} {})".format(lon2d[i, j], lat2d[i, j]))

        if basin_polygon.Contains(point):
            return i, j
        else:
            acc[i, j] = -1

        max_area = acc.max()

    print("Could not find the basin outlet...")


def get_basin_mask(basin_polygon, lon2d, lat2d):
    assert isinstance(basin_polygon, ogr.Geometry)
    lon2d[lon2d > 180] -= 360
    mask = np.zeros(np.prod(lon2d.shape))

    for i, (lon, lat) in enumerate(zip(lon2d.flatten(), lat2d.flatten())):
        point = ogr.CreateGeometryFromWkt("POINT({} {})".format(lon, lat))
        mask[i] = int(basin_polygon.Contains(point))


    return np.asarray(mask).reshape(lon2d.shape)


import matplotlib.pyplot as plt



def get_timeseries_from_netcdf(mask, cell_area, folder_path="", prefix="",
                               start_year=-np.inf, end_year=np.inf):

    dates = []
    data = []

    tot_runoff_vname = "TROF"

    folder = Path(folder_path)

    for p in folder.iterdir():
        if not p.name.startswith(prefix):
            continue

        # print("reading {}".format(p))
        with Dataset(str(p)) as ds:

            trof = ds.variables[tot_runoff_vname][:]
            time_var = ds.variables["time"]
            current_dates = nc.num2date(time_var[:], time_var.units)


            if current_dates[0].year > end_year or current_dates[-1].year < start_year:
                continue

            dates.extend(current_dates)

            # convert runoffs to m/s and areas to m**2
            data.extend(np.sum(trof * 1e-3 * cell_area[np.newaxis, :, :] * 1e6 * mask[np.newaxis, :, :],
                               axis=(1, 2)))



    df = pd.DataFrame(index=dates, data=data)


    dt_sec = (max(dates) - min(dates)) / float(len(dates) - 1)
    print("Time step: {}".format(dt_sec))
    assert isinstance(dt_sec, timedelta)
    dt_sec = dt_sec.total_seconds()

    df = df.groupby(lambda d: d.year).mean()
    df.sort(inplace=True)



    return df.index, df.values




def plot_runoff():
    plot_utils.apply_plot_params(width_pt=None, width_cm=24, height_cm=10, font_size=12)
    labels = ["Glacier-only", "All"]
    colors = ["r", "b"]
    infocell_path = "/skynet3_exec2/aganji/glacier_katja/watroute_gemera/infocell.nc"
    start_year = 2000
    end_year = 2099




    data_paths = [
        "/skynet3_exec2/aganji/glacier_katja/dyn/glacier_scaled/daily",
        "/skynet3_exec2/aganji/glacier_katja/dyn/glacier_AND_land/daily/",
    ]


    with Dataset(infocell_path) as ds:
        lons = ds.variables["lon"][:]
        lats = ds.variables["lat"][:]
        cell_area = ds.variables["cell_area"][:]

        acc_area = ds.variables["accumulation_area"][:]

        fldr = ds.variables["flow_direction_value"][:]




    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = driver.Open(path_to_basin_shape, 0)

    assert isinstance(data_source, ogr.DataSource)

    geom = None


    layer = data_source.GetLayer()
    assert isinstance(layer, ogr.Layer)

    for feature in layer:
        assert isinstance(feature, ogr.Feature)
        geom = feature.geometry()

        assert isinstance(geom, ogr.Geometry)

    i, j = get_outlet_indices(geom, acc_area, lons, lats)
    print("Accumulation area at the outlet (according to flow directions): {}".format(acc_area[i, j]))


    # cell_manager = CellManager(flow_dirs=fldr, lons2d=lons, lats2d=lats, accumulation_area_km2=acc_area)
    # model_mask = cell_manager.get_mask_of_cells_connected_with_by_indices(i, j)

    # comment model_mask when the correct shape file is in place
    mask = get_basin_mask(geom, lons, lats)
    # mask = model_mask
    print("Created basin mask with {} grid points inside.".format(int(np.sum(mask) + 0.5)))

    print("Modelled basin area: {} km**2".format(np.sum(mask * cell_area)))


    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)

    # Do the plotting
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2)

    # Plot the hydrograph
    ax = fig.add_subplot(gs[0, 0])

    for p, c, label in zip(data_paths, colors, labels):
        years, data = get_timeseries_from_netcdf(mask, cell_area, folder_path=p,
                                                 start_year=start_year, end_year=end_year)
        ax.plot(years, data, c, label=label, lw=2)
        print("Processed {}".format(label))

    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0, ncol=2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid()
    ax.set_title("{}-{}".format(start_year, end_year))

    # Plot the point position
    ax = fig.add_subplot(gs[0, 1])
    bsm = get_basemap_glaciers_nw_america()
    bsm.drawcoastlines(ax=ax)
    bsm.drawrivers(ax=ax, zorder=9, color="b")
    bsm.readshapefile(path_to_basin_shape.replace(".shp", ""), "basin", color="m", linewidth=2, zorder=0)

    # xx, yy = bsm(lons, lats)
    # cmap = cm.get_cmap("gray_r", 10)
    # bsm.pcolormesh(xx, yy, model_mask * 0.5, cmap=cmap, vmin=0, vmax=1)



    plt.savefig(str(img_folder.joinpath("fraser_runoff_ts.pdf")), bbox_inches="tight")
    plt.close(fig)



def plot_streamflow():
    plot_utils.apply_plot_params(width_pt=None, width_cm=19, height_cm=10, font_size=12)
    labels = ["Glacier-only", "All"]
    colors = ["r", "b"]
    paths = [
        "/skynet3_exec2/aganji/glacier_katja/watroute_gemera/discharge_stat_glac_00_99_2000_01_01_00_00.nc",
        "/skynet3_exec2/aganji/glacier_katja/watroute_gemera/discharge_stat_both_00_992000_01_01_00_00.nc"]

    infocell_path = "/skynet3_exec2/aganji/glacier_katja/watroute_gemera/infocell.nc"

    start_year = 2000
    end_year = 2099


    with Dataset(paths[0]) as ds:
        acc_area = ds.variables["accumulation_area"][:]
        lons = ds.variables["longitude"][:]
        lats = ds.variables["latitude"][:]
        x_index = ds.variables["x_index"][:]
        y_index = ds.variables["y_index"][:]

    with Dataset(infocell_path) as ds:
        fldr = ds.variables["flow_direction_value"][:]

    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = driver.Open(path_to_basin_shape, 0)

    assert isinstance(data_source, ogr.DataSource)

    geom = None

    print(data_source.GetLayerCount())

    layer = data_source.GetLayer()
    assert isinstance(layer, ogr.Layer)

    print(layer.GetFeatureCount())
    for feature in layer:
        assert isinstance(feature, ogr.Feature)
        geom = feature.geometry()

        assert isinstance(geom, ogr.Geometry)
        # print(str(geom))

        # geom = ogr.CreateGeometryFromWkt(geom.ExportToWkt())

    i, j = get_outlet_indices(geom, acc_area, lons, lats)
    print("Accumulation area at the outlet (according to flow directions): {}".format(acc_area[i, j]))


    cell_manager = CellManager(flow_dirs=fldr, lons2d=lons, lats2d=lats, accumulation_area_km2=acc_area)

    model_mask = cell_manager.get_mask_of_upstream_cells_connected_with_by_indices(i, j)


    cell_index = np.where((x_index == i) & (y_index == j))[0][0]

    print(cell_index)

    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)

    # Do the plotting
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, wspace=0.0)

    # Plot the hydrograph
    ax = fig.add_subplot(gs[0, 0])

    for p, c, label in zip(paths, colors, labels):
        with Dataset(p) as ds:
            stfl = ds.variables["water_discharge_accumulated"][:, cell_index]

            time = ds.variables["time"][:].astype(str)
            time = [datetime.strptime("".join(ts), "%Y_%m_%d_%H_%M") for ts in time]
            df = pd.DataFrame(index=time, data=stfl)

            # remove 29th of February
            df = df.select(lambda d: not (d.month == 2 and d.day == 29) and (start_year <= d.year <= end_year))

            df = df.groupby(lambda d: datetime(2001, d.month, d.day)).mean()

            ax.plot(df.index, df.values, c, lw=2, label=label)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda tickval, pos: num2date(tickval).strftime("%b")[0]))
    ax.xaxis.set_major_locator(MonthLocator())
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0)
    ax.set_title("{}-{}".format(start_year, end_year))

    # Plot the point position
    ax = fig.add_subplot(gs[0, 1])
    bsm = get_basemap_glaciers_nw_america()
    x, y = bsm(lons[i, j], lats[i, j])
    bsm.scatter(x, y, c="b", ax=ax, zorder=10)
    bsm.drawcoastlines()
    bsm.readshapefile(path_to_basin_shape.replace(".shp", ""), "basin", color="m", linewidth=2, zorder=5)

    # xx, yy = bsm(lons, lats)
    # cmap = cm.get_cmap("gray_r", 10)
    # bsm.pcolormesh(xx, yy, model_mask * 0.5, cmap=cmap, vmin=0, vmax=1)

    bsm.drawrivers(ax=ax, zorder=9, color="b")


    plt.savefig(str(img_folder.joinpath("stfl_at_outlets.pdf")), bbox_inches="tight")
    plt.close(fig)


def main():
    plot_runoff()
    # plot_streamflow()


if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()
    main()
