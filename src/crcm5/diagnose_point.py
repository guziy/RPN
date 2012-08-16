from datetime import datetime
import os
from matplotlib.axes import Axes
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.basemap import Basemap
from crcm5.model_data import Crcm5ModelDataManager
from cru.temperature import CRUDataManager
from data import cehq_station
from data.cehq_station import Station
from data.timeseries import TimeSeries
import matplotlib.pyplot as plt
from domains.rotated_lat_lon import RotatedLatLon
from gldas.gldas_manager import GldasManager
from rpn import level_kinds
from swe import SweDataManager
from util import plot_utils

__author__ = 'huziy'

import numpy as np

def plot_lake_fractions(ax, crcm5_manager, model_data, rot_latlon_projection,
                        margin = 5,
                        mask = None):
    """
    plot lake fractions 2d
    """
    assert isinstance(crcm5_manager, Crcm5ModelDataManager)
    assert isinstance(rot_latlon_projection, RotatedLatLon)
    lons2d, lats2d = crcm5_manager.lons2D, crcm5_manager.lats2D
    nx, ny = lons2d.shape

    i_model0, j_model0 = model_data.metadata["ix"], model_data.metadata["jy"]
    if mask is None:
        mask = crcm5_manager.get_mask_for_cells_upstream(i_model0, j_model0)

    i_interest, j_interest = np.where(mask == 1)

    i_min = max(0, min(i_interest) - margin)
    i_max = min(nx - 1, max(i_interest) + margin)

    j_min = max(0, min(j_interest) - margin )
    j_max = min(ny - 1, max(j_interest) + margin)

    #create basemap for the region
    basemap = Basemap(projection="omerc", llcrnrlat=lats2d[i_min, j_min], llcrnrlon=lons2d[i_min, j_min],
            urcrnrlon=lons2d[i_max, j_max], urcrnrlat=lats2d[i_max, j_max], no_rot=True,
            lon_1=rot_latlon_projection.lon1, lat_1=rot_latlon_projection.lat1,
            lon_2=rot_latlon_projection.lon2, lat_2=rot_latlon_projection.lat2,
        resolution="i"
    )


    basemap.drawcoastlines()
    basemap.drawrivers()

    lons_rot = np.zeros((i_max - i_min + 1, j_max - j_min + 1))
    lats_rot = np.zeros((i_max - i_min + 1, j_max - j_min + 1))

    for ii, i in enumerate( range(i_min, i_max + 1) ):
        for jj, j in enumerate( range(j_min, j_max + 1)):
            lons_rot[ii,jj], lats_rot[ii,jj] = rot_latlon_projection.toProjectionXY(lons2d[i,j], lats2d[i,j])


    dx = lons_rot[1,0] - lons_rot[0,0]
    dy = lats_rot[0,1] - lats_rot[0,0]

    lons_rot -= dx/2.0
    lats_rot -= dy/2.0


    lons_win = np.zeros((i_max - i_min + 1, j_max - j_min + 1))
    lats_win = np.zeros((i_max - i_min + 1, j_max - j_min + 1))

    lf_field = crcm5_manager.lake_fraction

    to_plot = np.ma.masked_all_like(lons_win)


    for ii, i in enumerate( range(i_min, i_max + 1) ):
        for jj, j in enumerate( range(j_min, j_max + 1)):
            lons_win[ii,jj], lats_win[ii,jj] = rot_latlon_projection.toGeographicLonLat(lons_rot[ii,jj], lats_rot[ii,jj])
            to_plot[ii, jj] = lf_field[i,j]




    x, y = basemap(lons2d, lats2d)
    basemap.pcolormesh(x, y, to_plot)
    plt.colorbar()
    print "lf: ", np.ma.min(to_plot), np.ma.max(to_plot)


    pass


def plot_gldas_runoff(ax, crcm5_manager, areas2d, model_data, mask = None):
    """
    model_data - is a Timeseries object of data at i0,j0 grid point
    """
    assert isinstance(ax, Axes)
    assert isinstance(crcm5_manager, Crcm5ModelDataManager)

    #model_ts = model_data.get_ts_of_daily_means()
    if mask is None:
        i_model0, j_model0 = model_data.metadata["ix"], model_data.metadata["jy"]
        mask = crcm5_manager.get_mask_for_cells_upstream(i_model0, j_model0)

    #time window, inclusive
    start_date = model_data.time[0]
    end_date = model_data.time[-1]

    gldas = GldasManager()
    assert isinstance(gldas, GldasManager)
    ts_surf = gldas.get_srof_spat_integrals_over_points_in_time(crcm5_manager.lons2D, crcm5_manager.lats2D, mask, areas2d=areas2d, start_date=start_date, end_date=end_date)



    ts_dr = gldas.get_subsrof_spat_integrals_over_points_in_time(crcm5_manager.lons2D, crcm5_manager.lats2D, mask, areas2d=areas2d, start_date=start_date, end_date=end_date)



    # /= 1000.0 -> transfom units mm/s -> m/s
    ax.plot(ts_surf.time, np.array( ts_surf.data ) / 1000.0, color = "k", label = "Surface")
    ax.plot(ts_dr.time, np.array( ts_dr.data ) / 1000.0, "r", lw = 1, label = "Subsurface")


    ax.annotate( "r = {0:.2f}".format( float( np.corrcoef([ts_surf.data, ts_dr.data])[0,1] )),
                xy = (0.1,0.8), xycoords= "axes fraction", zorder = 5)

    ax.set_ylabel("Integrated runoff (m^3/s)")
    ax.set_title("GLDAS (VIC)")
    ax.xaxis.set_major_formatter(DateFormatter("%y/%m"))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=range(1,13,2)))

    ax.legend()


def plot_swe_timeseries(ax, crcm5_manager, areas2d, model_data, mask = None):
    swe_obs_manager = SweDataManager(var_name="SWE")

    assert isinstance(crcm5_manager, Crcm5ModelDataManager)
    assert isinstance(ax, Axes)

    if mask is None:
        i_model0, j_model0 = model_data.metadata["ix"], model_data.metadata["jy"]
        mask = crcm5_manager.get_mask_for_cells_upstream(i_model0, j_model0)

    #print model_data.time[0], model_data.time[-1]

    #time window, inclusive
    start_date = model_data.time[0]
    end_date = model_data.time[-1]


    ts_swe_mod = crcm5_manager.get_monthly_means_over_points(mask, "I5", areas2d = areas2d,
        start_date = start_date, end_date = end_date)

    #cruManager = CRUDataManager(path="data/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.pre.dat.nc", var_name="pre")
    ts_swe_obs = swe_obs_manager.get_monthly_timeseries_using_mask(mask, crcm5_manager.lons2D, crcm5_manager.lats2D, areas2d,
        start_date=model_data.time[0], end_date = model_data.time[-1])



    mod = np.array( ts_swe_mod.data ) / 1000.0 #convert to m * m^2
    sta = np.array( ts_swe_obs.data ) / 1000.0 #convert to m * m^2


    print min(mod), max(mod)
    print min(sta), max(sta)

    ax.annotate( "r = {0:.2f}".format( float( np.corrcoef([mod, sta])[0,1] )),
            xy = (0.1,0.8), xycoords= "axes fraction", zorder = 5)
    ax.plot(ts_swe_mod.time, (mod - sta) , color = "k", linewidth = 2)
    ax.plot(ts_swe_mod.time, mod, color = "b")
    ax.plot(ts_swe_mod.time, sta, color = "r")

    ax.xaxis.set_major_formatter(DateFormatter("%y/%m"))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=range(1,13,2)))

    ax.set_title("CRCM5 versus Analysis (Ross Brown)")
    ax.set_ylabel("SWE, m^3, SWEmod - SWEobs")






def plot_runoff(ax, crcm5_manager, areas2d, model_data, mask = None):
    """
    model_data - is a Timeseries object of data at i0,j0 grid point
    """
    assert isinstance(ax, Axes)
    assert isinstance(crcm5_manager, Crcm5ModelDataManager)

    model_ts = model_data.get_ts_of_daily_means()
    if mask is None:
        i_model0, j_model0 = model_data.metadata["ix"], model_data.metadata["jy"]
        mask = crcm5_manager.get_mask_for_cells_upstream(i_model0, j_model0)

    ts_surf = crcm5_manager.get_daily_means_over_points(mask, "TRUN", level=5, areas2d=areas2d,
        start_date=model_ts.time[0], end_date=model_ts.time[-1]
    )

    ts_dr = crcm5_manager.get_daily_means_over_points(mask, "TDR", level=5, areas2d=areas2d,
        start_date=model_ts.time[0], end_date=model_ts.time[-1]
    )

    # /= 1000.0 -> transfom units mm/s -> m/s
    ax.plot(ts_surf.time, np.array( ts_surf.data ) / 1000.0, color = "k", label = "Surface")
    ax.plot(ts_dr.time, np.array( ts_dr.data ) / 1000.0, "r", lw = 1, label = "Subsurface")


    ax.annotate( "r = {0:.2f}".format( float( np.corrcoef([ts_surf.data, ts_dr.data])[0,1] )),
                xy = (0.1,0.8), xycoords= "axes fraction", zorder = 5)

    ax.set_ylabel("Integrated runoff (m^3/s)")

    ax.xaxis.set_major_formatter(DateFormatter("%y/%m"))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=range(1,13,2)))

    ax.legend()
    pass

def plot_total_precip_and_temp_re_1d(ax_pr, ax_temp, crcm5_manager,
                                 rot_latlon_projection, areas2d, model_data, mask = None):
    """
    plot relative error of total precipitation in time
    """
    assert isinstance(crcm5_manager, Crcm5ModelDataManager)
    assert isinstance(ax_pr, Axes)

    if mask is None:
        i_model0, j_model0 = model_data.metadata["ix"], model_data.metadata["jy"]
        mask = crcm5_manager.get_mask_for_cells_upstream(i_model0, j_model0)

    #print model_data.time[0], model_data.time[-1]



    #####Precipitation
    ts_prec_mod, dt_mod = crcm5_manager.get_monthly_sums_over_points(mask, "PR", areas2d=areas2d,
        start_date=model_data.time[0], end_date=model_data.time[-1]
    )

    cruManager = CRUDataManager(path="data/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.pre.dat.nc", var_name="pre")
    ts_prec_obs = cruManager.get_monthly_timeseries_using_mask(mask, crcm5_manager.lons2D, crcm5_manager.lats2D, areas2d,
        start_date=model_data.time[0], end_date = model_data.time[-1])

    
    
    mod = np.array( ts_prec_mod.data ) * dt_mod.seconds * 1000.0 #converting to mm/month
    sta = np.array( ts_prec_obs.data )                           #CRU data used was initially in mm/month

    #print sta
    #print mod
    #print mod.shape
    #print sta.shape


    assert len(sta) == len(mod)


    ax_pr.annotate( "r = {0:.2f}".format( float( np.corrcoef([mod, sta])[0,1] )),
            xy = (0.7,0.8), xycoords= "axes fraction")
    ax_pr.plot(ts_prec_mod.time, (mod - sta) / sta, color = "k", linewidth = 2)

    ax_pr.xaxis.set_major_formatter(DateFormatter("%y/%m"))
    ax_pr.xaxis.set_major_locator(MonthLocator(bymonth=range(1,13,2)))

    ax_pr.set_title("CRCM5 versus CRU")
    ax_pr.set_ylabel("Total Precip (Pmod - Pobs)/Pobs")



    #Temperature
    ts_temp_mod = crcm5_manager.get_spatial_integral_over_mask_of_dyn_field(mask, areas2d,
        var_name="TT", level=1000, level_kind=level_kinds.PRESSURE, path_to_folder=crcm5_manager.samples_folder,
        file_prefix="dp"
    )

    print "ts_temp_mod time interval: ", ts_temp_mod.time[0], ts_temp_mod.time[-1]
    ts_temp_mod = ts_temp_mod.time_slice(model_data.time[0], model_data.time[-1])

    cruManager = CRUDataManager(path="data/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.tmp.dat.nc", var_name="tmp")
    ts_temp_obs = cruManager.get_monthly_timeseries_using_mask(mask, crcm5_manager.lons2D, crcm5_manager.lats2D, areas2d,
        start_date=model_data.time[0], end_date = model_data.time[-1])



    mod = np.array( ts_temp_mod.get_ts_of_monthly_means().data ) / sum(areas2d[mask == 1]) #converting to mm/month
    sta = np.array( ts_temp_obs.data ) / sum(areas2d[mask == 1])                           #CRU data used was initially in mm/month


    assert len(sta) == len(mod)


    ax_temp.annotate( "r = {0:.2f}".format( float( np.corrcoef([mod, sta])[0,1] )),
            xy = (0.7,0.8), xycoords= "axes fraction")
    ax_temp.plot(ts_prec_mod.time, (mod - sta), color = "k", linewidth = 2)

    ax_temp.xaxis.set_major_formatter(DateFormatter("%y/%m"))
    ax_temp.xaxis.set_major_locator(MonthLocator(bymonth=range(1,13,2)))

    ax_temp.set_title("CRCM5 versus CRU")
    ax_temp.set_ylabel("Temperature, deg., Tmod - Tobs")






    pass

def plot_directions_and_positions(ax, station, model_data,
                                  crcm5_manager, rot_latlon_projection,
                                  margin = 5, mask = None):
    """
    margin - number of gridpoints to margin in order to focus on the gridpoint of interest
    """
    assert isinstance(ax, Axes)
    assert isinstance(station, Station)
    assert isinstance(model_data, TimeSeries)
    assert isinstance(crcm5_manager, Crcm5ModelDataManager)
    assert isinstance(rot_latlon_projection, RotatedLatLon) #grid projection


    lons2d, lats2d = crcm5_manager.lons2D, crcm5_manager.lats2D
    nx, ny = lons2d.shape

    i_model0, j_model0 = model_data.metadata["ix"], model_data.metadata["jy"]
    if mask is None:
        mask = crcm5_manager.get_mask_for_cells_upstream(i_model0, j_model0)

    i_interest, j_interest = np.where(mask == 1)

    i_min = max(0, min(i_interest) - margin)
    i_max = min(nx - 1, max(i_interest) + margin)

    j_min = max(0, min(j_interest) - margin )
    j_max = min(ny - 1, max(j_interest) + margin)




    #create basemap for the region
    basemap = Basemap(projection="omerc", llcrnrlat=lats2d[i_min, j_min], llcrnrlon=lons2d[i_min, j_min],
            urcrnrlon=lons2d[i_max, j_max], urcrnrlat=lats2d[i_max, j_max], no_rot=True,
            lon_1=rot_latlon_projection.lon1, lat_1=rot_latlon_projection.lat1,
            lon_2=rot_latlon_projection.lon2, lat_2=rot_latlon_projection.lat2,
        resolution="i"
    )

    sx, sy = basemap(station.longitude, station.latitude)
    basemap.scatter(sx, sy, c = "r", linewidths = 0, s = 80)
    #ax.annotate("s", xy = (sx, sy), va = "top", ha = "right", color= "r")

    mx, my = basemap(lons2d[i_model0, j_model0], lats2d[i_model0, j_model0])
    basemap.scatter(mx, my, c = "g", linewidths = 0, s = 20, marker = "s")
    #ax.annotate("m", xy = (mx, my), va = "bottom", ha = "left", color = "g")

    basemap.drawcoastlines(linewidth=0.2)
    basemap.drawrivers()


    for i,j in zip(i_interest, j_interest):
        next_cell = crcm5_manager.cell_manager.cells[i][j].next
        start = basemap(lons2d[i,j], lats2d[i,j])
        #ax.annotate("({0}, {1})".format(i,j), xy = start,font_properties = FontProperties(size = 2))
        if next_cell is None: continue
        next_i, next_j = next_cell.get_ij()

        end = basemap(lons2d[next_i, next_j], lats2d[next_i, next_j])


        print start, end
        ax.add_line(Line2D([start[0], end[0]], [start[1], end[1]], linewidth=0.5))


    pass




def plot_streamflows(ax, station, model_data):
    """
    :type ax: Axes
    :type station: Station
    :type model_data: TimeSeries
    """
    assert isinstance(ax, Axes)
    assert isinstance(station, Station)
    assert isinstance(model_data, TimeSeries)

    model_ts = model_data.get_ts_of_daily_means()
    mod_vals = model_ts.get_data_for_dates(station.dates)



    dt = station.dates[1] - station.dates[0]
    dt_sec = dt.days * 24 * 60 * 60 + dt.seconds


    mod_integral = sum(mod_vals) * dt_sec
    obs_integral = sum(station.values) * dt_sec


    print "+" * 20
    assert len(mod_vals) == len(station.dates)

    line_model = ax.plot(station.dates, mod_vals, label = "Model (CRCM5)", lw = 1, color = "b")
    line_obs = ax.plot(station.dates, station.values, label = "Observation", lw = 3, color = "r", alpha = 0.5)

    ax.annotate( "r = {0:.2f}".format( float( np.corrcoef([mod_vals, station.values])[0,1] )),
        xy = (0.1,0.8), xycoords= "axes fraction")

    ax.annotate("{0:3.2g} ".format(obs_integral) + "${\\rm m^3}$",
        xy = (0.1,0.7), xycoords= "axes fraction", color = "r"
    )
    ax.annotate("{0:3.2g} ".format(mod_integral) + "${\\rm m^3}$",
        xy = (0.1,0.6), xycoords= "axes fraction", color = "b"
    )


    ax.set_title("%s: da_diff=%.2f%%, d=%.1f km" % (station.id, (-station.drainage_km2+
                    model_ts.metadata["acc_area_km2"]) / station.drainage_km2 * 100.0,
                    model_ts.metadata["distance_to_obs_km"]))

    ax.xaxis.set_major_formatter(DateFormatter("%y/%m"))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=range(1,13,2)))
    ax.legend()


    pass


def plot_streamflow_re(ax, station, model_data):
    assert isinstance(ax, Axes)
    assert isinstance(station, Station)
    assert isinstance(model_data, TimeSeries)

    model_ts = model_data.get_ts_of_daily_means()
    mod_vals = model_ts.get_data_for_dates(station.dates)

    mod_vals = np.array(mod_vals)
    sta_vals = np.array(station.values)

    ax.plot(station.dates, (mod_vals - sta_vals) / sta_vals, color = "k", lw = 3)

    ax.set_ylabel("Streamflow, (Qmod - Qobs)/Qobs")

    ax.xaxis.set_major_formatter(DateFormatter("%y/%m"))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=range(1,13,2)))
    #ax.legend()



def diagnose(station_ids = None):
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_highres_spinup_12_month_without_lakes_v3"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_without_lakes"
    data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_without_lakes_v3"
    coord_file = os.path.join(data_path, "pm1985050100_00000000p")



    manager = Crcm5ModelDataManager(samples_folder_path=data_path,
            file_name_prefix="pm", all_files_in_samples_folder=True, need_cell_manager=True
    )

    nx, ny = manager.lons2D.shape

    rot_lat_lon = RotatedLatLon(lon1=-68, lat1=52, lon2=16.65, lat2=0.0)

    x00,y00 = rot_lat_lon.toProjectionXY(manager.lons2D[0,0], manager.lats2D[0,0])
    x10, y10 = rot_lat_lon.toProjectionXY(manager.lons2D[1,0], manager.lats2D[1,0])
    x01, y01 = rot_lat_lon.toProjectionXY(manager.lons2D[0,1], manager.lats2D[0,1])

    dx = x10 - x00
    dy = y01 - y00

    print "dx, dy = {0}, {1}".format( dx, dy )
    areas = rot_lat_lon.get_areas_of_gridcells(dx, dy, nx, ny, y00, 1)#1 -since the index is starting from 1
    print areas[0,0]

    start_date = datetime(1986, 1, 1)
    end_date = datetime(1986, 12, 31)

    stations = cehq_station.read_station_data(selected_ids = station_ids,
            start_date=start_date, end_date=end_date
    )

    stations.sort(key=lambda x: x.latitude, reverse=True)

    for i, s in enumerate(stations):

        fig = plt.figure()
        #3 columns
        gs = GridSpec( 5, 3, hspace=0.2, wspace=0.2, right=0.98, left=0.1, top=0.98 )

        model_ts = manager.get_streamflow_timeseries_for_station(s, start_date = start_date,
            end_date = end_date, nneighbours=10)

        print model_ts.time[0], model_ts.time[-1]

        i_model0, j_model0 = model_ts.metadata["ix"], model_ts.metadata["jy"]
        mask = manager.get_mask_for_cells_upstream(i_model0, j_model0)

        #hydrographs
        ax = fig.add_subplot( gs[0, 0] )
        plot_streamflows(ax, s, model_ts)

        #relative error
        ax = fig.add_subplot( gs[1, 0] )
        plot_streamflow_re(ax, s, model_ts)

        #directions
        plot_directions_and_positions(fig.add_subplot(gs[:2,1]), s, model_ts, manager, rot_lat_lon, mask = mask)

        #runoff
        ax = fig.add_subplot(gs[2,0])
        plot_runoff(ax, manager, areas, model_ts, mask= mask)

        #runoff from gldas
        ax = fig.add_subplot(gs[2,1])
        plot_gldas_runoff(ax, manager, areas, model_ts, mask = mask)


        #temperature
        ax_temp = fig.add_subplot(gs[3, 0])
        ax_prec = fig.add_subplot(gs[4, 0])

        plot_total_precip_and_temp_re_1d(ax_prec, ax_temp, manager, rot_lat_lon, areas, model_ts, mask=mask)


        #swe timeseries
        ax = fig.add_subplot(gs[3,1])
        plot_swe_timeseries(ax, manager, areas, model_ts, mask=mask)




        #print np.where(mask == 1)
        print "(i, j) = ({0}, {1})".format(model_ts.metadata["ix"], model_ts.metadata["jy"])

        fig.savefig("diagnose_{0}_{1:.2f}deg.pdf".format(s.id, dx))

def main():
    selected_ids = ["104001", "103715", "093806", "093801", "092715",
                        "081006", "061502", "040830", "080718"]

    #selected_ids = [selected_ids[0], ]
    diagnose(station_ids=selected_ids)

    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    plot_utils.apply_plot_params(width_pt=None, width_cm=40, height_cm=40, font_size=10)
    main()
    print "Hello world"
  
