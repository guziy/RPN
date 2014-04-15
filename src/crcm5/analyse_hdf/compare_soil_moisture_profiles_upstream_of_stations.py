from datetime import datetime, timedelta
import time
import os
import brewer2mpl
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm
from matplotlib.dates import date2num, DateFormatter, MonthLocator
from crcm5 import infovar
from crcm5.model_point import ModelPoint
from data import cehq_station
from data.cehq_station import Station
from data.cell_manager import CellManager
import do_analysis_using_pytables as analysis
import numpy as np
import matplotlib.pyplot as plt
import common_plot_params as cpp


__author__ = 'huziy'


#Used for comparing soil moisture profiles
# averaged over upstream areas for a given station



#noinspection PyNoneFunctionAssignment
def main():
    start_year = 1980
    end_year = 1988

    soil_layer_widths = infovar.soil_layer_widths_26_to_60
    soil_tops = np.cumsum(soil_layer_widths).tolist()[:-1]
    soil_tops = [0, ] + soil_tops



    selected_station_ids = [
        "061905", "074903", "090613", "092715", "093801", "093806"
    ]

    path1 = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl_spinup.hdf"
    label1 = "CRCM5-HCD-RL"

    path2 = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup_ITFS.hdf5"
    label2 = "CRCM5-HCD-RL-INTFL"

    ############
    images_folder = "images_for_lake-river_paper/comp_soil_profiles"
    if not os.path.isdir(images_folder):
        os.mkdir(images_folder)

    fldirs = analysis.get_array_from_file(path=path1, var_name=infovar.HDF_FLOW_DIRECTIONS_NAME)
    lons2d, lats2d, basemap = analysis.get_basemap_from_hdf(path1)

    lake_fractions = analysis.get_array_from_file(path=path1, var_name=infovar.HDF_LAKE_FRACTION_NAME)
    cell_areas = analysis.get_array_from_file(path=path1, var_name=infovar.HDF_CELL_AREA_NAME)
    acc_areakm2 = analysis.get_array_from_file(path=path1, var_name=infovar.HDF_ACCUMULATION_AREA_NAME)
    depth_to_bedrock = analysis.get_array_from_file(path=path1, var_name=infovar.HDF_DEPTH_TO_BEDROCK_NAME)


    cell_manager = CellManager(fldirs, lons2d=lons2d, lats2d=lats2d, accumulation_area_km2=acc_areakm2)

    #get climatologic liquid soil moisture and convert fractions to mm
    t0 = time.clock()
    daily_dates, levels, i1_nointfl = analysis.get_daily_climatology_of_3d_field(
        path_to_hdf_file=path1,
        var_name="I1",
        start_year=start_year,
        end_year=end_year
    )
    print "read I1 - 1"
    print "Spent {0} seconds ".format(time.clock() - t0)

    _, _, i1_intfl = analysis.get_daily_climatology_of_3d_field(
        path_to_hdf_file=path2,
        var_name="I1",
        start_year=start_year,
        end_year=end_year
    )
    print "read I1 - 2"

    #get climatologic frozen soil moisture and convert fractions to mm
    _, _, i2_nointfl = analysis.get_daily_climatology_of_3d_field(
        path_to_hdf_file=path1,
        var_name="I2",
        start_year=start_year,
        end_year=end_year
    )
    print "read I2 - 1"

    _, _, i2_intfl = analysis.get_daily_climatology_of_3d_field(
        path_to_hdf_file=path2,
        var_name="I2",
        start_year=start_year,
        end_year=end_year
    )
    print "read I2 - 2"
    #
    sm_intfl = i1_intfl + i2_intfl
    sm_nointfl = i1_nointfl + i2_nointfl


    #Get the list of stations to do the comparison with
    stations = cehq_station.read_station_data(
        start_date=datetime(start_year, 1, 1),
        end_date=datetime(end_year, 12, 31),
        selected_ids=selected_station_ids
    )


    print "sm_noinfl, min, max = {0}, {1}".format(sm_nointfl.min(), sm_nointfl.max())
    print "sm_infl, min, max = {0}, {1}".format(sm_intfl.min(), sm_intfl.max())
    diff = (sm_intfl - sm_nointfl)
    #diff *= soil_layer_widths[np.newaxis, :, np.newaxis, np.newaxis] * 1000  # to convert in mm

    #print "number of nans", np.isnan(diff).astype(int).sum()

    print "cell area min,max = {0}, {1}".format(cell_areas.min(), cell_areas.max())
    print "acc area min,max = {0}, {1}".format(acc_areakm2.min(), acc_areakm2.max())

    assert np.all(lake_fractions >= 0)
    print "lake fractions (min, max): ", lake_fractions.min(), lake_fractions.max()

    #Non need to go very deep
    nlayers = 3
    z, t = np.meshgrid(soil_tops[:nlayers], date2num(daily_dates))
    station_to_mp = cell_manager.get_model_points_for_stations(stations)


    plotted_global = False

    for the_station, mp in station_to_mp.iteritems():
        assert isinstance(mp, ModelPoint)
        assert isinstance(the_station, Station)
        fig = plt.figure()
        umask = cell_manager.get_mask_of_cells_connected_with_by_indices(mp.ix, mp.jy)

        #exclude lake cells from the profiles
        sel = (umask == 1) & (depth_to_bedrock > 3) & (acc_areakm2 >= 0)

        umaskf = umask.astype(float)
        umaskf *= (1.0 - lake_fractions) * cell_areas
        umaskf[~sel] = 0.0


        profiles = np.tensordot(diff, umaskf) / umaskf.sum()
        print profiles.shape, profiles.min(), profiles.max(), umaskf.sum(), umaskf.min(), umaskf.max()

        d = np.abs(profiles).max()
        print "d = {0}".format(d)
        clevs = np.round(np.linspace(-d, d, 12), decimals=5)

        diff_cmap = brewer2mpl.get_map("RdBu", "diverging", 11, reverse=True).get_mpl_colormap(N=len(clevs) - 1)
        bn = BoundaryNorm(clevs, len(clevs) - 1)

        img = plt.contourf(t, z, profiles[:, :nlayers], cmap = diff_cmap, levels = clevs, norm = bn)
        plt.colorbar(img, ticks = clevs)
        ax = plt.gca()
        assert isinstance(ax, Axes)

        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(DateFormatter("%b"))
        ax.xaxis.set_major_locator(MonthLocator())


        fig.savefig(os.path.join(images_folder, "{0}.jpeg".format(the_station.id)),
                    dpi = cpp.FIG_SAVE_DPI, bbox_inches = "tight")



        print u"processed: {0}".format(the_station)
        if not plotted_global:
            plotted_global = True
            fig = plt.figure()
            sel = (depth_to_bedrock >= 0.1) & (acc_areakm2 >= 0)

            umaskf = (1.0 - lake_fractions) * cell_areas
            umaskf[~sel] = 0.0


            profiles = np.tensordot(diff, umaskf) / umaskf.sum()
            print profiles.shape, profiles.min(), profiles.max(), umaskf.sum(), umaskf.min(), umaskf.max()

            d = np.abs(profiles).max()
            print "d = {0}".format(d)
            clevs = np.round(np.linspace(-d, d, 12), decimals=5)

            diff_cmap = brewer2mpl.get_map("RdBu", "diverging", 11, reverse=True).get_mpl_colormap(N=len(clevs) - 1)
            bn = BoundaryNorm(clevs, len(clevs) - 1)

            img = plt.contourf(t, z, profiles[:, :nlayers], cmap = diff_cmap, levels = clevs, norm = bn)
            plt.colorbar(img, ticks = clevs)
            ax = plt.gca()
            assert isinstance(ax, Axes)

            ax.invert_yaxis()
            ax.xaxis.set_major_formatter(DateFormatter("%b"))
            ax.xaxis.set_major_locator(MonthLocator())


            fig.savefig(os.path.join(images_folder, "global_mean.jpeg"),
                        dpi = cpp.FIG_SAVE_DPI, bbox_inches = "tight")


    pass



if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()

    pass