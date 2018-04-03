

# at 10km resolution 100km distance is approximated as 10 * dx
import os
from collections import OrderedDict

from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from pathlib import Path
from rpn import level_kinds
from rpn.rpn import RPN

from application_properties import main_decorator
from crcm5.nemo_vs_hostetler import nemo_hl_util
from crcm5.nemo_vs_hostetler import commons

import numpy as np
import matplotlib.pyplot as plt

from crcm5.nemo_vs_hostetler.rpn_lakeice_manager import RPNLakeIceManager
from util import plot_utils
from netCDF4 import Dataset, num2date, MFDataset

img_folder = "nemo_vs_hostetler"


def get_mask_of_points_near_lakes(lake_mask, npoints_radius=10):
    """
    Get the mask of points near lakes where lake effect snow is probable
    :param lake_mask:
    :param npoints_radius:
    :return:
    """
    i_list, j_list = np.where(lake_mask)

    nx, ny = lake_mask.shape

    the_mask = np.zeros_like(lake_mask, dtype=np.bool)

    for i, j in zip(i_list, j_list):
        imin = max(0, i - npoints_radius)
        imax = min(nx - 1, i + npoints_radius)

        jmin = max(0, j - npoints_radius)
        jmax = min(ny - 1, j + npoints_radius)

        the_mask[imin:imax + 1, jmin:jmax + 1] = True

    the_mask[lake_mask] = False

    return the_mask


def get_map_ij_to_nonlocal_mask(region_of_lake_effect_snow_mask, lake_mask, npoints_radius=50):
    """
    Return the non-local vicinity of each point from the reion of the lake effect snow
    :param region_of_lake_effect_snow_mask:
    :param lake_mask:
    :param npoints_radius:
    :return:
    """
    i_arr, j_arr = np.where(region_of_lake_effect_snow_mask)

    nx, ny = region_of_lake_effect_snow_mask.shape

    result = {}
    for i, j in zip(i_arr, j_arr):
        the_mask = np.zeros_like(region_of_lake_effect_snow_mask)

        imin = max(0, i - npoints_radius)
        imax = min(nx - 1, i + npoints_radius)
        jmin = max(0, j - npoints_radius)
        jmax = min(ny - 1, j + npoints_radius)


        the_mask[imin:imax + 1, jmin:jmax + 1] = True
        the_mask[lake_mask | region_of_lake_effect_snow_mask] = False
        result[(i, j)] = the_mask

    return result


def get_wind_blows_from_lake_mask(lake_mask, lake_effect_region, u_field, v_field, dx=0.1, dy=0.1, lake_ice_frac=None, lats_rot=None):
    """

    """

    if lake_ice_frac is None:
        lake_ice_frac = np.zeros_like(lake_mask)


    dtx = np.asarray(dx / np.abs(u_field))

    if lats_rot is not None:
        dtx *= np.cos(np.radians(lats_rot))



    dty = np.asarray(dy / np.abs(v_field))

    wind_blows_from_lake = np.zeros_like(lake_mask, dtype=np.bool)

    nx, ny = lake_mask.shape


    for i, j in zip(*np.where(lake_mask)):

        i1 = i
        j1 = j

        nsteps = 0

        if lake_ice_frac[i, j] > 0.7:
            continue

        while True:


            if dtx[i1, j1] < dty[i1, j1] / 3.0:
                sgn = np.sign(u_field[i1, j1])
                i1 += int(sgn + 0.5 * sgn)

            elif dtx[i1, j1] > dty[i1, j1] / 3.0:
                sgn = np.sign(v_field[i1, j1])
                j1 += int(sgn + 0.5 * sgn)

            else:
                i1 += int(np.sign(u_field[i1, j1]) * 1.5)
                j1 += int(np.sign(v_field[i1, j1]) * 1.5)


            nsteps += 1

            if (i1 < 0) or (i1 >= nx) or (j1 < 0) or (j1 >= ny):
                break
            else:
                if not (lake_effect_region[i1, j1] or lake_mask[i1, j1]):
                    break
                else:
                    if wind_blows_from_lake[i1, j1]:
                        break
                    else:
                        wind_blows_from_lake[i1, j1] = True

    return wind_blows_from_lake & lake_effect_region




@main_decorator
def main():
    start_year = 1979
    end_year = 1981

    HL_LABEL = "CRCM5_HL"
    NEMO_LABEL = "CRCM5_NEMO"

    dx = 0.1
    dy = 0.1


    file_prefix = "pm"
    PR_level = -1
    PR_level_type = level_kinds.ARBITRARY

    tprecip_vname = "PR"
    sprecip_vname = "SN"

    TT_level = 1
    TT_level_type = level_kinds.HYBRID

    sim_label_to_path = OrderedDict(
        [(HL_LABEL, "/RESCUE/skynet3_rech1/huziy/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl_oneway/Samples"),
         (NEMO_LABEL, "/HOME/huziy/skynet3_rech1/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl/Samples")]
    )

    # get a coord file ... (use pm* files, since they contain NEM1 variable)
    # Should be NEMO_LABEL, since the hostetler case does not calculate NEM? vars
    coord_file = ""
    found_coord_file = False
    for mdir in os.listdir(sim_label_to_path[NEMO_LABEL]):

        mdir_path = os.path.join(sim_label_to_path[NEMO_LABEL], mdir)
        if not os.path.isdir(mdir_path):
            continue

        for fn in os.listdir(mdir_path):

            if fn[:2] not in ["pm", ]:
                continue


            if fn[-9:-1] == "0" * 8:
                continue

            coord_file = os.path.join(mdir_path, fn)
            found_coord_file = True

        if found_coord_file:
            break

    bmp, lons, lats = nemo_hl_util.get_basemap_obj_and_coords_from_rpn_file(path=coord_file)
    xx, yy = bmp(lons, lats)


    r = RPN(coord_file)
    lats_rot = r.get_first_record_for_name("^^")
    lons_rot = r.get_first_record_for_name(">>")



    lake_mask = np.greater(commons.get_nemo_lake_mask_from_rpn(coord_file, vname="NEM1"), 0)


    # Get the 100km region around the lakes
    lake_effect_regions = get_mask_of_points_near_lakes(lake_mask, npoints_radius=10)



    local_amplification_limit = 4 * 1e-2 / (24.0 * 3600.0)
    # the radius is 500 km, i.e. 50 gridpoints
    ij_to_non_local_mask = get_map_ij_to_nonlocal_mask(lake_effect_regions, lake_mask, npoints_radius=50)


    # Snowfall amount criteria (>= 10 cm)
    lower_snow_fall_limit = 10 * 1e-2 / (24.0 * 3600.0)  # convert to M/s

    # wind blows from lake: time limit
    wind_blows_from_lake_time_limit_hours = 6.0

    months_of_interest = [10, 11, 12, 1, 2, 3, 4, 5]

    sim_label_to_duration_mean = {}
    sim_label_to_lake_effect_sprecip_mean = {}

    sim_label_to_year_to_lake_effect_snow_fall_duration = OrderedDict([(sim_label, OrderedDict()) for sim_label in sim_label_to_path])

    for sim_label, samples_dir_path in sim_label_to_path.items():

        # calculate the composites for the (Oct - March) period

        lake_effect_snowfall_mean_duration = None  # the duration is in time steps
        lake_effect_mean_snowrate_m_per_s = None


        snowfall_current_event = None
        duration_current_event = None  # the duration is in time steps

        n_events = None
        sn_previous = None

        time_wind_blows_from_lake = None

        samples_dir = Path(samples_dir_path)

        snowfall_file = samples_dir.parent / "{}_snow_fall_{}-{}.nc".format(sim_label, start_year, end_year)
        wind_components_file = samples_dir.parent / "rotated_wind_{}.nc".format(sim_label)

        ds_wind = Dataset(str(wind_components_file))

        print("Working on {} ...".format(sim_label))

        lkice_manager = RPNLakeIceManager(samples_dir=samples_dir)

        with Dataset(str(snowfall_file)) as ds:

            time_var = ds.variables["time"]
            nt = time_var.shape[0]
            snowfall_var_m_per_s = ds.variables["SN"]
            u_var = ds_wind.variables["UU"]
            v_var = ds_wind.variables["VV"]

            time_var_wind = ds_wind.variables["time"]

            assert time_var_wind.shape == time_var.shape
            assert time_var_wind[0] == time_var[0]
            assert time_var_wind[-1] == time_var_wind[-1]

            assert (u_var.shape == snowfall_var_m_per_s.shape) and (v_var.shape == snowfall_var_m_per_s.shape)

            times = num2date(time_var[:], time_var.units)

            dt_seconds = (times[1] - times[0]).total_seconds()

            year_to_lake_effect_snow_fall_duration = sim_label_to_year_to_lake_effect_snow_fall_duration[sim_label]



            for ti, t in enumerate(times):
                if t.month not in months_of_interest:
                    continue

                if t.year > end_year or t.year < start_year:
                    continue


                sn_current = snowfall_var_m_per_s[ti, :, :]

                if t.year not in year_to_lake_effect_snow_fall_duration:
                    year_to_lake_effect_snow_fall_duration[t.year] = np.zeros_like(sn_current)

                # initialize aggragtion fields
                if lake_effect_snowfall_mean_duration is None:
                    lake_effect_snowfall_mean_duration = np.zeros_like(sn_current)
                    lake_effect_mean_snowrate_m_per_s = np.zeros_like(sn_current)
                    n_events = np.zeros_like(sn_current)
                    snowfall_current_event = np.zeros_like(sn_current)
                    duration_current_event = np.zeros_like(sn_current)
                    sn_previous = np.zeros_like(sn_current)
                    time_wind_blows_from_lake = np.zeros_like(sn_current)



                where_lake_effect_snow = (sn_current > lower_snow_fall_limit) & lake_effect_regions & (~lake_mask)


                # add a condition on the local amplification
                i_arr, j_arr = np.where(where_lake_effect_snow)
                for i, j in zip(i_arr, j_arr):
                    the_mask = ij_to_non_local_mask[(i, j)]
                    where_lake_effect_snow[i, j] = sn_current[the_mask].mean() < sn_current[i, j] - local_amplification_limit



                # add a condition on the wind fetch from lakes and ice fraction.
                wind_blows_from_lake = get_wind_blows_from_lake_mask(lake_mask, lake_effect_regions, u_var[ti, :, :], v_var[ti, :, :],
                                                                    dx=dx, dy=dy,
                                                                     lake_ice_frac=lkice_manager.get_lake_fraction_for_date(the_date=t),
                                                                     lats_rot=lats_rot)



                time_wind_blows_from_lake[wind_blows_from_lake] += dt_seconds / 3600.0
                where_lake_effect_snow = where_lake_effect_snow & (time_wind_blows_from_lake >= wind_blows_from_lake_time_limit_hours)
                time_wind_blows_from_lake[~wind_blows_from_lake] = 0



                # update accumulators for current lake effect snowfall events
                snowfall_current_event[where_lake_effect_snow] += sn_current[where_lake_effect_snow]
                duration_current_event[where_lake_effect_snow] += 1.0

                where_lake_effect_snow_finished = (~where_lake_effect_snow) & (sn_previous > lower_snow_fall_limit)



                # recalculate mean lake effect snowfall duration and rate
                lake_effect_snowfall_mean_duration[where_lake_effect_snow_finished] = (lake_effect_snowfall_mean_duration[where_lake_effect_snow_finished] * n_events[where_lake_effect_snow_finished] + duration_current_event[where_lake_effect_snow_finished]) / (n_events[where_lake_effect_snow_finished] + 1)
                lake_effect_mean_snowrate_m_per_s[where_lake_effect_snow_finished] = (lake_effect_mean_snowrate_m_per_s[where_lake_effect_snow_finished] * n_events[where_lake_effect_snow_finished] + snowfall_current_event[where_lake_effect_snow_finished]) / (n_events[where_lake_effect_snow_finished] + 1)


                year_to_lake_effect_snow_fall_duration[t.year][where_lake_effect_snow_finished] += duration_current_event[where_lake_effect_snow_finished] * dt_seconds

                # reset the current accumulators
                snowfall_current_event[where_lake_effect_snow_finished] = 0
                duration_current_event[where_lake_effect_snow_finished] = 0

                n_events[where_lake_effect_snow_finished] += 1

                sn_previous = sn_current

                if ti % 1000 == 0:
                    print("Done {} of {}".format(ti + 1, nt))




        # normalization

        lake_effect_snowfall_mean_duration *= dt_seconds / (24 * 60 * 60.0)  # convert to days


        lake_effect_mean_snowrate_m_per_s = np.ma.masked_where(~lake_effect_regions, lake_effect_mean_snowrate_m_per_s)
        lake_effect_snowfall_mean_duration = np.ma.masked_where(~lake_effect_regions, lake_effect_snowfall_mean_duration)

        for y, yearly_durations in sim_label_to_year_to_lake_effect_snow_fall_duration[sim_label].items():
            sim_label_to_year_to_lake_effect_snow_fall_duration[sim_label][y] = np.ma.masked_where(~lake_effect_regions, yearly_durations) / (24 * 3600.0)

        sim_label_to_duration_mean[sim_label] = lake_effect_snowfall_mean_duration
        sim_label_to_lake_effect_sprecip_mean[sim_label] = lake_effect_mean_snowrate_m_per_s * 100 * 24 * 3600.0

        # close the file with rotated wind components
        ds_wind.close()

    plot_utils.apply_plot_params(font_size=6, width_cm=18, height_cm=10)
    fig = plt.figure()
    gs = GridSpec(3, 3)

    duration_clevs = 20  # np.arange(0, 1.1, 0.1)
    snowrate_clevs = 20  # np.arange(0, 36, 4)
    duration_clevs_diff = 20  # np.arange(-1, 1.1, 0.1)
    snowrate_clevs_diff = 20  # np.arange(-10, 12, 2)




    vmax_duration = None
    vmax_snowrate = None
    vmax_days_per_year = None
    for row, sim_label in enumerate(sim_label_to_path):

        if vmax_duration is None:
            vmax_duration = sim_label_to_duration_mean[sim_label].max()
            vmax_snowrate = sim_label_to_lake_effect_sprecip_mean[sim_label].max()
            vmax_days_per_year = sim_label_to_year_to_lake_effect_snow_fall_duration[sim_label][1980].max()
        else:
            vmax_duration = max(vmax_duration, sim_label_to_duration_mean[sim_label].max())
            vmax_snowrate = max(vmax_snowrate, sim_label_to_lake_effect_sprecip_mean[sim_label].max())

            vmax_days_per_year = max(vmax_days_per_year, sim_label_to_year_to_lake_effect_snow_fall_duration[sim_label][1980].max())


    for col, sim_label in enumerate(sim_label_to_path):
        # plot the duration of lake-effect snow events
        ax = fig.add_subplot(gs[0, col])
        cs = bmp.pcolormesh(xx, yy, sim_label_to_duration_mean[sim_label], ax=ax, vmin=0, vmax=vmax_duration, cmap="rainbow_r")
        bmp.drawcoastlines(linewidth=0.3, ax=ax)
        plt.colorbar(cs, ax=ax)
        ax.set_title("Duration (days)")
        ax.set_xlabel("{}".format(sim_label))

        # plot the mean intensity of the lake-effect snow events
        ax = fig.add_subplot(gs[1, col])
        cs = bmp.pcolormesh(xx, yy, sim_label_to_lake_effect_sprecip_mean[sim_label],
                          ax=ax, vmax=vmax_snowrate, vmin=lower_snow_fall_limit, cmap="rainbow_r")
        bmp.drawcoastlines(linewidth=0.3, ax=ax)

        plt.colorbar(cs, ax=ax)
        ax.set_title("Snowfall rate, (cm/day)")
        ax.set_xlabel("{}".format(sim_label))


        # plot the mean duration of the lake effect snowfall events per year
        ax = fig.add_subplot(gs[2, col])
        to_plot = sim_label_to_year_to_lake_effect_snow_fall_duration[sim_label][1980]
        clevs = [0, 0.1, ] + list(np.arange(0.4, 3.2, 0.4))
        bn = BoundaryNorm(clevs, len(clevs))
        cmap = cm.get_cmap("spectral_r", len(clevs))

        cs = bmp.pcolormesh(xx, yy, to_plot, ax=ax, norm=bn, cmap=cmap)
        bmp.drawcoastlines(linewidth=0.3, ax=ax)
        plt.colorbar(cs, ax=ax, extend="max")
        ax.set_title("# Days per year")
        ax.set_xlabel("{}".format(sim_label))

    # plot the difference

    # plot the duration of lake-effect snow events
    col = 2
    cmap = cm.get_cmap("seismic", 40)
    vmin = -np.max(sim_label_to_duration_mean[NEMO_LABEL] - sim_label_to_duration_mean[HL_LABEL])
    ax = fig.add_subplot(gs[0, col])
    cs = bmp.pcolormesh(xx, yy, sim_label_to_duration_mean[NEMO_LABEL] - sim_label_to_duration_mean[HL_LABEL], vmin=vmin, ax=ax, cmap=cmap)
    plt.colorbar(cs, ax=ax)
    bmp.drawcoastlines(linewidth=0.3, ax=ax)
    ax.set_title("Duration (days)")
    ax.set_xlabel("{} - {}".format(NEMO_LABEL, HL_LABEL))

    # plot the mean intensity of the lake-effect snow events
    ax = fig.add_subplot(gs[1, col])
    vmin = -np.max(sim_label_to_lake_effect_sprecip_mean[NEMO_LABEL] - sim_label_to_lake_effect_sprecip_mean[HL_LABEL])
    cs = bmp.pcolormesh(xx, yy, sim_label_to_lake_effect_sprecip_mean[NEMO_LABEL] - sim_label_to_lake_effect_sprecip_mean[HL_LABEL], ax=ax, vmin=vmin, cmap=cmap)  # convert to cm/day
    bmp.drawcoastlines(linewidth=0.3, ax=ax)
    plt.colorbar(cs, ax=ax)
    ax.set_title("Snowfall rate, (cm/day)")
    ax.set_xlabel("{} - {}".format(NEMO_LABEL, HL_LABEL))

    # plot the mean duration of the lake effect snowfall events per year
    ax = fig.add_subplot(gs[2, col])
    to_plot = (sim_label_to_year_to_lake_effect_snow_fall_duration[NEMO_LABEL][1980] - sim_label_to_year_to_lake_effect_snow_fall_duration[HL_LABEL][1980])
    cs = bmp.pcolormesh(xx, yy, to_plot, ax=ax, vmin=-to_plot.max(), cmap="seismic")
    bmp.drawcoastlines(linewidth=0.3, ax=ax)
    plt.colorbar(cs, ax=ax)
    ax.set_title("# Days per year")
    ax.set_xlabel("{} - {}".format(NEMO_LABEL, HL_LABEL))

    fig.tight_layout()
    fig.savefig(os.path.join(img_folder, "lake_effect_snow_10cm_limit_and_loc_ampl_{}-{}.png".format(start_year, end_year)), dpi=commons.dpi, transparent=True, bbox_inches="tight")






if __name__ == '__main__':
    main()