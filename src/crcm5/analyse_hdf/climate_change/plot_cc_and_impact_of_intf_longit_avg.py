import collections
from matplotlib import cm
from matplotlib.dates import date2num, DateFormatter, MonthLocator
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from crcm5.analyse_hdf.run_config import RunConfig
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis
import numpy as np
from rpn.rpn import RPN
import matplotlib.pyplot as plt
from util import plot_utils

__author__ = 'huziy'


def _avg_along_lon(data, land_fraction, land_fraction_crit=1.0e-4):
    # Average along "lon" (i.e x-axis of the grid)
    data = np.ma.masked_where(land_fraction <= land_fraction_crit, data)

    return (data * land_fraction).sum(axis=1) / land_fraction.sum(axis=1)



def main():
    start_year_c = 1980
    end_year_c = 2010

    img_folder = "cc_paper"

    current_path = "/RESCUE/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"
    base_label = "CRCM5-L"

    # Need to read land fraction
    geo_file_path = "/RESCUE/skynet3_rech1/huziy/hdf_store/pm1979010100_00000000p"

    r_obj = RPN(geo_file_path)
    mg_field = r_obj.get_first_record_for_name("MG")

    lons, lats = r_obj.get_longitudes_and_latitudes_for_the_last_read_rec()
    r_obj.close()

    future_shift_years = 75

    params = dict(
        data_path=current_path,
        start_year=start_year_c, end_year=end_year_c,
        label=base_label
    )

    base_config_c = RunConfig(**params)
    base_config_f = base_config_c.get_shifted_config(future_shift_years)

    data_path = "/RESCUE/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-intfl-cc-canesm2-1980-2010.hdf5"
    params.update(
        dict(data_path=data_path,
             label="CRCM5-LI"))

    modif_config_c = RunConfig(**params)
    modif_config_f = modif_config_c.get_shifted_config(future_shift_years)

    varnames = ["I1+I2", "I0", "PR", "TRAF", "AV", "I0_max", "I0_min"]
    levels = [0, 0, 0, 0, 0, 0, 0]
    var_labels = ["SM", "ST", "PR", "TRAF", "LHF", "STmin", "STmax"]

    # width of the first soil layer in mm
    multipliers = [100, 1, 1000 * 24 * 3600, 24 * 3600, 1, 1, 1]
    offsets = [0, ] * len(varnames)
    units = ["mm", "K", "mm/day", "mm/day", r"${\rm W/m^2}$", "K", "K"]

    SimData = collections.namedtuple("SimData", "base_c base_f modif_c modif_f")

    mg_fields = None

    fig = plt.figure()
    gs = GridSpec(len(varnames), 5, width_ratios=2 * [1, ] + [0.05, 1, 0.05])

    lats_agg = lats.mean(axis=0)

    diff_cmap = cm.get_cmap("RdBu_r", 10)
    the_zip = zip(varnames, levels, var_labels, multipliers, offsets, units)
    row = 0
    for vname, level, var_label, multiplier, offset, unit_label in the_zip:

        daily_dates, base_data_c = analysis.get_daily_climatology(path_to_hdf_file=base_config_c.data_path,
                                                                  var_name=vname, level=level,
                                                                  start_year=base_config_c.start_year,
                                                                  end_year=base_config_c.end_year)

        _, base_data_f = analysis.get_daily_climatology(path_to_hdf_file=base_config_f.data_path,
                                                        var_name=vname, level=level,
                                                        start_year=base_config_f.start_year,
                                                        end_year=base_config_f.end_year)

        _, modif_data_c = analysis.get_daily_climatology(path_to_hdf_file=modif_config_c.data_path,
                                                         var_name=vname, level=level,
                                                         start_year=modif_config_c.start_year,
                                                         end_year=modif_config_c.end_year)

        _, modif_data_f = analysis.get_daily_climatology(path_to_hdf_file=modif_config_f.data_path,
                                                         var_name=vname, level=level,
                                                         start_year=modif_config_f.start_year,
                                                         end_year=modif_config_f.end_year)

        if mg_fields is None:
            mg_fields = np.asarray([mg_field for d in daily_dates])
            num_dates = date2num(daily_dates)

            # create 2d dates and latitudes for the contour plots
            lats_agg_2d, num_dates_2d = np.meshgrid(lats_agg, num_dates)



        sim_data = SimData(_avg_along_lon(base_data_c, mg_fields),
                           _avg_along_lon(base_data_f, mg_fields),
                           _avg_along_lon(modif_data_c, mg_fields),
                           _avg_along_lon(modif_data_f, mg_fields))

        # Unit conversion
        sim_data = SimData(*[multiplier * si + offset for si in sim_data])


        # Plot the row for the variable
        all_axes = []



        # Calculate nice color levels
        delta = np.percentile(np.abs([sim_data.modif_c - sim_data.base_c,
                                      sim_data.modif_f - sim_data.base_f]), 99)
        vmin = -delta
        vmax = delta
        locator = MaxNLocator(nbins=10, symmetric=True)
        clevs = locator.tick_values(vmin=vmin, vmax=vmax)


        # Current
        ax = fig.add_subplot(gs[row, 0])
        cs = ax.contourf(num_dates_2d, lats_agg_2d, sim_data.modif_c - sim_data.base_c,
                         extend="both", levels=clevs, cmap=diff_cmap)
        if row == 0:
            ax.set_title("Current ({}-{})".format(
                base_config_c.start_year, base_config_c.end_year))
        all_axes.append(ax)

        # Future
        ax = fig.add_subplot(gs[row, 1])
        if row == 0:
            ax.set_title("Future ({}-{})".format(
                base_config_f.start_year, base_config_f.end_year))

        cs = ax.contourf(num_dates_2d, lats_agg_2d, sim_data.modif_f - sim_data.base_f,
                         levels=cs.levels, extend="both", cmap=diff_cmap)
        all_axes.append(ax)

        # Colorbar for value plots
        cax = fig.add_subplot(gs[row, 2])
        plt.colorbar(cs, cax=cax)
        cax.set_title("{} ({})\n".format(var_label, unit_label))


        diff = (sim_data.modif_f - sim_data.base_f) - (sim_data.modif_c - sim_data.base_c)
        delta = np.percentile(np.abs(diff), 99)
        vmin = -delta
        vmax = delta
        locator = MaxNLocator(nbins=10, symmetric=True)
        clevs = locator.tick_values(vmin=vmin, vmax=vmax)

        ax = fig.add_subplot(gs[row, 3])
        cs = ax.contourf(num_dates_2d, lats_agg_2d, diff, cmap=diff_cmap,
                         levels=clevs, extend="both")
        all_axes.append(ax)
        cb = plt.colorbar(cs, cax=fig.add_subplot(gs[row, 4]))

        if row == 0:
            ax.set_title("Future - Current")

        cb.ax.set_title("{}\n".format(unit_label))

        for i, the_ax in enumerate(all_axes):
            the_ax.xaxis.set_major_formatter(DateFormatter("%b"))
            the_ax.xaxis.set_major_locator(MonthLocator(interval=2))

            the_ax.grid()
            if i == 0:
                the_ax.set_ylabel("Latitude")

        row += 1


    fig.tight_layout()
    img_path = Path(img_folder).joinpath("{}_long_avg_intf_impact_{}-{}_vs_{}-{}.png".format("_".join(varnames),
                                                                                             base_config_f.start_year,
                                                                                             base_config_f.end_year,
                                                                                             base_config_c.start_year,
                                                                                             base_config_c.end_year))
    fig.savefig(str(img_path), bbox_inches="tight")


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()

    plot_utils.apply_plot_params(width_cm=35, width_pt=None, font_size=12)
    main()
