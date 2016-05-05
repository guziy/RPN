from datetime import datetime
from pathlib import Path
from matplotlib import cm
from matplotlib.colors import SymLogNorm, LogNorm, BoundaryNorm
from matplotlib.dates import date2num, DateFormatter, MonthLocator, num2date
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FuncFormatter
from crcm5.analyse_hdf.run_config import RunConfig

__author__ = 'huziy'

# Interflow rate in the CRCM5 outputs is in mm/s

from crcm5.analyse_hdf import do_analysis_using_pytables as analysis
import tables as tb
from rpn.rpn import RPN
import numpy as np
import matplotlib.pyplot as plt


def _avg_along_lon(data, mask):
    # Average along "lon" (i.e x-axis of the grid)
    return np.ma.masked_where(mask, data).mean(axis=1)


def main():
    start_year_c = 1980
    end_year_c = 2010

    img_folder = "cc_paper"

    current_path = "/RESCUE/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-intfl-cc-canesm2-1980-2010.hdf5"
    base_label = "CRCM5-LI"

    # Need to read land fraction
    geo_file_path = "/RESCUE/skynet3_rech1/huziy/hdf_store/pm1979010100_00000000p"

    r_obj = RPN(geo_file_path)
    mg_field = r_obj.get_first_record_for_name("MG")

    lons, lats = r_obj.get_longitudes_and_latitudes_for_the_last_read_rec()
    r_obj.close()

    future_shift_years = 90

    params = dict(
        data_path=current_path,
        start_year=start_year_c, end_year=end_year_c,
        label=base_label
    )

    base_config_c = RunConfig(**params)
    base_config_f = base_config_c.get_shifted_config(future_shift_years)

    varname = "INTF"
    level = 0
    daily_dates, intf_c = analysis.get_daily_climatology(path_to_hdf_file=base_config_c.data_path,
                                                         var_name=varname, level=level,
                                                         start_year=base_config_c.start_year,
                                                         end_year=base_config_c.end_year)

    _, intf_f = analysis.get_daily_climatology(path_to_hdf_file=base_config_f.data_path,
                                               var_name=varname, level=level,
                                               start_year=base_config_f.start_year,
                                               end_year=base_config_f.end_year)


    mg_fields = np.asarray([mg_field for d in daily_dates])

    mg_crit = 0.0001
    the_mask = mg_fields <= mg_crit
    # Convert to mm/day as well
    intf_c = _avg_along_lon(intf_c, the_mask) * 24 * 3600
    intf_f = _avg_along_lon(intf_f, the_mask) * 24 * 3600



    lats_agg = lats.mean(axis=0)
    num_dates = date2num(daily_dates)

    lats_agg_2d, num_dates_2d = np.meshgrid(lats_agg, num_dates)


    # Do the plotting
    fig = plt.figure()

    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.05])
    norm = SymLogNorm(5e-5)

    all_axes = []
    # Current
    ax = fig.add_subplot(gs[0, 0])
    cs = ax.contourf(num_dates_2d, lats_agg_2d, intf_c[:], 60, norm=norm)
    ax.set_title("Current ({}-{})".format(
        base_config_c.start_year, base_config_c.end_year))
    all_axes.append(ax)

    # Future
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Future ({}-{})".format(
        base_config_f.start_year, base_config_f.end_year))
    cs = ax.contourf(num_dates_2d, lats_agg_2d, intf_f[:], levels=cs.levels, norm=norm)
    all_axes.append(ax)





    # Colorbar for value plots
    cax = fig.add_subplot(gs[0, 2])

    sfmt = ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((-1, 2))

    plt.colorbar(cs, cax=cax, format=sfmt)
    cax.set_xlabel("mm/day")
    cax.yaxis.get_offset_text().set_position((-2, 10))



    # CC
    diff_cmap = cm.get_cmap("RdBu_r", 20)
    diff = (intf_f - intf_c) / (0.5 * (intf_c + intf_f)) * 100
    diff[(intf_f == 0) & (intf_c == 0)] = 0
    print(np.min(diff), np.max(diff))
    print(np.any(diff.mask))
    print(np.any(intf_c.mask))
    print(np.any(intf_f.mask))
    delta = 200
    vmin = -delta
    vmax = delta
    locator = MaxNLocator(nbins=20, symmetric=True)
    clevs = locator.tick_values(vmin=vmin, vmax=vmax)

    ax = fig.add_subplot(gs[1, 1])

    cs = ax.contourf(num_dates_2d, lats_agg_2d, diff, cmap=diff_cmap,
                     levels=clevs, extend="both")
    ax.set_title("Future - Current")
    # ax.set_aspect("auto")
    all_axes.append(ax)
    cb = plt.colorbar(cs, cax=fig.add_subplot(gs[1, -1]))
    cb.ax.set_xlabel(r"%")


    for i, the_ax in enumerate(all_axes):
        the_ax.xaxis.set_minor_formatter(FuncFormatter(lambda d, pos: num2date(d).strftime("%b")[0]))
        the_ax.xaxis.set_major_formatter(FuncFormatter(lambda d, pos: ""))
        the_ax.xaxis.set_minor_locator(MonthLocator(bymonthday=15))
        the_ax.xaxis.set_major_locator(MonthLocator())
        the_ax.grid()
        if i != 1:
            the_ax.set_ylabel(r"Latitude ${\rm \left(^\circ N \right)}$")



    # identify approximately the melting period and lower latitudes
    march1 = date2num(datetime(daily_dates[0].year, 3, 1))
    june1 = date2num(datetime(daily_dates[0].year, 6, 1))

    sel_mask = (num_dates_2d >= march1) & (num_dates_2d < june1) & (lats_agg_2d <= 50)
    print("Mean interflow decrease in the southern regions: {}%".format(diff[sel_mask].mean()))


    # identify the regions of max interflow rates in current and future climates
    lat_min = 55
    lat_max = 57.5

    may1 = date2num(datetime(daily_dates[0].year, 5, 1))
    july1 = date2num(datetime(daily_dates[0].year, 7, 1))

    mean_max_current = intf_c[(lats_agg_2d >= lat_min) & (lats_agg_2d <= lat_max) & (num_dates_2d <= july1) & (num_dates_2d >= june1)].mean()
    mean_max_future = intf_f[(lats_agg_2d >= lat_min) & (lats_agg_2d <= lat_max) & (num_dates_2d <= june1) & (num_dates_2d >= may1)].mean()

    print("Mean change in the maximum interflow rate: {} %".format((mean_max_future - mean_max_current) * 100 / mean_max_current))

    img_file = Path(img_folder).joinpath("INTF_rate_longit_avg.png")
    fig.tight_layout()
    from crcm5.analyse_hdf import common_plot_params
    fig.savefig(str(img_file), bbox_inches="tight", transparent=True, dpi=common_plot_params.FIG_SAVE_DPI)








if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()

    main()
