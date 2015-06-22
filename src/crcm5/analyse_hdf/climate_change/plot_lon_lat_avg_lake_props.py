from pathlib import Path
from matplotlib.colors import LogNorm
from matplotlib.dates import date2num, MonthLocator, num2date, DayLocator
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter, MaxNLocator, FuncFormatter
from rpn.rpn import RPN
from crcm5 import infovar
from crcm5.analyse_hdf.run_config import RunConfig
import numpy as np
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis
import matplotlib.pyplot as plt
from util.units import MM_PER_METER
from util import plot_utils

__author__ = 'huziy'


def _avg_along(data, axis="lon", layer_depths=None, lake_fraction=None):
    # Average along "lon" (i.e x-axis of the grid)
    if axis == "lon":
        dim = 1
    elif axis == "lat":
        dim = 2
    else:
        raise Exception("Unknown axis={}".format(axis))

    layer_depths = layer_depths if layer_depths is not None else np.ones_like(lake_fraction)

    if hasattr(layer_depths, "mask"):
        layer_depths_rep = np.array([layer_depths for _ in range(data.shape[0])])
        data = np.ma.masked_where(layer_depths_rep <= 0, data)

    res = (data * lake_fraction[np.newaxis, :, :] * layer_depths[np.newaxis, :, :]).sum(axis=dim) / lake_fraction.sum(
        axis=dim - 1)[np.newaxis, :]

    return res


def main():
    # import seaborn as sns
    # sns.set_context("paper", font_scale=2)
    # sns.set_style("whitegrid")

    level_widths_mm = MM_PER_METER * infovar.soil_layer_widths_26_to_60

    avg_axis = "lon"
    start_year_c = 1980
    end_year_c = 2010

    img_folder = Path("cc_paper/lake_props")
    if not img_folder.exists():
        img_folder.mkdir(parents=True)


    # Configuration without interflow, to be compared with the one with intf.
    base_config = RunConfig(start_year=start_year_c, end_year=end_year_c,
                            data_path="/home/huziy/skynet3_rech1/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5",
                            label="CRCM5-L")

    current_path = "/RESCUE/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"

    # Need to read land fraction
    geo_file_path = "/RESCUE/skynet3_rech1/huziy/hdf_store/pm1979010100_00000000p"

    r_obj = RPN(geo_file_path)
    mg_field = r_obj.get_first_record_for_name("MG")
    depth_to_bedrock_mm = r_obj.get_first_record_for_name("8L") * MM_PER_METER
    lake_fraction = r_obj.get_first_record_for_name("ML")


    # recompute layer widths to account for the depth to bedrock
    layer_widths_3d = np.ones(depth_to_bedrock_mm.shape + level_widths_mm.shape)
    layer_widths_3d *= level_widths_mm[np.newaxis, np.newaxis, :]

    layer_bottoms_3d = layer_widths_3d.cumsum(axis=2)

    corrections = depth_to_bedrock_mm[:, :, np.newaxis] - layer_bottoms_3d
    layer_widths_3d[corrections < 0] += corrections[corrections < 0]
    layer_widths_3d[layer_widths_3d < 0] = 0

    lons, lats = r_obj.get_longitudes_and_latitudes_for_the_last_read_rec()
    r_obj.close()

    # Current and future configurations
    current_config = RunConfig(start_year=start_year_c, end_year=end_year_c,
                               data_path=current_path,
                               label="CRCM5-L")

    n_shift_years = 90
    future_config = current_config.get_shifted_config(n_shift_years)
    print(future_config)

    varname = "L1"
    level = 0
    daily_dates, lake_temp_c = analysis.get_daily_climatology_for_rconf(current_config,
                                                                        var_name=varname, level=level)

    _, lake_temp_f = analysis.get_daily_climatology_for_rconf(future_config,
                                                              var_name=varname, level=level)

    # average along a dim
    lake_temp_c = _avg_along(lake_temp_c, axis=avg_axis, lake_fraction=lake_fraction)
    lake_temp_f = _avg_along(lake_temp_f, axis=avg_axis, lake_fraction=lake_fraction)

    zagg = None
    ztitle = ""
    if avg_axis == "lon":
        zagg = lats.mean(axis=0)
        ztitle = "Latitude"
    elif avg_axis == "lat":
        zagg = lons.mean(axis=1)
        ztitle = "Longitude"

    num_dates = date2num(daily_dates)

    z_agg_2d, num_dates_2d = np.meshgrid(zagg, num_dates)


    # Do the plotting
    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=20, height_cm=30)
    fig = plt.figure()

    gs = GridSpec(4, 2, width_ratios=[1, 0.05])

    all_axes = []
    # ----------------------------------Lake temperature----------------------------------
    row = 0
    ax = fig.add_subplot(gs[row, 0])
    cs = ax.contourf(num_dates_2d, z_agg_2d, lake_temp_f - lake_temp_c[:, :], cmap="jet")
    ax.set_title("Lake temperature (liquid)")
    all_axes.append(ax)

    # Colorbar for value plots
    cax = fig.add_subplot(gs[row, -1])

    sfmt = ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((-2, 3))

    plt.colorbar(cs, cax=cax, format=sfmt)
    cax.set_xlabel(r"${\rm ^\circ C}$")
    cax.yaxis.get_offset_text().set_position((-2, 10))


    # ----------------------------------Lake ice thickness----------------------------------
    level = 0
    varname = "LD"
    _, lake_ice_th_c = analysis.get_daily_climatology_for_rconf(current_config,
                                                                var_name=varname, level=level)

    _, lake_ice_th_f = analysis.get_daily_climatology_for_rconf(future_config,
                                                                var_name=varname, level=level)

    lake_ice_th = _avg_along(lake_ice_th_f - lake_ice_th_c, axis=avg_axis, lake_fraction=lake_fraction)

    row += 1
    ax = fig.add_subplot(gs[row, 0])
    cs = ax.contourf(num_dates_2d, z_agg_2d, lake_ice_th, cmap="jet")
    ax.set_title("Lake ice thickness")
    all_axes.append(ax)

    # Colorbar for value plots
    cax = fig.add_subplot(gs[row, -1])

    sfmt = ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((-2, 2))

    plt.colorbar(cs, cax=cax, format=sfmt)
    cax.set_xlabel("m")
    cax.yaxis.get_offset_text().set_position((-2, 10))

    # ----------------------------------Lake ice fraction----------------------------------
    level = 0
    varname = "LC"
    _, lake_depth_c = analysis.get_daily_climatology_for_rconf(current_config,
                                                               var_name=varname, level=level)

    _, lake_depth_f = analysis.get_daily_climatology_for_rconf(future_config,
                                                               var_name=varname, level=level)

    lake_ice_fraction = _avg_along(lake_depth_f - lake_depth_c, axis=avg_axis,
                                   lake_fraction=lake_fraction)

    row += 1
    ax = fig.add_subplot(gs[row, 0])
    cs = ax.contourf(num_dates_2d, z_agg_2d, lake_ice_fraction, cmap="jet")
    ax.set_title("Lake ice fraction")
    all_axes.append(ax)

    # Colorbar for value plots
    cax = fig.add_subplot(gs[row, -1])

    sfmt = ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((-2, 2))

    plt.colorbar(cs, cax=cax, format=sfmt)
    cax.set_xlabel("")
    cax.yaxis.get_offset_text().set_position((-2, 10))
    # ----------------------------------Lake ice fraction----------------------------------
    level = 0
    varname = "CLDP"
    _, lake_depth_c = analysis.get_daily_climatology_for_rconf(current_config,
                                                               var_name=varname, level=level)

    _, lake_depth_f = analysis.get_daily_climatology_for_rconf(future_config,
                                                               var_name=varname, level=level)

    lake_ice_fraction = _avg_along(lake_depth_f - lake_depth_c, axis=avg_axis,
                                   lake_fraction=lake_fraction)

    row += 1
    ax = fig.add_subplot(gs[row, 0])
    cs = ax.contourf(num_dates_2d, z_agg_2d, lake_ice_fraction, cmap="jet")
    ax.set_title("Water level")
    all_axes.append(ax)

    # Colorbar for value plots
    cax = fig.add_subplot(gs[row, -1])

    sfmt = ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((-2, 2))

    plt.colorbar(cs, cax=cax, format=sfmt)
    cax.set_xlabel("m")
    cax.yaxis.get_offset_text().set_position((-2, 10))

    for i, the_ax in enumerate(all_axes):
        the_ax.xaxis.set_major_formatter(FuncFormatter(lambda d, pos: num2date(d).strftime("%b")[0]))
        the_ax.xaxis.set_major_locator(MonthLocator(bymonthday=15))
        the_ax.xaxis.set_minor_locator(MonthLocator(bymonthday=1))
        the_ax.grid(which="minor")
        the_ax.set_ylabel(ztitle)

    img_file = Path(img_folder).joinpath("cc_{}_Lake_props_current_{}_avg_{}-{}_vs_{fsy}-{fey}.png".format(
        base_config.label, avg_axis, start_year_c, end_year_c,
        fsy=future_config.start_year, fey=future_config.end_year))

    fig.tight_layout()
    fig.savefig(str(img_file), bbox_inches="tight")

    plt.show()


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()

    main()

