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
from util import units
from util import plot_utils
__author__ = 'huziy'


def _avg_along(data, axis="lon", layer_depths=None, land_fraction=None):
    # Average along "lon" (i.e x-axis of the grid)
    if axis == "lon":
        dim = 1
    elif axis == "lat":
        dim = 2
    else:
        raise Exception("Unknown axis={}".format(axis))

    layer_depths = layer_depths if layer_depths is not None else np.ones_like(land_fraction)


    if hasattr(layer_depths, "mask"):
        layer_depths_rep = np.array([layer_depths for _ in range(data.shape[0])])
        data = np.ma.masked_where(layer_depths_rep <= 0, data)


    res = (data * land_fraction[np.newaxis, :, :] * layer_depths[np.newaxis, :, :]).sum(axis=dim) / land_fraction.sum(axis=dim - 1)[np.newaxis, :]

    return res


def main():
    # import seaborn as sns
    # sns.set_context("paper", font_scale=2)
    # sns.set_style("whitegrid")

    level_widths_mm = MM_PER_METER * infovar.soil_layer_widths_26_to_60

    avg_axis = "lon"
    start_year_c = 1980
    end_year_c = 2010

    img_folder = Path("impact_of_interflow")
    if not img_folder.exists():
        img_folder.mkdir(parents=True)


    # Configuration without interflow, to be compared with the one with intf.
    base_config = RunConfig(start_year=start_year_c, end_year=end_year_c,
                            data_path="/home/huziy/skynet3_rech1/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5",
                            label="NI")

    current_path = "/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5"

    # Need to read land fraction
    geo_file_path = "/RESCUE/skynet3_rech1/huziy/hdf_store/pm1979010100_00000000p"

    r_obj = RPN(geo_file_path)
    mg_field = r_obj.get_first_record_for_name("MG")
    depth_to_bedrock_mm = r_obj.get_first_record_for_name("8L") * MM_PER_METER


    # recompute layer widths to account for the depth to bedrock
    layer_widths_3d = np.ones(depth_to_bedrock_mm.shape + level_widths_mm.shape)
    layer_widths_3d *= level_widths_mm[np.newaxis, np.newaxis, :]

    layer_bottoms_3d = layer_widths_3d.cumsum(axis=2)

    corrections = depth_to_bedrock_mm[:, :, np.newaxis] - layer_bottoms_3d
    layer_widths_3d[corrections < 0] += corrections[corrections < 0]
    layer_widths_3d[layer_widths_3d < 0] = 0

    lons, lats = r_obj.get_longitudes_and_latitudes_for_the_last_read_rec()
    r_obj.close()

    modif_config = RunConfig(start_year=start_year_c, end_year=end_year_c,
                             data_path=current_path,
                             label="WI")

    varname = "INTF"
    level = 0
    daily_dates, intf_c = analysis.get_daily_climatology_for_rconf(modif_config,
                                                                   var_name=varname, level=level)

    # Convert to mm/day as well
    intf_c = _avg_along(intf_c, axis=avg_axis, land_fraction=mg_field) * 24 * 3600

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

    gs = GridSpec(8 - 3, 2, width_ratios=[1, 0.05])

    all_axes = []
    # ----------------------------------Interflow----------------------------------
    row = 0
    ax = fig.add_subplot(gs[row, 0])
    cs = ax.contourf(num_dates_2d, z_agg_2d, intf_c[:], cmap="jet", norm=LogNorm())
    ax.set_title("Interflow rate")
    all_axes.append(ax)

    # Colorbar for value plots
    cax = fig.add_subplot(gs[row, -1])

    sfmt = ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((-2, 2))

    plt.colorbar(cs, cax=cax, format=sfmt)
    cax.set_xlabel("mm/day")
    cax.yaxis.get_offset_text().set_position((-2, 10))


    # ----------------------------------Soil moisture----------------------------------
    layer_index = 0
    _, sm_mod = analysis.get_daily_climatology_for_rconf(modif_config, var_name="I1", level=layer_index)
    _, sm_base = analysis.get_daily_climatology_for_rconf(base_config, var_name="I1", level=layer_index)

    sm_mod = _avg_along(sm_mod, axis=avg_axis, land_fraction=mg_field, layer_depths=layer_widths_3d[:, :, layer_index])
    sm_base = _avg_along(sm_base, axis=avg_axis, land_fraction=mg_field, layer_depths=layer_widths_3d[:, :, layer_index])

    row += 1
    ax = fig.add_subplot(gs[row, 0])
    cs = ax.contourf(num_dates_2d, z_agg_2d, sm_mod - sm_base, cmap="jet")
    ax.set_title("Soil moisture (liq., level={})".format(layer_index + 1))
    all_axes.append(ax)

    # Colorbar for value plots
    cax = fig.add_subplot(gs[row, -1])

    sfmt = ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((-2, 2))

    plt.colorbar(cs, cax=cax, format=sfmt)
    cax.set_xlabel("mm")
    cax.yaxis.get_offset_text().set_position((-2, 10))

    # ----------------------------------Soil moisture (level=2)----------------------------------
    layer_index = 1
    _, sm_mod = analysis.get_daily_climatology_for_rconf(modif_config, var_name="I1", level=layer_index)
    _, sm_base = analysis.get_daily_climatology_for_rconf(base_config, var_name="I1", level=layer_index)

    sm_mod = _avg_along(sm_mod, axis=avg_axis, land_fraction=mg_field, layer_depths=layer_widths_3d[:, :, layer_index])
    sm_base = _avg_along(sm_base, axis=avg_axis, land_fraction=mg_field, layer_depths=layer_widths_3d[:, :, layer_index])

    row += 1
    ax = fig.add_subplot(gs[row, 0])
    cs = ax.contourf(num_dates_2d, z_agg_2d, sm_mod - sm_base, cmap="jet")
    ax.set_title("Soil moisture (liq., level={})".format(layer_index + 1))
    all_axes.append(ax)

    # Colorbar for value plots
    cax = fig.add_subplot(gs[row, -1])

    sfmt = ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((-2, 2))

    plt.colorbar(cs, cax=cax, format=sfmt)
    cax.set_xlabel("mm")
    cax.yaxis.get_offset_text().set_position((-2, 10))

    # ----------------------------------Soil moisture (level=3)----------------------------------
    layer_index = 2
    _, sm_mod = analysis.get_daily_climatology_for_rconf(modif_config, var_name="I1", level=layer_index)
    _, sm_base = analysis.get_daily_climatology_for_rconf(base_config, var_name="I1", level=layer_index)

    sm_mod = _avg_along(sm_mod, axis=avg_axis, land_fraction=mg_field, layer_depths=layer_widths_3d[:, :, layer_index])
    sm_base = _avg_along(sm_base, axis=avg_axis, land_fraction=mg_field, layer_depths=layer_widths_3d[:, :, layer_index])

    row += 1
    ax = fig.add_subplot(gs[row, 0])
    cs = ax.contourf(num_dates_2d, z_agg_2d, sm_mod - sm_base, cmap="jet")
    ax.set_title("Soil moisture (liq., level={})".format(layer_index + 1))
    all_axes.append(ax)

    # Colorbar for value plots
    cax = fig.add_subplot(gs[row, -1])

    sfmt = ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((-2, 2))

    plt.colorbar(cs, cax=cax, format=sfmt)
    cax.set_xlabel("mm")
    cax.yaxis.get_offset_text().set_position((-2, 10))

    # ----------------------------------Evaporation----------------------------------
    # _, av_mod = analysis.get_daily_climatology_for_rconf(modif_config, var_name="AV", level=0)
    # _, av_base = analysis.get_daily_climatology_for_rconf(base_config, var_name="AV", level=0)
    #
    # av_mod = _avg_along(av_mod, the_mask, axis=avg_axis)
    # av_base = _avg_along(av_base, the_mask, axis=avg_axis)
    #
    # row += 1
    # ax = fig.add_subplot(gs[row, 0])
    # cs = ax.contourf(num_dates_2d, z_agg_2d, av_mod - av_base, cmap="jet")
    # ax.set_title("Evaporation (level=1)")
    # all_axes.append(ax)
    #
    # # Colorbar for value plots
    # cax = fig.add_subplot(gs[row, -1])
    #
    # sfmt = ScalarFormatter(useMathText=True)
    # sfmt.set_powerlimits((-2, 2))
    #
    # plt.colorbar(cs, cax=cax, format=sfmt)
    # cax.set_xlabel("W/m**2")
    # cax.yaxis.get_offset_text().set_position((-2, 10))


    # ----------------------------------Drainage----------------------------------
    # _, dr_mod = analysis.get_daily_climatology_for_rconf(modif_config, var_name="TDRA", level=0)
    # _, dr_base = analysis.get_daily_climatology_for_rconf(base_config, var_name="TDRA", level=0)
    #
    # dr_mod = _avg_along(dr_mod, the_mask, axis=avg_axis)
    # dr_base = _avg_along(dr_base, the_mask, axis=avg_axis)
    #
    # row += 1
    # ax = fig.add_subplot(gs[row, 0])
    # cs = ax.contourf(num_dates_2d, z_agg_2d, (dr_mod - dr_base) * units.SECONDS_PER_DAY, cmap="jet")
    # ax.set_title("Drainage (level=1)")
    # all_axes.append(ax)
    #
    # # Colorbar for value plots
    # cax = fig.add_subplot(gs[row, -1])
    #
    # sfmt = ScalarFormatter(useMathText=True)
    # sfmt.set_powerlimits((-2, 2))
    #
    # plt.colorbar(cs, cax=cax, format=sfmt)
    # cax.set_xlabel("mm/day")
    # cax.yaxis.get_offset_text().set_position((-2, 10))

    # ----------------------------------Soil temperature (level_index=0)----------------------------------
    layer_index = 0
    _, t_mod = analysis.get_daily_climatology_for_rconf(modif_config, var_name="I0", level=layer_index)
    _, t_base = analysis.get_daily_climatology_for_rconf(base_config, var_name="I0", level=layer_index)

    t_mod = _avg_along(t_mod, land_fraction=mg_field, axis=avg_axis)
    t_base = _avg_along(t_base, land_fraction=mg_field, axis=avg_axis)

    row += 1
    ax = fig.add_subplot(gs[row, 0])
    cs = ax.contourf(num_dates_2d, z_agg_2d, t_mod - t_base, cmap="jet")
    ax.set_title("Soil temperature (level={})".format(layer_index + 1))
    all_axes.append(ax)

    # Colorbar for value plots
    cax = fig.add_subplot(gs[row, -1])

    sfmt = ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((-2, 2))

    plt.colorbar(cs, cax=cax, format=sfmt)
    cax.set_xlabel("K")
    cax.yaxis.get_offset_text().set_position((-2, 10))


    # ----------------------------------Soil moisture total (I1 + I2, level_index=0)----------------------------------
    # _, sm_mod = analysis.get_daily_climatology_for_rconf(modif_config, var_name="I1+I2", level=0)
    # _, sm_base = analysis.get_daily_climatology_for_rconf(base_config, var_name="I1+I2", level=0)
    #
    # sm_mod = _avg_along(sm_mod, the_mask, axis=avg_axis)
    # sm_base = _avg_along(sm_base, the_mask, axis=avg_axis)
    #
    # row += 1
    # ax = fig.add_subplot(gs[row, 0])
    # cs = ax.contourf(num_dates_2d, z_agg_2d, (sm_mod - sm_base) * level_widths_mm[0], cmap="jet")
    # ax.set_title("Soil moisture (liq.+ice, level=1)")
    # all_axes.append(ax)
    #
    # # Colorbar for value plots
    # cax = fig.add_subplot(gs[row, -1])
    #
    # sfmt = ScalarFormatter(useMathText=True)
    # sfmt.set_powerlimits((-2, 2))
    #
    # plt.colorbar(cs, cax=cax, format=sfmt)
    # cax.set_xlabel("mm")
    # cax.yaxis.get_offset_text().set_position((-2, 10))




    delta = 200
    vmin = -delta
    vmax = delta

    for i, the_ax in enumerate(all_axes):
        the_ax.xaxis.set_major_formatter(FuncFormatter(lambda d, pos: num2date(d).strftime("%b")[0]))
        the_ax.xaxis.set_major_locator(MonthLocator(bymonthday=15))
        the_ax.xaxis.set_minor_locator(MonthLocator(bymonthday=1))
        the_ax.grid(which="minor")
        the_ax.set_ylabel(ztitle)

    img_file = Path(img_folder).joinpath("INTF_rate_{}_avg_{}-{}.png".format(avg_axis, start_year_c, end_year_c))
    fig.tight_layout()
    fig.savefig(str(img_file), bbox_inches="tight")

    plt.show()


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()

    main()

