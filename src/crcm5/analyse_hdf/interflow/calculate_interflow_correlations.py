from collections import defaultdict
from datetime import datetime, timedelta
import os
import matplotlib
from matplotlib.dates import DateFormatter
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

matplotlib.use("Agg")

from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import maskoceans
from util import plot_utils

__author__ = 'huziy'

import crcm5.analyse_hdf.do_analysis_using_pytables as analysis
import numpy as np
import matplotlib.pyplot as plt
import crcm5.analyse_hdf.common_plot_params as cpp
from crcm5 import crcm_constants
from crcm5 import infovar


def calculate_correlation_nd(data1, data2, axis=0):
    nt = data1.shape[axis]
    assert data1.shape == data2.shape

    view1 = data1
    view2 = data2

    if axis:
        view1 = np.rollaxis(data1, axis)
        view2 = np.rollaxis(data2, axis)

    data1_norm = (view1 - data1.mean(axis=axis)) / data1.std(axis=axis)
    data2_norm = (view2 - data2.mean(axis=axis)) / data2.std(axis=axis)

    return np.sum(data1_norm * data2_norm / float(nt), axis=0)


def calculate_correlation(data1, data2):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    nt = data1.shape[0]

    eps1 = 1e-20 * data1.std(axis=0).max()
    eps2 = 1e-20 * data2.std(axis=0).max()

    data1_norm = (data1 - data1.mean(axis=0)) / (data1.std(axis=0) + eps1)
    data2_norm = (data2 - data2.mean(axis=0)) / (data2.std(axis=0) + eps2)

    return np.sum(data1_norm * data2_norm / float(nt), axis=0)


def calculate_correlation_field_for_climatology(start_year=None,
                                                end_year=None,
                                                path1="",
                                                varname1="",
                                                level1=None,
                                                path2="",
                                                varname2="",
                                                level2=None, months=None):
    dates, data1 = analysis.get_daily_climatology(path_to_hdf_file=path1, var_name=varname1, level=level1,
                                                  start_year=start_year, end_year=end_year)

    dates, data2 = analysis.get_daily_climatology(path_to_hdf_file=path2, var_name=varname2, level=level2,
                                                  start_year=start_year, end_year=end_year)

    if months is None:
        months = list(range(1, 13))

    selfields1 = [f for date, f in zip(dates, data1) if date.month in months]
    selfields2 = [f for date, f in zip(dates, data2) if date.month in months]

    return calculate_correlation(selfields1, selfields2), np.array(selfields1), np.array(selfields2)


def calculate_correlation_of_infiltration_rate_with(start_year=None,
                                                    end_year=None,
                                                    path_for_infiltration_data="",
                                                    path2="",
                                                    varname2="",
                                                    level2=None, months=None):
    dates, pr_data = analysis.get_daily_climatology(path_to_hdf_file=path_for_infiltration_data, var_name="PR",
                                                    level=0,
                                                    start_year=start_year, end_year=end_year)

    # Take interflow calculated for soil subareas
    dates, srunoff_data = analysis.get_daily_climatology(path_to_hdf_file=path_for_infiltration_data, var_name="TRAF",
                                                         level=0,
                                                         start_year=start_year, end_year=end_year)

    dates, evap_data = analysis.get_daily_climatology(path_to_hdf_file=path_for_infiltration_data, var_name="AV",
                                                      level=0,
                                                      start_year=start_year, end_year=end_year)
    # Convert lists to numpy arrays
    pr_data = np.array(pr_data)
    srunoff_data = np.array(srunoff_data)
    evap_data = np.array(evap_data)

    # calculate infiltration from runoff precip and evap
    infiltration = pr_data - srunoff_data / crcm_constants.rho_water - \
                   evap_data / (crcm_constants.Lv_J_per_kg * crcm_constants.rho_water)

    dates, data2 = analysis.get_daily_climatology(path_to_hdf_file=path2, var_name=varname2, level=level2,
                                                  start_year=start_year, end_year=end_year)

    if months is None:
        months = list(range(1, 13))

    selfields1 = [f for date, f in zip(dates, infiltration) if date.month in months]
    selfields2 = [f for date, f in zip(dates, data2) if date.month in months]

    return calculate_correlation(selfields1, selfields2)


def plot_tmin_tmax_correlations(start_year=1980, end_year=2010, months=None):
    default_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5"
    # default_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS_avoid_truncation1979-1989.hdf5"

    if months is None:
        months = list(range(1, 13))

    img_folder = os.path.join("interflow_corr_images", os.path.basename(default_path))
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)

    img_filename = "Tmin_Tmax_interflow_correlations_months={}_{}-{}.jpg".format("-".join(str(m) for m in months),
                                                                                 start_year, end_year)

    lons, lats, basemap = analysis.get_basemap_from_hdf(file_path=default_path)
    lons[lons > 180] -= 360
    x, y = basemap(lons, lats)

    # Correlate interflow rate and soil moisture
    params = dict(
        path1=default_path,
        varname1="INTF",
        level1=0,

        path2=default_path,
        level2=0,
        varname2="TT_max",
        months=months
    )

    params.update(dict(
        start_year=start_year, end_year=end_year,
    ))

    title_list = []
    data_list = []

    corr1, intf_clim, i1_clim = calculate_correlation_field_for_climatology(**params)
    to_plot1 = maskoceans(lons, lats, corr1)
    title_list.append("Corr({}, {})".format(params["varname1"], params["varname2"]))
    data_list.append(to_plot1)

    # correlate interflow and precip
    # params.update(dict(varname2="PR", level2=0))
    # corr2 = calculate_correlation_field_for_climatology(**params)
    # to_plot2 = np.ma.masked_where(to_plot1.mask, corr2)
    # title_list.append("Corr({}, {})".format(params["varname1"], params["varname2"]))
    # data_list.append(to_plot2)

    # correlate precip and soil moisture
    # params.update(dict(varname1="I1", level1=0))
    # corr3 = calculate_correlation_field_for_climatology(**params)
    # to_plot3 = np.ma.masked_where(to_plot2.mask, corr3)
    # title_list.append("Corr({}, {})".format(params["varname1"], params["varname2"]))
    # data_list.append(to_plot3)


    # correlate evaporation and soil moisture
    params.update(dict(varname2="TT_min", level2=0, varname1="INTF", level1=0))
    corr4, i1_clim, av_clim = calculate_correlation_field_for_climatology(**params)
    to_plot3 = np.ma.masked_where(to_plot1.mask, corr4)
    title_list.append("Corr({}, {})".format(params["varname1"], params["varname2"]))
    data_list.append(to_plot3)




    # Do plotting
    clevels = np.arange(-1, 1.2, 0.2)

    npanels = len(data_list)
    gs = GridSpec(1, npanels + 1, width_ratios=[1.0, ] * npanels + [0.05, ])

    fig = plt.figure()
    assert isinstance(fig, Figure)
    fig.set_figheight(1.5 * fig.get_figheight())

    img = None
    for col in range(npanels):
        ax = fig.add_subplot(gs[0, col])
        basemap.drawmapboundary(fill_color="0.75", ax=ax)

        img = basemap.contourf(x, y, data_list[col], levels=clevels, cmap=cm.get_cmap("RdBu_r", len(clevels) - 1))
        plt.title(title_list[col])
        basemap.drawcoastlines(linewidth=cpp.COASTLINE_WIDTH, ax=ax)

    plt.colorbar(img, cax=fig.add_subplot(gs[0, npanels]))
    fig.savefig(os.path.join(img_folder, img_filename), dpi=cpp.FIG_SAVE_DPI)


def main(start_year=1980, end_year=2010, months=None):
    default_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5"
    # default_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS_avoid_truncation1979-1989.hdf5"

    if months is None:
        months = list(range(1, 13))

    img_folder = os.path.join("interflow_corr_images", os.path.basename(default_path))
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)

    img_filename = "interflow_correlations_months={}_{}-{}.pdf".format("-".join(str(m) for m in months),
                                                                       start_year, end_year)

    lons, lats, basemap = analysis.get_basemap_from_hdf(file_path=default_path)
    lons[lons > 180] -= 360
    x, y = basemap(lons, lats)

    # Correlate interflow rate and soil moisture
    params = dict(
        path1=default_path,
        varname1="INTF",
        level1=0,

        path2=default_path,
        level2=0,
        varname2="I1",
        months=months
    )

    params.update(dict(
        start_year=start_year, end_year=end_year,
    ))

    title_list = []
    data_list = []

    corr1, intf_clim, i1_clim = calculate_correlation_field_for_climatology(**params)
    to_plot1 = maskoceans(lons, lats, corr1)
    title_list.append("Corr({}, {})".format(
        infovar.get_display_label_for_var(params["varname1"]),
        infovar.get_display_label_for_var(params["varname2"])))
    data_list.append(to_plot1)

    # correlate interflow and precip
    params.update(dict(varname2="PR", level2=0))
    corr2, i1_clim, pr_clim = calculate_correlation_field_for_climatology(**params)
    to_plot2 = np.ma.masked_where(to_plot1.mask, corr2)
    title_list.append("Corr({}, {})".format(
        infovar.get_display_label_for_var(params["varname1"]),
        infovar.get_display_label_for_var(params["varname2"])))
    data_list.append(to_plot2)

    # correlate precip and soil moisture
    # params.update(dict(varname1="I1", level1=0))
    # corr3 = calculate_correlation_field_for_climatology(**params)
    # to_plot3 = np.ma.masked_where(to_plot2.mask, corr3)
    # title_list.append("Corr({}, {})".format(params["varname1"], params["varname2"]))
    # data_list.append(to_plot3)


    # correlate evaporation and soil moisture
    params.update(dict(varname2="AV", level2=0, varname1="I1", level1=0))
    corr4, i1_clim, av_clim = calculate_correlation_field_for_climatology(**params)
    to_plot3 = np.ma.masked_where(to_plot1.mask, corr4)
    title_list.append("Corr({}, {})".format(
        infovar.get_display_label_for_var(params["varname1"]),
        infovar.get_display_label_for_var(params["varname2"])))
    data_list.append(to_plot3)



    # correlate interflow and evaporation
    params.update(dict(varname2="AV", level2=0, varname1="INTF", level1=0))
    corr4, intf_clim, av_clim = calculate_correlation_field_for_climatology(**params)
    to_plot4 = np.ma.masked_where(to_plot1.mask, corr4)
    title_list.append("Corr({}, {})".format(
        infovar.get_display_label_for_var(params["varname1"]),
        infovar.get_display_label_for_var(params["varname2"])))
    data_list.append(to_plot4)





    # TODO: Correlate infiltration and surface runoff

    # Do plotting
    clevels = np.arange(-1, 1.2, 0.2)

    npanels = len(data_list)
    gs = GridSpec(1, npanels + 1, width_ratios=[1.0, ] * npanels + [0.05, ])

    fig = plt.figure()
    assert isinstance(fig, Figure)
    fig.set_figheight(1.5 * fig.get_figheight())

    img = None
    for col in range(npanels):
        ax = fig.add_subplot(gs[0, col])
        basemap.drawmapboundary(fill_color="0.75", ax=ax)

        img = basemap.contourf(x, y, data_list[col], levels=clevels, cmap=cm.get_cmap("RdBu_r", len(clevels) - 1))
        plt.title(title_list[col])
        basemap.drawcoastlines(linewidth=cpp.COASTLINE_WIDTH, ax=ax)

    plt.colorbar(img, cax=fig.add_subplot(gs[0, npanels]))
    fig.savefig(os.path.join(img_folder, img_filename), dpi=cpp.FIG_SAVE_DPI)



    # plot timeseries
    the_mask = corr4 < -0.1
    varname_to_ts = {
        "INTF": get_mean_over(the_mask, intf_clim),
        "LH": get_mean_over(the_mask, av_clim),
        "SM": get_mean_over(the_mask, i1_clim)
    }

    from matplotlib import gridspec

    fig = plt.figure()
    fig.set_figheight(3 * fig.get_figheight())
    gs = gridspec.GridSpec(len(varname_to_ts), 1)

    d0 = datetime(2001, 1, 1)
    dt = timedelta(days=1)
    dates = [d0 + dt * i for i in range(365) if (d0 + dt * i).month in months]
    sfmt = ScalarFormatter()
    dfmt = DateFormatter("%d%b")
    for i, (label, data) in enumerate(varname_to_ts.items()):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(dates, data, label=label, lw=2)
        ax.grid()

        ax.legend()
        ax.yaxis.set_major_formatter(sfmt)
        if i < len(varname_to_ts) - 1:
            ax.xaxis.set_ticklabels([])
        else:
            ax.xaxis.set_major_formatter(dfmt)

    fig.savefig(os.path.join(img_folder, "aa_ts_{}_{}.png".format(os.path.basename(default_path),
                                                                  "-".join(str(m) for m in months))),
                dpi=cpp.FIG_SAVE_DPI)
    # plt.show()


def get_mean_over(mask, field_3d):
    """
    field_3d - is a 3 dimensional field (t, lon, lat)
    mask - is a 2d field (true - over the region of interest, false - otherwize)

    :type field_3d: np.ndarray
    """

    print(mask.shape, field_3d.shape)
    return np.mean((field_3d * mask[np.newaxis, :, :]), axis=1).mean(axis=1)


def demo_equal_fields():
    x = np.random.randn(20, 10, 10)
    y = np.random.randn(20, 10, 10)

    c = calculate_correlation(x, y)

    print(c.shape)
    print(c.min(), c.max())

    c = calculate_correlation(x, x)
    print(c.min(), c.max())

    c = calculate_correlation(np.sin(x), np.cos(x))
    print(c.min(), c.max())

    c = calculate_correlation(np.sin(x), np.sin(-x))
    print(c.min(), c.max())

    c = calculate_correlation_nd(x, y, axis=2)
    print(c.shape)


if __name__ == '__main__':
    import application_properties
    # Plot the last figure in paper 2.
    application_properties.set_current_directory()
    plot_utils.apply_plot_params(font_size=14, width_pt=None, width_cm=17, height_cm=5)

    seasons = (
        list(range(3, 12)),
    )

    start_year = 1980
    end_year = 2010

    for months in seasons:
        # plot_tmin_tmax_correlations(
        # start_year=start_year, end_year=end_year, months=months
        # )

        main(start_year=start_year,
             end_year=end_year,
             months=months)

        # demo_equal_fields()