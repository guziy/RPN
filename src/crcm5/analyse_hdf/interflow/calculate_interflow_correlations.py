import matplotlib
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

    data1_norm = (data1 - data1.mean(axis=0)) / (data1.std(axis=0) + 1e-10 * data1.std(axis=0).max())
    data2_norm = (data2 - data2.mean(axis=0)) / (data2.std(axis=0) + 1e-10 * data2.std(axis=0).max())

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
        months = range(1, 13)

    selfields1 = [f for date, f in zip(dates, data1) if date.month in months]
    selfields2 = [f for date, f in zip(dates, data2) if date.month in months]

    return calculate_correlation(selfields1, selfields2)


def calculate_correlation_of_infiltration_rate_with(start_year=None,
                                                    end_year=None,
                                                    path_for_infiltration_data="",
                                                    path2="",
                                                    varname2="",
                                                    level2=None, months=None):
    dates, pr_data = analysis.get_daily_climatology(path_to_hdf_file=path_for_infiltration_data, var_name="PR",
                                                    level=None,
                                                    start_year=start_year, end_year=end_year)

    # Take interflow calculated for soil subareas
    dates, srunoff_data = analysis.get_daily_climatology(path_to_hdf_file=path_for_infiltration_data, var_name="TRAF",
                                                         level=1,
                                                         start_year=start_year, end_year=end_year)

    dates, evap_data = analysis.get_daily_climatology(path_to_hdf_file=path_for_infiltration_data, var_name="AV",
                                                      level=None,
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
        months = range(1, 13)

    selfields1 = [f for date, f in zip(dates, infiltration) if date.month in months]
    selfields2 = [f for date, f in zip(dates, data2) if date.month in months]

    return calculate_correlation(selfields1, selfields2)


def main(start_year=1980, end_year=2010):
    default_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5"
    # default_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS_avoid_truncation1979-1989.hdf5"

    months = range(3, 6)

    img_filename = "interflow_correlations_months={}.jpg".format("-".join(str(m) for m in months))

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

    corr1 = calculate_correlation_field_for_climatology(**params)
    to_plot1 = maskoceans(lons, lats, corr1)
    title_list.append("Corr({}, {})".format(params["varname1"], params["varname2"]))
    data_list.append(to_plot1)

    # correlate interflow and precip
    params.update(dict(varname2="PR", level2=0))
    corr2 = calculate_correlation_field_for_climatology(**params)
    to_plot2 = np.ma.masked_where(to_plot1.mask, corr2)
    title_list.append("Corr({}, {})".format(params["varname1"], params["varname2"]))
    data_list.append(to_plot2)

    # correlate precip and soil moisture
    params.update(dict(varname1="I1", level1=0))
    corr3 = calculate_correlation_field_for_climatology(**params)
    to_plot3 = np.ma.masked_where(to_plot2.mask, corr3)
    title_list.append("Corr({}, {})".format(params["varname1"], params["varname2"]))
    data_list.append(to_plot3)

    # correlate interflow and evaporation
    params.update(dict(varname2="AV", level2=0, varname1="INTF", level1=0))
    corr4 = calculate_correlation_field_for_climatology(**params)
    to_plot4 = np.ma.masked_where(to_plot2.mask, corr4)
    title_list.append("Corr({}, {})".format(params["varname1"], params["varname2"]))
    data_list.append(to_plot4)

    # TODO: Correlate infiltration and surface runoff

    # Do plotting
    clevels = np.arange(-1, 1.2, 0.2)
    gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.05])

    fig = plt.figure()

    img = None
    for col in range(4):
        ax = fig.add_subplot(gs[0, col])
        basemap.drawmapboundary(fill_color="0.75", ax=ax)
        img = basemap.contourf(x, y, data_list[col], levels=clevels, cmap=cm.get_cmap("RdBu_r", len(clevels) - 1))
        plt.title(title_list[col])
        basemap.drawcoastlines(linewidth=cpp.COASTLINE_WIDTH, ax=ax)

    plt.colorbar(img, cax=fig.add_subplot(gs[0, 4]))
    fig.savefig(img_filename, dpi=cpp.FIG_SAVE_DPI)
    # plt.show()


def demo_equal_fields():
    x = np.random.randn(20, 10, 10)
    y = np.random.randn(20, 10, 10)

    c = calculate_correlation(x, y)

    print c.shape
    print c.min(), c.max()

    c = calculate_correlation(x, x)
    print c.min(), c.max()

    c = calculate_correlation(np.sin(x), np.cos(x))
    print c.min(), c.max()

    c = calculate_correlation(np.sin(x), np.sin(-x))
    print c.min(), c.max()

    c = calculate_correlation_nd(x, y, axis=2)
    print c.shape


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()
    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=17, height_cm=5)
    main(start_year=1980, end_year=1989)
    # demo_equal_fields()