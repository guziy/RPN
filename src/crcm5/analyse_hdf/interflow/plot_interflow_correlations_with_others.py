from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import maskoceans
from crcm5.analyse_hdf.interflow.calculate_interflow_correlations import calculate_correlation_field_for_climatology, \
    calculate_correlation_of_infiltration_rate_with

import numpy as np
import os
import matplotlib.pyplot as plt
import crcm5.analyse_hdf.common_plot_params as cpp
import crcm5.analyse_hdf.do_analysis_using_pytables as analysis


__author__ = 'huziy'


def main(start_year=1980, end_year=2010, months=None):
    default_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5"
    # default_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS_avoid_truncation1979-1989.hdf5"

    if months is None:
        months = range(1, 13)

    img_folder = os.path.join("interflow_corr_images", os.path.basename(default_path))
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)

    img_filename = "4x1_correlations_INTF-PR-I1-Infilt_months={}_{}-{}.jpg".format("-".join(str(m) for m in months),
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
    title_list.append("Corr({}, {})".format(params["varname1"], params["varname2"]))
    data_list.append(to_plot1)

    # correlate interflow and precip
    params.update(dict(varname2="PR", level2=0))
    corr2, _, pr_clim = calculate_correlation_field_for_climatology(**params)
    to_plot2 = np.ma.masked_where(to_plot1.mask, corr2)
    title_list.append("Corr({}, {})".format(params["varname1"], params["varname2"]))
    data_list.append(to_plot2)

    # correlate precip and soil moisture
    params.update(dict(varname1="I1", level1=0))
    corr3, _, _ = calculate_correlation_field_for_climatology(**params)
    to_plot3 = np.ma.masked_where(to_plot2.mask, corr3)
    title_list.append("Corr({}, {})".format(params["varname1"], params["varname2"]))
    data_list.append(to_plot3)

    # correlate interflow and infiltration
    corr4 = calculate_correlation_of_infiltration_rate_with(start_year=start_year,
                                                            end_year=end_year,
                                                            path_for_infiltration_data=default_path,
                                                            path2=default_path,
                                                            varname2="INTF",
                                                            level2=0, months=months)

    to_plot4 = np.ma.masked_where(to_plot1.mask, corr4)
    title_list.append("Corr({}, {})".format(params["varname1"], "Infiltr."))
    data_list.append(to_plot4)

    # Do plotting
    clevels = np.arange(-1, 1.1, 0.1)

    npanels = len(data_list)
    gs = GridSpec(1, npanels + 1, width_ratios=[1.0, ] * npanels + [0.05, ])

    fig = plt.figure()
    assert isinstance(fig, Figure)
    # fig.set_figheight(1.5 * fig.get_figheight())

    img = None
    for col in range(npanels):
        ax = fig.add_subplot(gs[0, col])
        basemap.drawmapboundary(fill_color="0.75", ax=ax)

        img = basemap.contourf(x, y, data_list[col], levels=clevels, cmap=cm.get_cmap("RdBu_r", len(clevels) - 1))
        plt.title(title_list[col])
        basemap.drawcoastlines(linewidth=cpp.COASTLINE_WIDTH, ax=ax)

    plt.colorbar(img, cax=fig.add_subplot(gs[0, npanels]))
    fig.savefig(os.path.join(img_folder, img_filename), dpi=cpp.FIG_SAVE_DPI)


if __name__ == '__main__':
    import application_properties
    font_size = 25
    params = {
        'axes.labelsize': font_size,
        'font.size': font_size,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'figure.figsize': (18, 4),
        "axes.titlesize": font_size
    }

    plt.rcParams.update(params)


    application_properties.set_current_directory()
    start_year = 1980
    end_year = 2010
    months = range(3, 12)
    main(start_year=start_year, end_year=end_year, months=months)