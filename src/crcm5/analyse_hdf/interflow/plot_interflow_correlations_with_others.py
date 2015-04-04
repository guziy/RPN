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
from util import plot_utils


__author__ = 'huziy'


def main(start_year=1980, end_year=2010, months=None, ylabel="",
         fig=None, current_row=0, gs=None):
    default_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5"
    # default_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS_avoid_truncation1979-1989.hdf5"

    if months is None:
        months = list(range(1, 13))

    img_folder = os.path.join("interflow_corr_images", os.path.basename(default_path))
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)

    img_filename = "4x1_correlations_INTF-PR-I1-Infilt_months={}_{}-{}.png".format("-".join(str(m) for m in months),
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
    # corr4 = calculate_correlation_of_infiltration_rate_with(start_year=start_year,
    #                                                         end_year=end_year,
    #                                                         path_for_infiltration_data=default_path,
    #                                                         path2=default_path,
    #                                                         varname2="INTF",
    #                                                         level2=0, months=months)

    # Correlate interflow rate with latent heat flux
    params = dict(
        path1=default_path,
        varname1="INTF",
        level1=0,

        path2=default_path,
        level2=0,
        varname2="AV",
        months=months,

        start_year=start_year, end_year=end_year,
    )

    corr4, _, _ = calculate_correlation_field_for_climatology(**params)
    to_plot4 = np.ma.masked_where(to_plot1.mask, corr4)
    title_list.append("Corr({}, {})".format(params["varname1"], params["varname2"]))
    data_list.append(to_plot4)

    # Do plotting
    clevels = np.arange(-1, 1.2, 0.2)


    npanels = len(data_list)

    if gs is None:
        gs = GridSpec(1, npanels + 1, width_ratios=[1.0, ] * npanels + [0.05, ])

    is_subplot = fig is not None
    fig = plt.figure() if fig is None else fig
    assert isinstance(fig, Figure)
    # fig.set_figheight(1.5 * fig.get_figheight())

    img = None
    for col in range(npanels):

        ax = fig.add_subplot(gs[current_row, col])

        if not col:
            ax.set_ylabel(ylabel)

        basemap.drawmapboundary(fill_color="0.75", ax=ax)

        img = basemap.contourf(x, y, data_list[col], levels=clevels, cmap=cm.get_cmap("RdBu_r", len(clevels) - 1))
        if current_row == 0:
            ax.set_title(title_list[col])
        basemap.drawcoastlines(linewidth=cpp.COASTLINE_WIDTH, ax=ax)

    if not is_subplot:
        plt.colorbar(img, cax=fig.add_subplot(gs[0, npanels]))
        fig.savefig(os.path.join(img_folder, img_filename), dpi=cpp.FIG_SAVE_DPI)

    return img, img_folder


def plot_correlations_for_seasons_as_subplots():
    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=20, height_cm=20)

    start_year = 1980
    end_year = 2010


    fig = plt.figure()

    gs = GridSpec(4, 5, width_ratios=[1.0, 1.0, 1.0, 1.0, 0.05])
    # Winter
    months = [12, 1, 2]
    main(start_year=start_year, end_year=end_year, months=months, ylabel="Winter",
         fig=fig, current_row=0, gs=gs)

    # Spring
    months = list(range(3, 6))
    main(start_year=start_year, end_year=end_year, months=months, ylabel="Spring",
         fig=fig, current_row=1, gs=gs)

    # Summer
    months = list(range(6, 9))
    main(start_year=start_year, end_year=end_year, months=months, ylabel="Summer",
         fig=fig, current_row=2, gs=gs)

    # Fall
    months = list(range(9, 12))
    img, imfolder = main(start_year=start_year, end_year=end_year, months=months, ylabel="Fall",
                         fig=fig, current_row=3, gs=gs)


    plt.colorbar(cax=fig.add_subplot(gs[:, 4]))
    plt.tight_layout()

    fig.savefig(os.path.join(imfolder, "corr_all_seasons_separately.png"))

if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()
    plot_correlations_for_seasons_as_subplots()
