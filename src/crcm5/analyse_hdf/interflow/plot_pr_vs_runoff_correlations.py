from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import maskoceans
from crcm5.analyse_hdf.interflow.calculate_interflow_correlations import calculate_correlation_field_for_climatology
from util import plot_utils

__author__ = 'huziy'

import os
import numpy as np
import crcm5.analyse_hdf.do_analysis_using_pytables as analysis
import matplotlib.pyplot as plt
import crcm5.analyse_hdf.common_plot_params as cpp


def plot_pr_runoff_correlations(start_year=1980, end_year=2010, months=None):
    default_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5"
    # default_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS_avoid_truncation1979-1989.hdf5"

    if months is None:
        months = range(1, 13)

    img_folder = os.path.join("interflow_corr_images", os.path.basename(default_path))
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)

    img_filename = "PR_TRAF_TDRA_interflow_correlations_months={}_{}-{}.jpg".format("-".join(str(m) for m in months),
                                                                                    start_year, end_year)

    lons, lats, basemap = analysis.get_basemap_from_hdf(file_path=default_path)
    lons[lons > 180] -= 360
    x, y = basemap(lons, lats)

    # Correlate surface runoff and PR
    params = dict(
        path1=default_path,
        varname1="PR",
        level1=0,

        path2=default_path,
        level2=0,
        varname2="TRAF",
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



    # correlate drainage and PR
    params.update(dict(varname2="TDRA", level2=0, varname1="PR", level1=0))
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

if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()
    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=17, height_cm=5)

    seasons = (
        range(3, 6), range(6, 9), range(9, 12)
    )

    start_year = 1980
    end_year = 2010

    for months in seasons:
        plot_pr_runoff_correlations(
            start_year=start_year, end_year=end_year, months=months
        )
