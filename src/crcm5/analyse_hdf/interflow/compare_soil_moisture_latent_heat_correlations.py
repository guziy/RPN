import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use("Agg")

from collections import OrderedDict
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
import numpy as np

__author__ = 'san'

import matplotlib.pyplot as plt
from calculate_interflow_correlations import calculate_correlation_field_for_climatology
import crcm5.analyse_hdf.do_analysis_using_pytables as analysis
from mpl_toolkits.basemap import maskoceans


def plot_for_simulation(axis=None, sim_path="", cmap=None, cnorm=None,
                        start_year=None, end_year=None, months=None):
    """
    plot a panel for each simulation

    :param axis:
    :param sim_path:
    :param cmap:
    :param cnorm:
    """

    if months is None:
        months = range(1, 13)

    lons, lats, bm = analysis.get_basemap_from_hdf(sim_path)

    params = dict(
        path1=sim_path, path2=sim_path,
        start_year=start_year, end_year=end_year,
        varname1="I1", level1=0,
        varname2="AV", level2=0,
        months=months
    )

    corr, i1_clim, av_clim = calculate_correlation_field_for_climatology(**params)

    # convert longitudes to the [-180, 180] range
    lons[lons > 180] -= 360
    corr = maskoceans(lons, lats, corr)

    x, y = bm(lons, lats)

    im = bm.pcolormesh(x, y, corr, norm=cnorm, cmap=cmap, ax=axis)
    bm.drawcoastlines()
    return im, corr


def plot_correlation_diff(sim_label_to_corr, file_for_basemap="", ax=None, cnorm=None, cmap=None):
    lons, lats, bm = analysis.get_basemap_from_hdf(file_path=file_for_basemap)
    x, y = bm(lons, lats)
    im = bm.pcolormesh(x, y, sim_label_to_corr.values()[1] - sim_label_to_corr.values()[0], cmap=cmap, norm=cnorm)
    ax.set_title("(2) - (1)")
    bm.drawmapboundary(fill_color="0.75")
    return im


def main(months=None):
    """
    Note: if you want to compare more than 2 simulations at the same time
    make sure that the differences are plotted correctly
    :param months:
    """
    matplotlib.rc("font", size=20)
    # List of simulations to compare
    label_to_path = OrderedDict([
        ("(1) Without interflow", "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5"),
        ("(2) With interflow-a", "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5"),
        # ("With interflow (b)",
        # "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS_avoid_truncation1979-1989.hdf5")
    ])

    nsims = len(label_to_path)
    width_ratios = nsims * [1.0, ] + [0.05, ] + [1.3, 0.05]
    gs = GridSpec(1, nsims + 3, width_ratios=width_ratios, wspace=0.25)

    cdelta = 0.05
    clevels = np.arange(-1.0, 1.0 + cdelta, cdelta)
    cnorm = BoundaryNorm(clevels, len(clevels) - 1)
    cmap = cm.get_cmap("RdBu_r", len(clevels) - 1)

    start_year = 1980
    end_year = 1989

    keys = [k.replace(" ", "-") for k in label_to_path]
    keys.extend([str(i) for i in (start_year, end_year)])
    keys.append("-".join(str(m) for m in months))
    img_path = "corr-SM-LH_{}_{}_{}-{}_{}.png".format(*keys)

    fig = plt.figure()
    assert isinstance(fig, Figure)
    fig.set_figwidth(fig.get_figwidth() * 3)

    im = None
    sim_label_to_corr = OrderedDict()
    for col, (sim_label, sim_path) in enumerate(label_to_path.iteritems()):
        ax = fig.add_subplot(gs[0, col])
        im, corr = plot_for_simulation(axis=ax, sim_path=sim_path, cmap=cmap, cnorm=cnorm,
                                       start_year=start_year, end_year=end_year, months=months)
        ax.set_title(sim_label)
        sim_label_to_corr[sim_label] = corr

    plt.colorbar(im, cax=fig.add_subplot(gs[0, nsims]))

    # plot differences in correlation
    cdelta = 0.05
    clevels = np.arange(-0.5, -cdelta, cdelta)
    clevels = clevels.tolist() + [-c for c in reversed(clevels)]
    cnorm = BoundaryNorm(clevels, len(clevels) - 1)
    cmap = cm.get_cmap("RdBu_r", len(clevels) - 1)

    im = plot_correlation_diff(sim_label_to_corr, file_for_basemap=label_to_path.values()[0],
                               ax=fig.add_subplot(gs[0, nsims + 1]), cnorm=cnorm, cmap=cmap)

    plt.colorbar(im, cax=fig.add_subplot(gs[0, nsims + 2]))
    # fig.tight_layout()
    fig.savefig(img_path)

if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()
    seasons = (
        range(3, 6),
        range(6, 9),
        range(9, 12)
    )

    for months in seasons:
        main(months=months)