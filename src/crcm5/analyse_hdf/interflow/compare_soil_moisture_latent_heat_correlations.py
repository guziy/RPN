import matplotlib
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

    bm, lons, lats = analysis.get_basemap_from_hdf(sim_path)

    params = dict(
        path1=sim_path, path2=sim_path,
        start_year=start_year, end_year=end_year,
        varname1="I0", level1=0,
        varname2="AV", level2=0,
        months=months
    )

    corr = calculate_correlation_field_for_climatology(**params)

    x, y = bm(lons, lats)
    im = bm.pcolormesh(x, y, corr, norm=cnorm, cmap=cmap, ax=axis)
    bm.drawcoastlines()
    return im


def main():
    # List of simulations to compare
    label_to_path = OrderedDict([
        ("Without interflow", "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5"),
        ("With interflow", "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5"),
    ])

    nsims = len(label_to_path)
    width_ratios = nsims * [1.0, ] + [0.05, ]
    gs = GridSpec(1, nsims + 1, width_ratios=width_ratios)

    cdelta = 0.2
    clevels = np.arange(-1.0, 1.0 + cdelta, cdelta)
    cnorm = BoundaryNorm(clevels, len(clevels) - 1)
    cmap = cm.get_cmap("RdBu_r", len(clevels) - 1)


    start_year = 1980
    end_year = 2010
    months = range(3, 6)


    keys = [k.replace(" ", "-") for k in label_to_path]
    keys.extend([str(i) for i in (start_year, end_year)])
    img_path = "corr-SM-LH_{}_{}_{}-{}.png".format(*keys)

    fig = plt.figure()

    im = None
    for col, sim_path in enumerate(label_to_path.values()):
        ax = fig.add_subplot(gs[0, col])
        im = plot_for_simulation(axis=ax, sim_path=sim_path, cmap=cmap, cnorm=cnorm,
                                 start_year=start_year, end_year=end_year, months=months)

    plt.colorbar(im, cax=fig.add_subplot(gs[0, nsims]), ticks=clevels)

    fig.savefig(img_path)


if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()
    main()