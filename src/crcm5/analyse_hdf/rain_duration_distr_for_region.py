import os
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon

__author__ = 'huziy'


# Compare distributions of the durations of precipitation events


import matplotlib.pyplot as plt
import numpy as np


class Selection(object):

    def __init__(self, start_i=0, start_j=0, ni=0, nj=0):
        self.nj = nj
        self.ni = ni
        self.start_j = start_j
        self.start_i = start_i

    def ll_indices(self):
        return self.start_i, self.start_j

    def ur_indices(self):
        return self.start_i + self.ni - 1, self.start_j + self.nj - 1


class Period(object):

    def __init__(self, start_year=1980, end_year=2010):
        self.start_year = start_year
        self.end_year = end_year


    def get_nyears(self):
        return self.end_year - self.start_year + 1

    nyears = property(fget=get_nyears)


def get_duration_to_occurences_array(data_path, period=None, selection=None):

    assert isinstance(selection, Selection)
    assert isinstance(period, Period)







def main():
    from crcm5.analyse_hdf import do_analysis_using_pytables as analysis

    import application_properties
    application_properties.set_current_directory()

    # define the region of interest using indices
    sel = Selection(start_i=30, start_j=50, ni=40, nj=40)


    start_year = 1991
    end_year = 2010


    # Labels and paths to the data
    label_base = "CRCM5-NL"
    path_base = "/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-r.hdf5"


    labels = ["CRCM5-L1"]
    paths = ["/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r.hdf5"]

    bmp_info = analysis.get_basemap_info_from_hdf(file_path=path_base)


    # Do the calculations


    # Do the plotting

    fig = plt.figure()

    gs = GridSpec(2, 2)

    # Plot the domain
    ax = fig.add_subplot(gs[0, :])
    assert isinstance(ax, Axes)
    bmp_info.basemap.drawcoastlines(ax=ax)
    xx, yy = bmp_info.get_proj_xy()

    # add the region of interest to the map
    ll_i, ll_j = sel.ll_indices()
    ur_i, ur_j = sel.ur_indices()

    coords = ((ll_i, ll_j), (ll_i, ur_j), (ur_i, ur_j), (ur_i, ll_j))
    coords = [(xx[i, j], yy[i, j]) for (i, j) in coords]
    coords = np.array(coords)
    ax.add_patch(Polygon(coords, facecolor="none", linewidth=3))


    plt.show()
    # fig.savefig(os.path.expanduser("~/test.png"))




if __name__ == '__main__':
    main()
