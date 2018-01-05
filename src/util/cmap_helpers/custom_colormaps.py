from matplotlib import cm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def get_with_white_added(cm_name_default, white_start=0.0, white_end=0.1,
                         ncolors_out=100):

    cmap = cm.get_cmap(cm_name_default)
    ncolors = cmap.N


    clist = []

    lower = []
    if white_start > 0:
        lower = cmap(np.linspace(0, white_start, int(white_start * ncolors)))
    clist.append(lower)

    white = np.ones((int((white_end - white_start) * ncolors), 4))
    clist.append(white)

    upper = []
    if white_end < 1:
        upper = cmap(np.linspace(white_end, 1, int((1 - white_end) * ncolors)))
    clist.append(upper)

    colors = np.vstack(tuple([p for p in clist if len(p) > 0]))

    return LinearSegmentedColormap.from_list("mycmap", colors, N=ncolors_out)

