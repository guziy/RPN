from datetime import datetime

from matplotlib.axes import Axes

from application_properties import main_decorator

import matplotlib.pyplot as plt

from crcm5.analyse_hdf import do_analysis_using_pytables as analysis
from util import plot_utils

from crcm5.analyse_hdf import common_plot_params
import numpy as np


@main_decorator
def main():

    data_path = "/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5"
    start_year = 1980
    end_year = 2010
    vname = "TRAF"
    level_index = 0


    fldr = analysis.get_array_from_file(data_path, var_name="flow_direction")
    lkfr = analysis.get_array_from_file(data_path, var_name="lake_fraction")

    the_mask = np.ma.masked_all_like(fldr)

    the_mask[fldr > 0] = (1 - lkfr)[fldr > 0]


    ser = analysis.get_area_mean_timeseries(hdf_path=data_path, var_name=vname, level_index=level_index,
                                            start_year=start_year, end_year=end_year, the_mask=the_mask)



    monthly_ser = ser.groupby(lambda d: datetime(d.year, d.month, 15)).mean()





    # do the plotting
    plot_utils.apply_plot_params()
    fig = plt.figure()

    monthly_ser = monthly_ser * 24 * 3600  # convert to mm/day

    monthly_ser.groupby(lambda d: d.month).plot()
    ax = plt.gca()
    assert isinstance(ax, Axes)
    ax.grid()

    fig.savefig(data_path[:-5] + "_{}_level_index_{}_{}-{}_timeseries.png".format(vname, level_index, start_year, end_year),
                transparent=True, dpi=common_plot_params.FIG_SAVE_DPI, bbox_inches="tight")

    plt.show()


if __name__ == '__main__':
    main()
