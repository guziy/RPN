from collections import OrderedDict
from datetime import datetime, timedelta

from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.dates import date2num, DateFormatter, MonthLocator
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
from rpn import level_kinds
from rpn.rpn import RPN
from rpn.rpn_multi import MultiRPN

from application_properties import main_decorator
from crcm5.nemo_vs_hostetler import commons
import numpy as np
from crcm5.nemo_vs_hostetler import nemo_hl_util
import os

from util import plot_utils
import matplotlib.pyplot as plt
import pandas as pd

img_folder = "nemo_vs_hostetler"


def get_nemo_lakes_mask(samples_dir=""):
    for mfolder in os.listdir(samples_dir):
        mfolder_path = os.path.join(samples_dir, mfolder)

        for fn in os.listdir(mfolder_path):

            if fn.startswith("pm") and fn[-9:-1] != 8 * "0":
                fp = os.path.join(mfolder_path, fn)
                r = RPN(fp)
                tlake = r.get_first_record_for_name("NEM1")
                r.close()

                return tlake > 0


def get_daily_clim_profiles(samples_dir, start_year=-np.Inf, end_year=np.Inf, filename_prefix="dp", varname="", mask=None):

    """
    dates_2d, levels_2d, values - needed for plotting
    :param start_year:
    :param end_year:
    :param filename_prefix:
    :param varname:
    """


    dates_sorted = None
    levels_sorted = None
    yearly_fields = []

    day = timedelta(days=1)
    stamp_dates = [datetime(2001, 1, 1) + i * day for i in range(365)]

    for y in range(start_year, end_year + 1):
        files_for_year = []

        mfolders = [os.path.join(samples_dir, f) for f in os.listdir(samples_dir) if f[:-2].endswith(str(y))]

        for mfolder in mfolders:
            files_for_year += [os.path.join(mfolder, fn) for fn in os.listdir(mfolder) if fn.startswith(filename_prefix)]


        mrpn = MultiRPN(files_for_year)
        data = mrpn.get_4d_field(varname=varname)

        # calculate the area-average
        for d, lev_to_field in data.items():
            for lev in lev_to_field:
                lev_to_field[lev] = lev_to_field[lev][mask].mean()

        #
        dates_sorted = list(sorted([d for d in data if not (d.month == 2 and d.day == 29) and (d.year == y)]))
        levels_sorted = list(sorted([lev for lev in data[dates_sorted[0]] if lev >= 500]))

        #
        year_field = np.zeros((len(dates_sorted), len(levels_sorted)))
        for j, lev in enumerate(levels_sorted):
            for i, the_date in enumerate(dates_sorted):
                year_field[i, j] = data[the_date][lev]

        panel = pd.DataFrame(data=year_field, index=dates_sorted, columns=range(len(levels_sorted)))

        daily = panel.groupby(by=lambda d: datetime(d.year, d.month, d.day)).mean()


        yearly_fields.append(daily.values)

    values = np.mean(yearly_fields, axis=0)


    assert not hasattr(values, "mask")

    dates_num = date2num(stamp_dates)

    levs_2d, dates_num_2d = np.meshgrid(levels_sorted, dates_num)

    assert levs_2d.shape == values.shape, "levels and values shapes ({} and {}, respectively) are not consistent.".format(levs_2d.shape, values.shape)

    return dates_num_2d, levs_2d, values



class MyMonthFormatter(DateFormatter):
    def __call__(self, *args, **kwargs):
        lbl = super(MyMonthFormatter, self).__call__(*args, **kwargs)
        return lbl[0]



@main_decorator
def main():
    start_year = 1980
    end_year = 1980

    HL_LABEL = "CRCM5_HL"
    NEMO_LABEL = "CRCM5_NEMO"

    sim_label_to_path = OrderedDict(
        [(HL_LABEL, "/RESCUE/skynet3_rech1/huziy/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl_oneway/Samples"),
         (NEMO_LABEL, "/HOME/huziy/skynet3_rech1/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl/Samples")]
    )

    var_name_list = ["TT", "HU"]

    season_to_months = commons.season_to_months


    vname_to_level_kind = {
        "TT": level_kinds.PRESSURE, "HU": level_kinds.PRESSURE
    }

    vname_to_file_prefix = {"TT": "dp", "HU": "dp"}

    vname_to_clevs_diff = {
        "TT": np.arange(-5.1, 5.3, 0.2),
        "PR": np.arange(-5.1, 5.3, 0.2),
        "SN": np.arange(-5.1, 5.3, 0.2),
        "LC": [v for v in np.arange(-0.52, 0.54, 0.08)],
        "HR": [v for v in np.arange(-0.52, 0.54, 0.03)],
        "HU": np.arange(-0.5, 0.54, 0.04),
        "AV": np.arange(-150, 170, 20),
        "I5": np.arange(-30, 34, 4),
    }




    vname_to_label = {
        "TT": "Air temperature",
        "PR": "Total precipitation",
        "HU": "Specific humidity"
    }

    vname_to_coeff = {
        "PR": 24 * 3600 * 1000,
        "HU": 1000
    }

    vname_to_units = {
        "TT": r"$^\circ$C",
        "PR": "mm/day",
        "HU": "g/kg",
        "AV": r"W/m$^2$",
        "I5": "mm"
    }

    avg_mask = get_nemo_lakes_mask(samples_dir=sim_label_to_path[NEMO_LABEL])



    # Do the calculations
    hl_data = OrderedDict()
    nemo_data = OrderedDict()

    dates_2d, levels_2d = None, None
    for vname in var_name_list:
        field_props = dict(start_year=start_year, end_year=end_year, filename_prefix=vname_to_file_prefix[vname], varname=vname, mask=avg_mask)
        dates_2d, levels_2d, hl_data[vname] = get_daily_clim_profiles(samples_dir=sim_label_to_path[HL_LABEL], **field_props)
        dates_2d, levels_2d, nemo_data[vname] = get_daily_clim_profiles(samples_dir=sim_label_to_path[NEMO_LABEL], **field_props)


    # Plotting
    plot_utils.apply_plot_params(font_size=6, width_cm=26, height_cm=16)
    fig = plt.figure()



    nrows = len(var_name_list)
    ncols = 3
    gs = GridSpec(nrows, ncols)




    for row, vname in enumerate(hl_data):

        row_axes = []

        # CRCM5_HL
        ax = fig.add_subplot(gs[row, 0])
        ax.set_title(HL_LABEL)

        cs = ax.contourf(dates_2d, levels_2d, hl_data[vname] * vname_to_coeff.get(vname, 1), 20, extend="both")
        plt.colorbar(cs, ax=ax)
        row_axes.append(ax)


        # CRCM5_NEMO
        ax = fig.add_subplot(gs[row, 1])
        ax.set_title(NEMO_LABEL)

        ax.contourf(dates_2d, levels_2d, nemo_data[vname] * vname_to_coeff.get(vname, 1), levels=cs.levels, norm=cs.norm, cmap=cs.cmap, extend="both")
        plt.colorbar(cs, ax=ax)
        row_axes.append(ax)


        ax = fig.add_subplot(gs[row, 2])
        norm = None

        if vname_to_clevs_diff[vname] is not None:
            norm = BoundaryNorm(vname_to_clevs_diff[vname], len(vname_to_clevs_diff[vname]) - 1)
            cmap = cm.get_cmap("seismic", len(vname_to_clevs_diff[vname]) - 1)
        else:
            cmap = cm.get_cmap("seismic", 11)

        cs = ax.contourf(dates_2d, levels_2d, (nemo_data[vname] - hl_data[vname]) * vname_to_coeff.get(vname, 1), cmap=cmap, norm=norm, levels=vname_to_clevs_diff[vname], extend="both")

        row_axes.append(ax)
        ax.set_ylabel(vname_to_label.get(vname, vname))
        ax.set_title("{} minus {}".format(NEMO_LABEL, HL_LABEL))

        cb = plt.colorbar(cs, ax=ax)
        cb.ax.set_title(vname_to_units.get(vname, "-"))



        for ax in row_axes:
            ax.set_yscale("log")
            ax.invert_yaxis()
            ax.set_yticks([1000, 900, 800, 700, 500])
            ax.get_yaxis().set_major_formatter(ScalarFormatter())

            ax.xaxis.set_major_formatter(MyMonthFormatter("%b"))

            ax.xaxis.set_major_locator(MonthLocator(bymonthday=15))
            ax.xaxis.set_minor_locator(MonthLocator(bymonthday=1))
            ax.grid(which="minor")



    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    img_file = os.path.join(img_folder, "profiles_{}-{}.png".format(start_year, end_year))
    fig.savefig(img_file, dpi=commons.dpi, transparent=True, bbox_inches="tight")







if __name__ == '__main__':
    main()