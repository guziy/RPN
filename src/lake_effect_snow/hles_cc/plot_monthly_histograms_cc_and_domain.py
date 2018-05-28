

# plot monthly histograms for the CanESM2-driven simulations future vs current
from collections import OrderedDict

from datetime import datetime

import xarray
from matplotlib.dates import MonthLocator, num2date, date2num
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

from application_properties import main_decorator
from crcm5.nemo_vs_hostetler.main_for_lake_effect_snow import get_mask_of_points_near_lakes
from lake_effect_snow.hles_cc import common_params
from lake_effect_snow.hles_cc.plot_cc_2d_all_variables_for_all_periods import get_gl_mask
from lake_effect_snow.hles_cc.plot_domain_from_ncfile_using_cartopy import plot_domain_and_interest_region
from lake_effect_snow.plot_monthly_histograms import get_monthly_accumulations_area_avg
from util import plot_utils
import matplotlib.pyplot as plt
import numpy as np


@main_decorator
def main(varname=""):
    plot_utils.apply_plot_params(width_cm=22, height_cm=5, font_size=8)
    # series = get_monthly_accumulations_area_avg(data_dir="/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_Obs_monthly_1980-2009",
    #                                             varname=varname)

    # series = get_monthly_accumulations_area_avg(data_dir="/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_NEMO_1980-2009_monthly",
    #                                             varname=varname)

    # series = get_monthly_accumulations_area_avg(data_dir="/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_HL_1980-2009_monthly",
    #                                             varname=varname)

    selected_months = [9, 10, 11, 12, 1, 2, 3, 4, 5]



    data_root = common_params.data_root

    label_to_datapath = OrderedDict([
        # ("Obs", "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_Obs_monthly_1980-2009"),
        # ("Obs", "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_daily_Obs_monthly_icefix_1980-2009"),
        (common_params.crcm_nemo_cur_label, data_root / "lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_1989-2010_1989-2010"),
        (common_params.crcm_nemo_fut_label, data_root / "lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_2079-2100_2079-2100"),
    ])


    label_to_series = OrderedDict()
    label_to_color = {
        common_params.crcm_nemo_cur_label: "skyblue",
        common_params.crcm_nemo_fut_label: "salmon"

    }

    gl_mask = get_gl_mask(label_to_datapath[common_params.crcm_nemo_cur_label] / "merged")
    hles_region_mask = get_mask_of_points_near_lakes(gl_mask, npoints_radius=20)

    # select a file from the directory
    sel_file = None
    for f in label_to_datapath[common_params.crcm_nemo_cur_label].iterdir():
        sel_file = f

    with xarray.open_dataset(sel_file) as ds:
        hles_region_mask_lons, hles_region_mask_lats = [ds[k].values for k in ["lon", "lat"]]

    for label, datapath in label_to_datapath.items():
        series = get_monthly_accumulations_area_avg(data_dir=datapath, varname=varname,
                                                    fname_suffix="_daily.nc",
                                                    region_of_interest_mask=hles_region_mask)
        label_to_series[label] = series


    #
    # print(series)
    # assert isinstance(series, pd.Series)
    # ax = series.plot(kind="bar", width=1)
    #
    # ax.set_ylabel("%")
    # ax.set_xlabel("Month")

    gs = GridSpec(1, 2, wspace=0.)

    fig = plt.figure()
    ax = fig.add_subplot(gs[0, 1])

    start_date = datetime(2001, 10, 1)

    dates = [start_date.replace(month=(start_date.month + i) % 13 + int((start_date.month + i) % 13 == 0),
                                year=start_date.year + (start_date.month + i) // 13) for i in range(13)]


    def format_month_label(x, pos):
        print(num2date(x))
        return "{:%b}".format(num2date(x))



    # calculate bar widths
    dates_num = date2num(dates)
    width = np.diff(dates_num) / (len(label_to_series) * 1.5)
    width = np.array([width[0] for _ in width])


    # select the months
    width = np.array([w for w, d in zip(width, dates) if d.month in selected_months])
    dates = [d for d in dates[:-1] if d.month in selected_months]
    dates_num = date2num(dates)


    label_to_handle = OrderedDict()


    label_to_annual_hles = OrderedDict()

    for i, (label, series) in enumerate(label_to_series.items()):
        values = [series[d.month] for d in dates]

        # convert to percentages
        values_sum = sum(values)

        # save the total annual hles for later reuse
        label_to_annual_hles[label] = values_sum

        values = [v / values_sum * 100 for v in values]

        print(label, values)
        print(f"sum(values) = {sum(values)}")

        h = ax.bar(dates_num + i * width, values, width=width, align="edge", linewidth=0.5,
               edgecolor="k", facecolor=label_to_color[label], label=label, zorder=10)
        label_to_handle[label] = h

    ax.set_ylabel("% of total HLES")
    ax.set_title("(b) Monthly HLES distribution")


    ax.xaxis.set_major_formatter(FuncFormatter(func=format_month_label))
    ax.xaxis.set_major_locator(MonthLocator(bymonthday=int(sum(width[:len(label_to_series)]) / 2.) + 1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_title(common_params.varname_to_display_name[varname])
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5)
    # ax.text(1, 1, "(a)", fontdict=dict(weight="bold"), transform=ax.transAxes, va="top", ha="right")
    ax_with_legend = ax

    # area average annual total HLES
    text_align_props = dict(transform=ax.transAxes, va="bottom", ha="right")

    cur_hles_annual = label_to_annual_hles[common_params.crcm_nemo_cur_label]
    fut_hles_annual = label_to_annual_hles[common_params.crcm_nemo_fut_label]

    ax.text(1, 0.2, r"$\Delta_{\rm total}$" + f"({(fut_hles_annual - cur_hles_annual) / cur_hles_annual * 100:.1f}%)",
            **text_align_props, fontdict=dict(size=6))

    # Plot the domain and the HLES region of interest
    ax = fig.add_subplot(gs[0, 0])
    topo_nc_file = data_root / "geophys_452x260_me.nc"
    ax = plot_domain_and_interest_region(ax, topo_nc_file)
    ax.set_title("(a) Experimental domain")


    # Add a common legend
    labels = list(label_to_handle)
    handles = [label_to_handle[l] for l in labels]
    ax_with_legend.legend(handles, labels, bbox_to_anchor=(0, -0.18), loc="upper left", borderaxespad=0., ncol=2)

    # ax.grid()
    sel_months_str = "_".join([str(m) for m in selected_months])

    common_params.img_folder.mkdir(exist_ok=True)
    img_file = common_params.img_folder / f"{varname}_histo_cc_m{sel_months_str}_domain.png"
    print(f"Saving plot to {img_file}")
    fig.savefig(img_file, **common_params.image_file_options)


if __name__ == '__main__':
    # main(varname="hles_snow")

    for varname in ["hles_snow", ]:
        main(varname=varname)