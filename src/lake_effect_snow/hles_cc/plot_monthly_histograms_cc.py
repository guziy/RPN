

# plot monthly histograms for the CanESM2-driven simulations future vs current
from collections import OrderedDict

from datetime import datetime
from matplotlib.dates import MonthLocator, num2date, date2num
from matplotlib.ticker import FuncFormatter

from lake_effect_snow.plot_monthly_histograms import get_monthly_accumulations_area_avg
from util import plot_utils
import matplotlib.pyplot as plt
import numpy as np


def main(varname=""):
    plot_utils.apply_plot_params(width_cm=8, height_cm=5.5, font_size=8)
    # series = get_monthly_accumulations_area_avg(data_dir="/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_Obs_monthly_1980-2009",
    #                                             varname=varname)

    # series = get_monthly_accumulations_area_avg(data_dir="/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_NEMO_1980-2009_monthly",
    #                                             varname=varname)

    # series = get_monthly_accumulations_area_avg(data_dir="/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_HL_1980-2009_monthly",
    #                                             varname=varname)

    selected_months = [9, 10, 11, 12, 1, 2, 3, 4, 5]


    label_to_datapath = OrderedDict([
        # ("Obs", "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_Obs_monthly_1980-2009"),
        # ("Obs", "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_daily_Obs_monthly_icefix_1980-2009"),
        ("CRCM5_NEMOc", "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_1989-2010_1989-2010"),
        ("CRCM5_NEMOf", "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_2079-2100_2079-2100"),
    ])


    label_to_series = OrderedDict()
    label_to_color = {
        "CRCM5_NEMOc": "skyblue",
        "CRCM5_NEMOf": "salmon"

    }
    for label, datapath in label_to_datapath.items():
        series = get_monthly_accumulations_area_avg(data_dir=datapath, varname=varname, fname_suffix="_daily.nc")

        label_to_series[label] = series


    #
    # print(series)
    # assert isinstance(series, pd.Series)
    # ax = series.plot(kind="bar", width=1)
    #
    # ax.set_ylabel("%")
    # ax.set_xlabel("Month")

    fig = plt.figure()
    ax = plt.gca()

    start_date = datetime(2001, 10, 1)

    dates = [start_date.replace(month=(start_date.month + i) % 13 + int((start_date.month + i) % 13 == 0),
                                year=start_date.year + (start_date.month + i) // 13) for i in range(13)]


    for d in dates:
        print(d)



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


    for i, (label, series) in enumerate(label_to_series.items()):
        values = [series[d.month] for d in dates]

        # convert to percentages
        values_sum = sum(values)
        values = [v / values_sum * 100 for v in values]

        print(label, values)
        print(f"sum(values) = {sum(values)}")

        ax.bar(dates_num + i * width, values, width=width, align="edge", linewidth=0.5,
               edgecolor="k", facecolor=label_to_color[label], label=label)

    ax.set_ylabel("%")


    ax.xaxis.set_major_formatter(FuncFormatter(func=format_month_label))
    ax.xaxis.set_major_locator(MonthLocator(bymonthday=int(sum(width[:len(label_to_series)]) / 2.) + 1))
    ax.legend(bbox_to_anchor=(0, -0.18), loc="upper left", borderaxespad=0., ncol=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    print(width[:len(label_to_series)])

    # ax.grid()
    sel_months_str = "_".join([str(m) for m in selected_months])
    img_file = f"{varname}_histo_cc_m{sel_months_str}.png"
    print(f"Saving plot to {img_file}")
    fig.savefig(img_file, bbox_inches="tight", dpi=400)


if __name__ == '__main__':
    # main(varname="hles_snow")
    main(varname="lake_ice_fraction")