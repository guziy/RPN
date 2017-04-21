import calendar
from collections import OrderedDict
from datetime import datetime

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import num2date, date2num, DateLocator, MonthLocator
from matplotlib.ticker import FuncFormatter

from util import plot_utils


def get_monthly_accumulations_area_avg(data_dir="/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_Obs_monthly_1980-2009",
                                       varname="snow_fall"):



    month_to_accumulations = {}

    months_of_interest = list(range(1, 13))

    for month in months_of_interest:
        with xr.open_mfdataset("{}/*m{}-{}.nc".format(data_dir, month, month)) as ds:
            da = ds[varname]

            print(ds["year"])
            print(da.shape)
            print("---" * 10)
            da2d = da.mean(dim="year").values
            month_to_accumulations[month] = da2d[~np.isnan(da2d)].mean()


    series = pd.Series(data=[month_to_accumulations[m] for m in months_of_interest], index=months_of_interest)


    # convert to percentages
    # series /= series.sum()
    # series *= 100

    return series






def main(varname="snow_fall"):


    plot_utils.apply_plot_params(width_cm=8, height_cm=5.5, font_size=8)
    # series = get_monthly_accumulations_area_avg(data_dir="/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_Obs_monthly_1980-2009",
    #                                             varname=varname)

    # series = get_monthly_accumulations_area_avg(data_dir="/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_NEMO_1980-2009_monthly",
    #                                             varname=varname)

    # series = get_monthly_accumulations_area_avg(data_dir="/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_HL_1980-2009_monthly",
    #                                             varname=varname)

    selected_months = [11, 12, 1]


    label_to_datapath = OrderedDict([
        ("Obs", "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_Obs_monthly_1980-2009"),
        ("CRCM5_NEMO", "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_NEMO_1980-2009_monthly"),
        ("CRCM5_HL", "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_HL_1980-2009_monthly"),
    ])


    label_to_series = OrderedDict()
    label_to_color = {
        "Obs": "skyblue",
        "CRCM5_HL": "yellowgreen",
        "CRCM5_NEMO": "salmon"

    }
    for label, datapath in label_to_datapath.items():
        series = get_monthly_accumulations_area_avg(data_dir=datapath, varname=varname)

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

    dates = [start_date.replace(month=(start_date.month + i) % 13 + int((start_date.month + i) % 13 == 0), year=start_date.year + (start_date.month + i) // 13) for i in range(13)]


    for d in dates:
        print(d)



    def format_month_label(x, pos):
        print(num2date(x))
        return "{:%b}".format(num2date(x))



    # calculate bar widths
    dates_num = date2num(dates)
    width = np.diff(dates_num) / (len(label_to_series) * 1.5)


    # select the months
    width = np.array([w for w, d in zip(width, dates) if d.month in selected_months])
    dates = [d for d in dates[:-1] if d.month in selected_months]
    dates_num = date2num(dates)


    for i, (label, series) in enumerate(label_to_series.items()):
        values = [series[d.month] for d in dates]

        # convert to percentages
        values_sum = sum(values)
        values = [v / values_sum * 100 for v in values]

        ax.bar(dates_num + i * width, values, width=width, align="edge", linewidth=0.5, edgecolor="k", facecolor=label_to_color[label], label=label)



    ax.set_ylabel("%")
    #ax.set_xlabel("Month")

    ax.xaxis.set_major_formatter(FuncFormatter(func=format_month_label))
    ax.xaxis.set_major_locator(MonthLocator(bymonthday=int(sum(width) / 2 + 0.5)))
    ax.legend(bbox_to_anchor=(0, -0.18), loc="upper left", borderaxespad=0., ncol=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    #ax.grid()
    fig.savefig("hles_histo_all_m{}.png".format("_".join([str(m) for m in selected_months])), bbox_inches="tight", dpi=400)




if __name__ == '__main__':
    main()