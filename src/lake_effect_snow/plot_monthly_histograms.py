import calendar
import glob
from collections import OrderedDict
from datetime import datetime

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import num2date, date2num, MonthLocator
from matplotlib.ticker import FuncFormatter

from util import plot_utils


def get_monthly_accumulations_area_avg_from_merged(data_file, varname="snow_fall", region_of_interest_mask=None):
    months_of_interest = list(range(1, 13))
    months_to_accumulations = {}

    with xr.open_dataset(data_file) as ds:
        da = ds[varname]

        # monthly accumulations of hles
        da_monthly = da.resample(t="1M").sum(dim="t")

        da_monthclim = da_monthly.groupby("t.month").mean(dim="t")

        for m in months_of_interest:
            data = da_monthclim.sel(month=m).values
            if region_of_interest_mask is None:
                months_to_accumulations[m] = data[~np.isnan(data)].mean()
            else:
                months_to_accumulations[m] = data[region_of_interest_mask].mean()

    return pd.Series(data=[months_to_accumulations[m] for m in months_of_interest], index=months_of_interest)



def get_monthly_accumulations_area_avg(data_dir="/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_Obs_monthly_1980-2009",
                                       varname="snow_fall", fname_suffix=".nc",
                                       region_of_interest_mask=None):

    month_to_accumulations = {}

    months_of_interest = list(range(1, 13))

    for month in months_of_interest:

        files = glob.glob(f"{data_dir}/*m{month:02d}-{month:02d}{fname_suffix}")
        if len(files) == 0:
            files = glob.glob(f"{data_dir}/*m{month}-{month}{fname_suffix}")

        print(f"Trying to read data from {files}")

        with xr.open_mfdataset(files) as ds:

            da = ds[varname]

            print(da.shape)
            print("---" * 10)

            if "year" in da.dims:
                da2d = da.mean(dim="year").values
            else:
                da2d = da.mean(dim="t").values

            if region_of_interest_mask is None:
                month_to_accumulations[month] = da2d[~np.isnan(da2d)].mean()
            else:
                month_to_accumulations[month] = da2d[region_of_interest_mask].mean()



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

    selected_months = [9, 10, 11, 12, 1, 2, 3, 4, 5]


    label_to_datapath = OrderedDict([
        #("Obs", "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_Obs_monthly_1980-2009"),
        ("Obs", "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_daily_Obs_monthly_icefix_1980-2009"),
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
        series = get_monthly_accumulations_area_avg_from_merged(data_dir=datapath, varname=varname)

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
    #ax.set_xlabel("Month")

    ax.xaxis.set_major_formatter(FuncFormatter(func=format_month_label))
    ax.xaxis.set_major_locator(MonthLocator(bymonthday=int(sum(width[:len(label_to_series)]) / 2.) + 1))
    ax.legend(bbox_to_anchor=(0, -0.18), loc="upper left", borderaxespad=0., ncol=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    print(width[:len(label_to_series)])

    # ax.grid()
    img_file = "hles_histo_all_m{}.png".format("_".join([str(m) for m in selected_months]))
    print(f"Saving plot to {img_file}")
    fig.savefig(img_file, bbox_inches="tight", dpi=400)




if __name__ == '__main__':
    main()