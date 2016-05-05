import os
from collections import namedtuple, OrderedDict

from mpl_toolkits.basemap import Basemap

from application_properties import main_decorator

import xarray as xr

from scipy import stats

import matplotlib.pyplot as plt

# Plot the climate change between the current and future intervals

# Since CanESM2 outputs do not have leap years -> assumes 365-day year

import calendar
import numpy as np

from crcm5.analyse_hdf import do_analysis_using_pytables as analysis

from crcm5.analyse_hdf import common_plot_params
from util import plot_utils


def _month_to_ndays(month, leap=False, year=None):
    if year is None or not leap:
        year = 2001

    wkd, ndays = calendar.monthrange(month=month, year=year)
    return ndays


@main_decorator
def main():

    # get the data for basemap
    crcm_data_path = "/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5"
    bmp_info = analysis.get_basemap_info_from_hdf(file_path=crcm_data_path)

    season_key = "JJA"
    season_to_months = OrderedDict([
        (season_key, [6, 7, 8]),
    ])

    month_to_ndays = {m: _month_to_ndays(m) for m in range(1, 13)}

    #
    current_filepath = "/RESCUE/skynet3_rech1/huziy/GCM_outputs/CanESM2/pr_Amon_CanESM2_historical_r1i1p1_185001-200512.nc"
    future_filepath = "/RESCUE/skynet3_rech1/huziy/GCM_outputs/CanESM2/pr_Amon_CanESM2_rcp85_r1i1p1_200601-210012.nc"

    Period = namedtuple("Period", ["start_year", "end_year"])

    current = Period(start_year=1980, end_year=2010)
    future = Period(start_year=2070, end_year=2100)


    ds = xr.open_mfdataset([current_filepath, future_filepath])


    # select the season
    ds = ds.isel(time=ds["time.season"] == season_key)

    # select the data for the current and future periods
    years = ds["time.year"]
    pr_current = ds.isel(time=(years >= current.start_year) & (years <= current.end_year)).pr
    pr_future = ds.isel(time=(years >= future.start_year) & (years <= future.end_year)).pr


    assert isinstance(pr_current, xr.DataArray)


    weights_current = xr.DataArray([month_to_ndays[m] for m in pr_current["time.month"].values], coords=[pr_current.time])
    weights_current = weights_current / weights_current.sum()

    weights_future = xr.DataArray([month_to_ndays[m] for m in pr_future["time.month"].values], coords=[pr_future.time])
    weights_future = weights_future / weights_future.sum()


    # seasonal means
    pr_current_smean = (pr_current * weights_current).groupby("time.year").sum(dim="time")
    pr_future_smean = (pr_future * weights_future).groupby("time.year").sum(dim="time")



    # climatology and stds
    pr_current_clim = pr_current_smean.mean(dim="year")
    pr_current_std = pr_current_smean.std(dim="year")

    pr_future_clim = pr_future_smean.mean(dim="year")
    pr_future_std = pr_future_smean.std(dim="year")




    # calculate significance
    n_current = current.end_year - current.start_year + 1
    n_future = future.end_year - future.start_year + 1
    tval, pval = stats.ttest_ind_from_stats(pr_current_clim.values, pr_current_std.values, nobs1=n_current,
                                            mean2=pr_future_clim.values, std2=pr_future_std.values, nobs2=n_future)


    print(weights_current[:3].values, weights_current[:3].sum())

    print(pr_current_smean.shape)

    print(pr_future.shape)
    print(pr_current.shape)
    print(ds["time.year"][-12:])


    # do the plotting
    plot_utils.apply_plot_params()
    fig = plt.figure()
    b = bmp_info.basemap
    xx, yy = bmp_info.get_proj_xy()

    lons, lats = np.meshgrid(ds.lon, ds.lat)

    xg, yg = b(lons, lats)

    dom_mask = (xg >= xx[0, 0]) & (xg <= xx[-1, -1]) & (yg >= yy[0, 0]) & (yg <= yy[-1, -1])

    i_list, j_list = np.where(dom_mask)

    imax, jmax = i_list.max(), j_list.max()
    imin, jmin = i_list.min(), j_list.min()

    marginx, marginy = 10, 10
    imax += marginx
    jmax += marginy
    imin -= marginx
    jmin -= marginy


    dom_mask[imin:imax, jmin:jmax] = True

    print(pr_current_clim.shape)
    print(ds.lon.shape)


    cchange = (pr_future_clim - pr_current_clim) * 24 * 3600  # Convert to mm/day

    cchange = np.ma.masked_where(~dom_mask, cchange)



    # cchange = np.ma.masked_where(pval > 0.1, cchange)


    plt.title("{}, (mm/day)".format(season_key))
    im = b.contourf(xg, yg, cchange)
    cb = b.colorbar(im)


    sign = np.ma.masked_where(~dom_mask, pval <= 0.05)

    cs = b.contourf(xg, yg, sign, levels=[0, 0.5, 1], hatches=["/", None, None], colors="none")

    b.drawcoastlines()


    # create a legend for the contour set
    artists, labels = cs.legend_elements()
    plt.legend([artists[0], ], ["not sign. (pvalue > 0.05)", ], handleheight=2)




    img_folder = "cc-paper-comments"
    fig.savefig(os.path.join(img_folder, "canesm_cc_{}_precip.png".format(season_key)), dpi=common_plot_params.FIG_SAVE_DPI / 2, bbox_inches="tight")
    plt.show()



if __name__ == '__main__':
    main()
