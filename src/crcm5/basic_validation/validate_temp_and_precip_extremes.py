




# compare 90percentile of Tmin and 10th percentile of
# preprocess data
# For Tmin and Tmax - 5-day moving window centered on each calendar day
# For PR - 29-day moving window
from multiprocessing.pool import ThreadPool

import dask
import matplotlib
from memory_profiler import profile
from scipy.stats import ttest_ind_from_stats

matplotlib.use("Agg")


from collections import OrderedDict
from pathlib import Path

from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap, maskoceans
from rpn import level_kinds

from crcm5.basic_validation.diag_manager import DiagCrcmManager
from cru.temperature import CRUDataManager
from data.highres_data_manager import HighResDataManager
from lake_effect_snow.base_utils import VerticalLevel
from util import plot_utils
from util.seasons_info import MonthPeriod

import numpy as np
import matplotlib.pyplot as plt


var_name_to_level = {
    "TT": VerticalLevel(1, level_type=level_kinds.HYBRID),
    "PR": VerticalLevel(-1, level_type=level_kinds.ARBITRARY)
}




clevs = {
    "mean": {
        "TT": list(np.arange(-5, 0, 1)) + list(np.arange(1, 6, 1)),
        "TT_min": list(np.arange(-5, 0, 1)) + list(np.arange(1, 6, 1)),
        "TT_max": list(np.arange(-5, 0, 1)) + list(np.arange(1, 6, 1)),
        "PR_max": list(np.arange(-5, -0.5, 1)) + [-0.5, 0.5] + list(np.arange(1, 6, 1))
    },
    "std": {
        "TT": list(np.arange(-3, -0.0, 0.5)) + [-0.1, 0.1] + list(np.arange(0.5, 3.5, 0.5)),
        "PR": list(np.arange(-3, -0.0, 0.5)) + [-0.1, 0.1] + list(np.arange(0.5, 3.5, 0.5))
    }
}

cmaps = {
    "mean": {
        "TT": "bwr",
        "TT_min": "bwr",
        "TT_max": "bwr",
        "PR": "bwr",
    },
    "std": {
        "TT": "bwr",
        "PR": "bwr"
    }
}



var_name_to_cru_name = {
    "TT": "tmp", "PR": "pre"
}


var_name_to_file_prefix = {
    "TT": "dm", "PR": "pm"
}

var_name_to_mul_default = {
     "TT": 1, "PR": 1000 * 24 * 3600
}

area_thresh_km2 = 5000




def _plot_seasonal_deltas(seas_data:dict, data_label="", vname="", img_dir:Path=Path(), map:Basemap=None, lons=None, lats=None,
                          var_name_to_mul=var_name_to_mul_default):

    xx, yy = map(lons, lats)





    print("lons.shape = {}".format(lons.shape))
    for para_index, param in enumerate(["mean", "std"]):

        # plot the mean
        fig = plt.figure()

        color_levels = clevs[param][vname]

        norm = BoundaryNorm(color_levels, len(color_levels) - 1)
        cmap = cm.get_cmap(cmaps[param][vname], len(color_levels) - 1)


        gs = GridSpec(1, len(seas_data), wspace=0.01)
        for col, (season, data) in enumerate(seas_data.items()):
            ax = fig.add_subplot(gs[0, col])
            ax.set_title(season)



            to_plot = maskoceans(np.where(lons <= 180, lons, np.subtract(lons, 360.)), lats, data[para_index] * var_name_to_mul[vname])

            im = map.pcolormesh(xx, yy, to_plot, norm=norm, cmap=cmap, ax=ax)
            cb = map.colorbar(im, location="bottom", ticks=color_levels)
            map.drawcoastlines(linewidth=0.5)

            cb.ax.set_visible(col == 0)

            if col == 0:
                ax.set_ylabel("{}({})".format(vname, param))

            if col == 1:
                ax.set_xlabel(r"$\Delta$" + data_label, ha="left")



        img_path = "deltas_{}_{}_{}.png".format(data_label, vname, param)
        img_path = img_dir / img_path
        fig.savefig(str(img_path), bbox_inches="tight", dpi=400)
        plt.close(fig)



@profile
def main():
    # dask.set_options(pool=ThreadPool(20))
    img_folder = Path("nei_validation")

    if not img_folder.exists():
        img_folder.mkdir()

    pval_crit = 0.1



    # TT_min and TT_max mean daily min and maximum temperatures
    var_names = ["TT_max", "T_min", "PR"]

    var_name_to_rolling_window_days = {
        "TT_min": 5, "TT_max": 5,
        "PR": 29
    }


    var_name_to_percentiles = {
        "TT_min": [0.9, ],
        "TT_max": [0.1, ],
        "PR": [0.9, ]
    }

    # var_names = ["PR"]

    seasons = OrderedDict([
        ("DJF", MonthPeriod(12, 3)),
        ("MAM", MonthPeriod(3, 3)),
        ("JJA", MonthPeriod(6, 3)),
        ("SON", MonthPeriod(9, 3)),
    ])

    sim_paths = OrderedDict()

    start_year = 1980
    end_year = 1991

    sim_paths["WC_0.44deg_default"] = Path("/snow3/huziy/NEI/WC/NEI_WC0.44deg_default/Samples")
    sim_paths["WC_0.44deg_ctem+frsoil+dyngla"] = Path("/snow3/huziy/NEI/WC/debug_NEI_WC0.44deg_Crr1/Samples")
    # sim_paths["WC_0.11deg_ctem+frsoil+dyngla"] = Path("/HOME/huziy/skynet3_rech1/CRCM5_outputs/NEI/diags/NEI_WC0.11deg_Crr1/Diagnostics")

    # -- daymet monthly

    daymet_vname_to_path = {
        "prcp": "/snow3/huziy/Daymet_daily/daymet_v3_prcp_*_na.nc4",
        "tavg": "/snow3/huziy/Daymet_daily/daymet_v3_tavg_*_na.nc4",
        "tmin": "/snow3/huziy/Daymet_daily/daymet_v3_tmin_*_na.nc4",
        "tmax": "/snow3/huziy/Daymet_daily/daymet_v3_tmax_*_na.nc4",
    }

    vname_to_daymet_vname = {
        "PR": "prcp",
        "TT_min": "tmin",
        "TT_max": "tmax"
    }

    plot_utils.apply_plot_params(font_size=8)



    def __map_block(block_data, block_mask):
        if np.any(~block_mask):
            return block_data[~block_mask].mean()[None, None]
        else:
            return np.ma.masked_all((1,1))


    for vname in var_names:
        daymet_vname = vname_to_daymet_vname[vname]
        obs_manager = HighResDataManager(path=daymet_vname_to_path[daymet_vname], vname=daymet_vname, chunks=(5, 100, 100))


        print("Lazy computing")
        daily_perc, mask = obs_manager.get_daily_percenile_fields_lazy(obs_manager.data, start_year=start_year, end_year=end_year, percentile=var_name_to_percentiles[vname][0],
                                                                       rolling_mean_window_days=var_name_to_rolling_window_days[vname])


        print("Actual computing")
        agg_block = (50, 50)
        daily_perc_obs_mean = daily_perc.mean(axis=0).rechunk(agg_block)
        mask = mask.rechunk(agg_block)

        print("Computing block means")

        daily_perc_obs_mean = daily_perc_obs_mean.map_blocks(__map_block, mask).compute()

        plt.figure()
        im = plt.pcolormesh(daily_perc_obs_mean.T)
        plt.colorbar(im)
        plt.savefig("{}_rw{}_p{}_{}-{}.png".format(vname, var_name_to_rolling_window_days[vname], var_name_to_percentiles[vname][0], start_year, end_year), bbox_inches="tight")


        break
















if __name__ == '__main__':
    main()