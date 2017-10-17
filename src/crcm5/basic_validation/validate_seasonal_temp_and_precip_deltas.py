




# Do a quick plot for temperature and precip biases for ~0.44 and 0.11 simulations

import matplotlib
from matplotlib.font_manager import FontProperties
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
from scipy import stats
from matplotlib import colors




var_name_to_level = {
    "TT": VerticalLevel(1, level_type=level_kinds.HYBRID),
    "PR": VerticalLevel(-1, level_type=level_kinds.ARBITRARY)
}




clevs = {
    "mean": {
        "TT": list(np.arange(-5, 0, 1)) + list(np.arange(1, 6, 1)),
        "PR": list(np.arange(-5, -0.5, 1)) + [-0.5, 0.5] + list(np.arange(1, 6, 1))
    },
    "std": {
        "TT": list(np.arange(-3, -0.0, 0.5)) + [-0.1, 0.1] + list(np.arange(0.5, 3.5, 0.5)),
        "PR": list(np.arange(-3, -0.0, 0.5)) + [-0.1, 0.1] + list(np.arange(0.5, 3.5, 0.5))
    }
}

cmaps = {
    "mean": {
        "TT": "bwr",
        "PR": "bwr"
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
                          var_name_to_mul=var_name_to_mul_default, seas_to_stats=None):

    xx, yy = map(lons, lats)





    print("lons.shape = {}".format(lons.shape))
    for para_index, param in enumerate(["mean", "std"]):

        # plot the mean
        fig = plt.figure()

        color_levels = clevs[param][vname]

        norm = BoundaryNorm(color_levels, len(color_levels) - 1)


        # Special colormap treatment for some variables
        if vname in ["TT"]:
            cmap = cm.get_cmap(cmaps[param][vname])

            cmap = colors.LinearSegmentedColormap.from_list("bwr_cut", cmap(np.arange(0.2, 0.9, 0.1)),
                                                            N=len(color_levels) - 1)

        else:
            cmap = cm.get_cmap(cmaps[param][vname], len(color_levels) - 1)


        gs = GridSpec(1, len(seas_data), wspace=0.01)
        for col, (season, data) in enumerate(seas_data.items()):
            ax = fig.add_subplot(gs[0, col])
            ax.set_title(season)



            to_plot = maskoceans(np.where(lons <= 180, lons, np.subtract(lons, 360.)), lats, data[para_index] * var_name_to_mul[vname])

            im = map.pcolormesh(xx, yy, to_plot, norm=norm, cmap=cmap, ax=ax)
            cb = map.colorbar(im, location="bottom", ticks=color_levels)
            map.drawcoastlines(linewidth=0.5)
            map.drawcountries(linewidth=0.5)
            map.drawstates(linewidth=0.5)


            cb.ax.set_visible(col == 0)

            if col == 0:
                ax.set_ylabel("{}({})".format(vname, param))

                cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=75)

            if col == 1:
                ax.set_xlabel(r"$\Delta$" + data_label, ha="left")


            # show some stats in each panel
            if seas_to_stats is not None:
                ax.annotate(seas_to_stats[season], (0.01, 0.01), xycoords="axes fraction", va="bottom", ha="left",
                            font_properties=FontProperties(size=8, weight="bold"))



        img_path = "deltas_{}_{}_{}.png".format(data_label, vname, param)
        img_path = img_dir / img_path
        fig.savefig(str(img_path), bbox_inches="tight", dpi=400)
        plt.close(fig)




def main():
    img_folder = Path("nei_validation")

    if not img_folder.exists():
        img_folder.mkdir()



    pval_crit = 0.1

    var_names = ["TT", "PR"]
    # var_names = ["PR"]

    seasons = OrderedDict([
        ("DJF", MonthPeriod(12, 3)),
        ("MAM", MonthPeriod(3, 3)),
        ("JJA", MonthPeriod(6, 3)),
        ("SON", MonthPeriod(9, 3)),
    ])

    sim_paths = OrderedDict()


    start_year = 1980
    end_year = 1998

    # sim_paths["WC_0.44deg_default"] = Path("/HOME/huziy/skynet3_rech1/CRCM5_outputs/NEI/diags/NEI_WC0.44deg_default/Diagnostics")
    sim_paths["WC_0.44deg_ctem+frsoil+dyngla"] = Path("/HOME/huziy/skynet3_rech1/CRCM5_outputs/NEI/diags/debug_NEI_WC0.44deg_Crr1/Diagnostics")
    sim_paths["WC_0.11deg_ctem+frsoil+dyngla"] = Path("/snow3/huziy/NEI/WC/NEI_WC0.11deg_Crr1/Diagnostics")



    # -- daymet monthly

    daymet_vname_to_path = {
        "prcp": "/HOME/data/Validation/Daymet/Monthly_means/NetCDF/daymet_v3_prcp_monttl_*_na.nc4",
        "tavg": "/HOME/huziy/skynet3_rech1/obs_data/daymet_tavg_monthly/daymet_v3_tavg_monavg_*_na_nc4classic.nc4"
    }


    vname_to_daymet_vname = {
        "PR": "prcp",
        "TT": "tavg"
    }



    plot_utils.apply_plot_params(font_size=14)


    basemap_for_obs = None
    # plot simulation data
    for sim_label, sim_path in sim_paths.items():


        manager_mod = DiagCrcmManager(data_dir=sim_path)



        for vname in var_names:

            daymet_vname = vname_to_daymet_vname[vname]
            manager_obs = HighResDataManager(path=daymet_vname_to_path[daymet_vname], vname=daymet_vname)

            seas_to_clim_mod = manager_mod.get_seasonal_means_with_ttest_stats(
                season_to_monthperiod=seasons,
                start_year=start_year, end_year=end_year, vname=vname,
                vertical_level=var_name_to_level[vname], data_file_prefix=var_name_to_file_prefix[vname]
            )

            seas_to_clim_obs = manager_obs.get_seasonal_means_with_ttest_stats_interpolated_to(
                manager_mod.lons, manager_mod.lats,
                season_to_monthperiod=seasons,
                start_year=start_year, end_year=end_year, convert_monthly_accumulators_to_daily=(vname == "PR")
            )


            season_to_diff = OrderedDict()
            season_to_summary_stats = OrderedDict()

            for season in seas_to_clim_mod:
                mod_mean, mod_std, mod_n = seas_to_clim_mod[season]
                obs_mean, obs_std, obs_n = seas_to_clim_obs[season]


                if vname == "PR":
                    # Convert model data to mm/day from M/s
                    mod_mean *= 1000 * 3600 * 24
                    mod_std *= 1000 * 3600 * 24


                tval, pval = ttest_ind_from_stats(mod_mean, mod_std, mod_n, obs_mean, obs_std, obs_n, equal_var=False)



                valid_points = ~(obs_mean.mask | np.isnan(obs_mean))
                mod_1d = mod_mean[valid_points]
                obs_1d = obs_mean[valid_points]

                rms = (((mod_1d - obs_1d) ** 2).sum() / len(mod_1d)) ** 0.5
                spat_corr, p_spat_corr = stats.pearsonr(mod_1d, obs_1d)

                season_to_summary_stats[season] = f"RMSE={rms:.1f}\nr={spat_corr:.2f}\nPVr={p_spat_corr:.2f}"



                season_to_diff[season] = []
                season_to_diff[season].append(np.ma.masked_where(pval >= pval_crit, mod_mean - obs_mean)) # mask not-significant biases
                season_to_diff[season].append(mod_std - obs_std)
                season_to_diff[season].append(-1)


            _plot_seasonal_deltas(
                seas_data=season_to_diff, data_label="{}_{}-{}".format(sim_label, start_year, end_year),
                img_dir=img_folder, map=manager_mod.get_basemap(resolution="i", area_thresh=area_thresh_km2),
                lons=manager_mod.lons, lats=manager_mod.lats, vname=vname,
                var_name_to_mul={"TT": 1, "PR": 1}, seas_to_stats=season_to_summary_stats
            )





if __name__ == '__main__':
    main()
