# Do a quick plot for temperature and precip biases for ~0.44 and 0.11 simulations

import matplotlib
# matplotlib.use("Agg")


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
        "TT": np.arange(-40, 42, 2),
        "PR": np.arange(0, 10.5, 0.5)
    },
    "std": {
        "TT": np.arange(0, 4.2, 0.2),
        "PR": np.arange(0, 1.1, 0.1)
    }
}

cmaps = {
    "mean": {
        "TT": "bwr",
        "PR": "jet_r"
    },
    "std": {
        "TT": "jet",
        "PR": "jet"
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



def _plot_seasonal_data(seas_data: dict, data_label="", vname="", img_dir: Path = Path(),
                        map: Basemap = None, lons=None, lats=None,
                        var_name_to_mul=var_name_to_mul_default):


    xx, yy = map(np.where(lons < 0, lons + 360, lons), lats)

    print("lons.shape = {}".format(lons.shape))
    print("vname_to_mult: {}".format(var_name_to_mul))


    not_plot_reg = ((xx < map.xmin) | (xx > map.xmax) | (yy < map.ymin) | (yy > map.ymax))

    for para_index, param in enumerate(["mean", "std"]):

        # plot the mean
        fig = plt.figure()

        color_levels = clevs[param][vname]

        norm = BoundaryNorm(color_levels, len(color_levels) - 1)
        cmap = cm.get_cmap(cmaps[param][vname], len(color_levels) - 1)


        gs = GridSpec(1, len(seas_data))
        for col, (season, data) in enumerate(seas_data.items()):
            ax = fig.add_subplot(gs[0, col])
            ax.set_title(season)


            to_plot = np.ma.masked_where(not_plot_reg, data[para_index]) * var_name_to_mul[vname]

            im = map.pcolormesh(xx, yy, to_plot, norm=norm, cmap=cmap, ax=ax)
            cb = map.colorbar(im, location="bottom")
            map.drawcoastlines(linewidth=0.5)
            map.drawcountries(linewidth=0.5)
            map.drawstates(linewidth=0.5)

            cb.ax.set_visible(col == 0)

            if col == 0:
                ax.set_ylabel("{}({})".format(vname, param))
                cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=45)

            if col == 1:
                ax.set_xlabel(data_label, ha="left")

        img_path = "{}_{}_{}.png".format(data_label, vname, param)
        img_path = img_dir / img_path
        fig.savefig(str(img_path), bbox_inches="tight", dpi=400)
        plt.close(fig)


def main():
    img_folder = Path("nei_validation")
    img_folder.mkdir(parents=True, exist_ok=True)

    var_names = ["TT", "PR"]

    seasons = OrderedDict([
        ("DJF", MonthPeriod(12, 3)),
        ("MAM", MonthPeriod(3, 3)),
        ("JJA", MonthPeriod(6, 3)),
        ("SON", MonthPeriod(9, 3)),
    ])

    sim_paths = OrderedDict()

    start_year = 1980
    end_year = 1998

    sim_paths["WC_0.44deg_default"] = Path \
        ("/HOME/huziy/skynet3_rech1/CRCM5_outputs/NEI/diags/NEI_WC0.44deg_default/Diagnostics")
    sim_paths["WC_0.44deg_ctem+frsoil+dyngla"] = Path \
        ("/HOME/huziy/skynet3_rech1/CRCM5_outputs/NEI/diags/debug_NEI_WC0.44deg_Crr1/Diagnostics")
    sim_paths["WC_0.11deg_ctem+frsoil+dyngla"] = Path("/snow3/huziy/NEI/WC/NEI_WC0.11deg_Crr1/Diagnostics")

    cru_vname_to_path = {
        "pre": "/HOME/data/Validation/CRU_TS_3.1/Original_files_gzipped/cru_ts_3_10.1901.2009.pre.dat.nc",
        "tmp": "/HOME/data/Validation/CRU_TS_3.1/Original_files_gzipped/cru_ts_3_10.1901.2009.tmp.dat.nc"
    }

    plot_cru_data = True
    plot_model_data = True
    plot_naobs_data = False
    plot_daymet_data = True

    plot_utils.apply_plot_params(font_size=14)

    basemap_for_obs = None
    # plot simulation data
    for sim_label, sim_path in sim_paths.items():
        manager = DiagCrcmManager(data_dir=sim_path)

        # get the basemap to be reused for plotting observation data
        if basemap_for_obs is None:
            basemap_for_obs = manager.get_basemap(resolution="i", area_thresh=area_thresh_km2)

        if not plot_model_data:
            break

        for vname in var_names:
            seas_to_clim = manager.get_seasonal_means_with_ttest_stats(
                season_to_monthperiod=seasons,
                start_year=start_year, end_year=end_year, vname=vname,
                vertical_level=var_name_to_level[vname], data_file_prefix=var_name_to_file_prefix[vname]
            )

            _plot_seasonal_data(
                seas_data=seas_to_clim, data_label="{}_{}-{}".format(sim_label, start_year, end_year),
                img_dir=img_folder, map=manager.get_basemap(resolution="i", area_thresh=area_thresh_km2),
                lons=manager.lons, lats=manager.lats, vname=vname
            )

    assert basemap_for_obs is not None

    # plot obs data
    # -- CRU
    for vname in var_names:

        if not plot_cru_data:
            break

        cru_vname = var_name_to_cru_name[vname]

        manager = CRUDataManager(path=cru_vname_to_path[cru_vname], var_name=cru_vname)

        seas_to_clim = manager.get_seasonal_means_with_ttest_stats(
            season_to_monthperiod=seasons,
            start_year=start_year, end_year=end_year
        )

        manager.close()


        _plot_seasonal_data(
            seas_data=seas_to_clim, data_label="{}_{}-{}".format("CRU", start_year, end_year),
            img_dir=img_folder, map=basemap_for_obs,
            lons=manager.lons2d, lats=manager.lats2d, vname=vname,
            var_name_to_mul={"TT": 1, "PR": 1}
        )

    # -- NAOBS
    naobs_vname_to_path = {
        "TT": "/HOME/huziy/skynet3_rech1/obs_data/anuspl_uw_0.11_wc_domain/anusplin+_interpolated_tt_pr.nc",
        "PR": "/HOME/huziy/skynet3_rech1/obs_data/anuspl_uw_0.11_wc_domain/anusplin+_interpolated_tt_pr.nc"
    }

    for vname in var_names:

        if not plot_naobs_data:
            break

        manager = CRUDataManager(path=naobs_vname_to_path[vname], var_name=vname)

        seas_to_clim = manager.get_seasonal_means_with_ttest_stats(
            season_to_monthperiod=seasons,
            start_year=start_year, end_year=end_year
        )


        # mask no data points
        for s, data in seas_to_clim.items():
            for i in [0, 1]:
                data[i] = np.ma.masked_where(manager.lats2d > 60, data[i])
                data[i] = np.ma.masked_where(manager.lons2d < -150, data[i])
                data[i] = maskoceans(manager.lons2d, manager.lats2d, datain=data[i])


        _plot_seasonal_data(
            seas_data=seas_to_clim, data_label="{}_{}-{}".format("NAOBS", start_year, end_year),
            img_dir=img_folder, map=basemap_for_obs,
            lons=manager.lons2d, lats=manager.lats2d, vname=vname
        )

        manager.close()

    # -- daymet monthly

    daymet_vname_to_path = {
        "prcp": "/HOME/data/Validation/Daymet/Monthly_means/NetCDF/daymet_v3_prcp_monttl_*_na.nc4",
        "tavg": "/HOME/huziy/skynet3_rech1/obs_data/daymet_tavg_monthly/daymet_v3_tavg_monavg_*_na_nc4classic.nc4"
    }

    vname_to_daymet_vname = {
        "PR": "prcp",
        "TT": "tavg"
    }

    for vname in var_names:

        if not plot_daymet_data:
            break

        daymet_vname = vname_to_daymet_vname[vname]

        manager = HighResDataManager(path=daymet_vname_to_path[daymet_vname], vname=daymet_vname)

        seas_to_clim = manager.get_seasonal_means_with_ttest_stats_dask(
            season_to_monthperiod=seasons,
            start_year=start_year, end_year=end_year, convert_monthly_accumulators_to_daily=(vname == "PR")
        )

        _plot_seasonal_data(
            seas_data=seas_to_clim, data_label="{}_{}-{}".format("DAYMET", start_year, end_year),
            img_dir=img_folder, map=basemap_for_obs,
            lons=manager.lons, lats=manager.lats, vname=vname,
            var_name_to_mul={"PR": 1, "TT": 1}
        )

        manager.close()


if __name__ == '__main__':
    main()
