import calendar
from collections import OrderedDict
from pathlib import Path

from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap
from scipy.stats import ttest_ind, ttest_ind_from_stats

from nemo.nemo_yearly_files_manager import NemoYearlyFilesManager
from util import plot_utils
import matplotlib.pyplot as plt
import numpy as np


img_folder = Path("nemo/climate_change")


# ----
vname_to_clevs_diff = {
    "soicecov": np.arange(-0.55, 0.6, 0.1),
    "sosstsst": np.arange(-7, 8, 2)
}



def main():
    current_start_year = 1981
    current_end_year = 2010

    future_start_year = 2070
    future_end_year = 2099


    LABEL_CURRENT = "Current"
    LABEL_FUTURE = "Future"

    pval_crit = 0.05



    label_to_period = {
        LABEL_CURRENT: (current_start_year, current_end_year),
        LABEL_FUTURE: (future_start_year, future_end_year)
    }

    season_to_months = OrderedDict()


    selected_months = [11, 12, 1]



    for i in selected_months:
        season_to_months[calendar.month_name[i]] = [i, ]




    print(season_to_months)

    nemo_icefrac_vname = "soicecov"
    nemo_sst_vname = "sosstsst"

    vname = nemo_sst_vname


    exp_label = "cc_canesm2_nemo_offline"

    nemo_managers_coupled_cc_slices_canesm2_rcp85 = OrderedDict([
        (LABEL_CURRENT, NemoYearlyFilesManager(folder="/HOME/huziy/skynet3_rech1/CRCM5_outputs/cc_canesm2_rcp85_gl/coupled-GL-current_CanESM2/CRCMNEMO_GL_CanESM2_RCP85", suffix="grid_T.nc")),
        (LABEL_FUTURE, NemoYearlyFilesManager(folder="/HOME/huziy/skynet3_rech1/CRCM5_outputs/cc_canesm2_rcp85_gl/coupled-GL-future_CanESM2/CRCMNEMO_GL_CanESM2_RCP85_future", suffix="grid_T.nc"))
    ])

    nemo_managers_offline_cc_canesm2_rcp85 = OrderedDict([
        (LABEL_CURRENT, NemoYearlyFilesManager(folder="/HOME/huziy/skynet3_rech1/NEMO_OFFICIAL/Simulations/cc_canesm2_nemo_offline_gathered_corrected_from_guillimin", suffix="grid_T.nc")),
        (LABEL_FUTURE, NemoYearlyFilesManager(folder="/HOME/huziy/skynet3_rech1/NEMO_OFFICIAL/Simulations/cc_canesm2_nemo_offline_gathered_corrected_from_guillimin", suffix="grid_T.nc"))
    ])


    # nemo_managers = OrderedDict([
    #     (LABEL_CURRENT, NemoYearlyFilesManager(folder="/BIG1/huziy/CRCM5_NEMO_coupled_sim_nemo_outputs/NEMO", suffix="grid_T.nc")),
    #     (LABEL_FUTURE, NemoYearlyFilesManager(folder="/BIG1/huziy/CRCM5_NEMO_coupled_sim_nemo_outputs/NEMO", suffix="grid_T.nc")),
    # ])


    nemo_managers = nemo_managers_offline_cc_canesm2_rcp85


    # calculate cc for LSWT and ice cover




    # Calculate seasonal mean projected changes
    label_to_data = OrderedDict()

    lons, lats = None, None

    for label, manager in nemo_managers.items():
        assert isinstance(manager, NemoYearlyFilesManager)

        start_year, end_year = label_to_period[label]
        label_to_data[label] = manager.get_seasonal_clim_fields_with_ttest_data(start_year=start_year, end_year=end_year,
                                                                                season_to_months=season_to_months, varname=vname)

        if lons is None:
            lons, lats = manager.lons, manager.lats






    # ----------- plot the plots
    #
    plot_utils.apply_plot_params(font_size=10, width_cm=8 * len(season_to_months), height_cm=5)

    map = Basemap(llcrnrlon=-93, llcrnrlat=41, urcrnrlon=-73,
                  urcrnrlat=48.5, projection='lcc', lat_1=33, lat_2=45,
                  lon_0=-90, resolution='i', area_thresh=10000)

    xx, yy = map(lons, lats)



    fig = plt.figure()
    gs = GridSpec(nrows=1, ncols=len(season_to_months), wspace=0.02, hspace=0.02)

    for col, season in enumerate(season_to_months):
        mean_c, std_c, nobs_c = label_to_data[LABEL_CURRENT][season]
        mean_f, std_f, nobs_f = label_to_data[LABEL_FUTURE][season]


        cc = mean_f - mean_c

        tval, pval = ttest_ind_from_stats(mean_c, std_c, nobs_c, mean_f, std_f, nobs_f, equal_var=False)

        cc = np.ma.masked_where(pval > pval_crit, cc)

        clevs = vname_to_clevs_diff[vname]
        cmap = cm.get_cmap("bwr", len(clevs) - 1)
        bn = BoundaryNorm(clevs, len(clevs) - 1)

        ax = fig.add_subplot(gs[0, col])
        im = map.pcolormesh(xx, yy, cc, cmap=cmap, norm=bn, ax=ax)
        cb = map.colorbar(im, location="bottom")

        cb.ax.set_visible(col == 0)

        map.drawcoastlines(linewidth=0.3)

        ax.set_frame_on(False)
        ax.set_title(season)

        if col == 0:
            ax.set_ylabel("F - C")



    # create the image folder if it does not exist yet
    if not img_folder.exists():
        img_folder.mkdir()


    fname = "{}_{}_{}vs{}.png".format(exp_label, vname, future_start_year, future_end_year, current_start_year, current_end_year)
    fig.savefig(str(img_folder / fname), bbox_inches="tight", dpi=300)



if __name__ == '__main__':
    main()
