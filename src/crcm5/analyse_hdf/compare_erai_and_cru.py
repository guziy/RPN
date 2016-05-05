# compare ERA-Interim 1.5 degree and CRU precipitation
# ERA-I, CRU, ERAI minus CRU
import os
from collections import OrderedDict

from matplotlib import gridspec
from mpl_toolkits.basemap import Basemap, maskoceans
from rpn import level_kinds
from rpn.rpn_multi import MultiRPN

from application_properties import main_decorator
from crcm5.analyse_hdf.compare_driving_field_with_crcm_hdf_field import get_files_for_season
from cru.temperature import CRUDataManager
import matplotlib.pyplot as plt
import numpy as np

from util import plot_utils


img_folder = "cc-paper-comments"

@main_decorator
def main():
    erainterim_075_folder = "/HOME/data/Validation/ERA-Interim_0.75/Offline_driving_data/3h_Forecast"

    vname = "PR"
    start_year = 1980
    end_year = 2010


    season_key = "summer"
    season_labels = {season_key: "Summer"}
    season_to_months = OrderedDict([
        (season_key, [6, 7, 8])
    ])


    # Validate temperature and precip
    model_vars = ["TT", "PR"]
    obs_vars = ["tmp", "pre"]

    obs_paths = [
        "/HOME/data/Validation/CRU_TS_3.1/Original_files_gzipped/cru_ts_3_10.1901.2009.tmp.dat.nc",
        "/HOME/data/Validation/CRU_TS_3.1/Original_files_gzipped/cru_ts_3_10.1901.2009.pre.dat.nc"
    ]

    model_var_to_obs_var = dict(zip(model_vars, obs_vars))
    model_var_to_obs_path = dict(zip(model_vars, obs_paths))

    obs_path = model_var_to_obs_path[vname]

    cru = CRUDataManager(var_name=model_var_to_obs_var[vname], path=obs_path)

    # Calculate seasonal means for CRU
    seasonal_clim_fields_cru = cru.get_seasonal_means(season_name_to_months=season_to_months,
                                                      start_year=start_year,
                                                      end_year=end_year)



    # Calculate seasonal mean for erai
    flist = get_files_for_season(erainterim_075_folder, start_year=start_year, end_year=end_year, months=season_to_months[season_key])

    rpf = MultiRPN(flist)
    date_to_field_erai075 = rpf.get_all_time_records_for_name_and_level(varname=vname, level=-1)

    # Convert to mm/day
    era075 = np.mean([field for field in date_to_field_erai075.values()], axis=0) * 24 * 3600 * 1000
    lons_era, lats_era = rpf.get_longitudes_and_latitudes_of_the_last_read_rec()



    seasonal_clim_fields_cru_interp = OrderedDict()

    # Calculate biases
    for season, cru_field in seasonal_clim_fields_cru.items():
        seasonal_clim_fields_cru_interp[season] = cru.interpolate_data_to(cru_field,
                                                                          lons2d=lons_era,
                                                                          lats2d=lats_era,
                                                                          nneighbours=1)




    # Do the plotting ------------------------------------------------------------------------------
    plot_utils.apply_plot_params()
    fig = plt.figure()


    b = Basemap()
    gs = gridspec.GridSpec(nrows=3, ncols=1)


    ax = fig.add_subplot(gs[0, 0])
    xx, yy = b(cru.lons2d, cru.lats2d)
    cs = b.contourf(xx, yy, seasonal_clim_fields_cru[season_key], 20)
    b.drawcoastlines(ax=ax)
    ax.set_title("CRU")
    plt.colorbar(cs, ax=ax)


    ax = fig.add_subplot(gs[1, 0])



    lons_era[lons_era > 180] -= 360
    lons_era, era075 = b.shiftdata(lons_era, datain=era075, lon_0=0)
    xx, yy = b(lons_era, lats_era)

    # mask oceans in the era plot as well
    era075 = maskoceans(lons_era, lats_era, era075)

    cs = b.contourf(xx, yy, era075, levels=cs.levels, norm=cs.norm, cmap=cs.cmap, ax=ax)
    b.drawcoastlines(ax=ax)
    ax.set_title("ERA-Interim 0.75")
    plt.colorbar(cs, ax=ax)


    # differences
    ax = fig.add_subplot(gs[2, 0])
    diff = era075 - seasonal_clim_fields_cru_interp[season_key]
    delta = np.percentile(np.abs(diff)[~diff.mask], 90)
    clevs = np.linspace(-delta, delta, 20)

    cs = b.contourf(xx, yy, diff, levels=clevs, cmap="RdBu_r", extend="both")
    b.drawcoastlines(ax=ax)
    ax.set_title("ERA-Interim 0.75 - CRU")
    plt.colorbar(cs, ax=ax)

    plt.show()

    fig.savefig(os.path.join(img_folder, "erai0.75_vs_cru_precip.png"), bbox_inches="tight")





if __name__ == '__main__':
    main()