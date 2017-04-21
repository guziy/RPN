import calendar
from collections import OrderedDict

from mpl_toolkits.basemap import Basemap

from application_properties import main_decorator
from nemo.nemo_yearly_files_manager import NemoYearlyFilesManager
from nemo.nic_cis_ice_cover_manager import CisNicIceManager

import numpy as np
import calendar
import matplotlib.pyplot as plt


def validate_2d_maps(nemo_managers, obs_manager:CisNicIceManager, start_year=-np.Inf, end_year=np.Inf,
                     season_to_months=None, nemo_icecover_name="soicecov", nemo_field_level_index=0, basemap=None):
    # read obs data
    obs_data = obs_manager.get_seasonal_mean_climatologies(start_year=start_year, end_year=end_year,
                                                       season_to_months=season_to_months)



    label_to_nemo_data = OrderedDict()
    for label, nemo_manager in nemo_managers.items():
        assert isinstance(nemo_manager, NemoYearlyFilesManager)

        label_to_nemo_data[label] = nemo_manager.get_seasonal_clim_field_for_dates(start_year=start_year, end_year=end_year, varname=nemo_icecover_name,
                                                       season_to_selected_dates=obs_data[-1], level_index=nemo_field_level_index)



    # figure
    #   cols = seasons
    #   rows = (obs, mod1 - obs, mod2 - obs, ..., modn - obs)


    plt.figure()


    for season, months in season_to_months.items():


        pass






def validate_areaavg_annual_max(nemo_configs, obs_manager:CisNicIceManager, start_year=-np.Inf, end_year=np.Inf,
                                season_start_month=10, season_end_month=4, mask_shape_file=""):
    """
    the year of the start of the season corresonds to the aggregated value for the season, i.e. if season starts in Oct 2009 and ends in March 2010, then the maximum value 
    for the season would correspond to 2009
    :param nemo_configs: 
    :param obs_manager: 
    :param start_year: 
    :param end_year: 
    """
    pass


@main_decorator
def main():

    obs_data_path = "/RESCUE/skynet3_rech1/huziy/obs_data_for_HLES/interploated_to_the_same_grid/GL_0.1_452x260/cis_nic_glerl_interpolated_lc.nc"

    season_to_months = {calendar.month_name[i]: [i, ] for i in range(1, 13)}

    obs_manager = CisNicIceManager()


    start_year = 1980
    end_year = 2010


    season_to_months = OrderedDict()


    for i, m in enumerate(calendar.month_name[1:], 1):
        season_to_months[m] = [i, ]

    print(season_to_months)

    nemo_icefrac_vname = "soicecov"
    nemo_managers = OrderedDict([
        ("CRCM5_NEMO", NemoYearlyFilesManager(folder="/BIG1/huziy/CRCM5_NEMO_coupled_sim_nemo_outputs/NEMO", suffix="grid_T.nc")),
    ])




    # calculate and plot
    map = Basemap(llcrnrlon=-91, llcrnrlat=40, urcrnrlon=-75,
                  urcrnrlat=50, projection='lcc', lat_1=33, lat_2=45,
                  lon_0=-90, resolution='i', area_thresh=10000)


    map.drawmeridians(np.arange(-180, 180, 5), labels=[1, 0, 0, 1])
    map.drawparallels(np.arange(-90, 90, 5), labels=[1, 0, 0, 1])

    map.drawcoastlines(linewidth=0.5)
    map.drawcountries()
    map.drawstates()

    plt.show()

if __name__ == '__main__':
    main()