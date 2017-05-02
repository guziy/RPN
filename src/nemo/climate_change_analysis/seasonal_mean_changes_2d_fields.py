import calendar
from collections import OrderedDict

from mpl_toolkits.basemap import Basemap

from nemo.nemo_yearly_files_manager import NemoYearlyFilesManager


def main():

    current_start_year = 1980
    current_end_year = 2010

    future_start_year = 2070
    future_end_year = 2100


    LABEL_CURRENT = "NEMO_offline_current"
    LABEL_FUTURE = "NEMO_offline_future"



    label_to_period = {
        LABEL_CURRENT: (current_start_year, current_end_year),
        LABEL_FUTURE: (future_start_year, future_end_year)
    }

    season_to_months = OrderedDict()


    selected_months = [1, 2, 3, 4]

    for i, m in enumerate(calendar.month_name[1:], 1):
        if i in selected_months:
            season_to_months[m] = [i, ]

    print(season_to_months)

    nemo_icefrac_vname = "soicecov"
    nemo_managers = OrderedDict([
        (LABEL_CURRENT, NemoYearlyFilesManager(folder="/BIG1/huziy/CRCM5_NEMO_coupled_sim_nemo_outputs/NEMO", suffix="grid_T.nc")),
        (LABEL_FUTURE, NemoYearlyFilesManager(folder="/BIG1/huziy/CRCM5_NEMO_coupled_sim_nemo_outputs/NEMO", suffix="grid_T.nc")),
    ])




    # calculate and plot
    map = Basemap(llcrnrlon=-93, llcrnrlat=41, urcrnrlon=-73,
                  urcrnrlat=48.5, projection='lcc', lat_1=33, lat_2=45,
                  lon_0=-90, resolution='i', area_thresh=10000)




    




    pass





if __name__ == '__main__':
    main()
