from collections import OrderedDict

from cru.temperature import CRUDataManager
from util.seasons_info import MonthPeriod


import matplotlib.pyplot as plt
import numpy as np



def main():

    # seasons = OrderedDict([
    #     ("DJF", MonthPeriod(12, 3)),
    #     ("MAM", MonthPeriod(3, 3))
    # ])


    season_to_month_period = OrderedDict([
        ("DJF", MonthPeriod(12, 3))
    ])




    start_year = 1980
    end_year = 1982

    vname_to_path = {
        "pre": "/HOME/data/Validation/CRU_TS_3.1/Original_files_gzipped/cru_ts_3_10.1901.2009.pre.dat.nc",
        "tmp": "/HOME/data/Validation/CRU_TS_3.1/Original_files_gzipped/cru_ts_3_10.1901.2009.tmp.dat.nc"
    }



    for vname, path in vname_to_path.items():

        manager = CRUDataManager(path=path, var_name=vname)

        res = manager.get_seasonal_means_with_ttest_stats(
            season_to_monthperiod=season_to_month_period, start_year=start_year, end_year=end_year)


        plt.figure()
        mean_field, std_field, nobs = res["DJF"]

        im = plt.pcolormesh(mean_field.T)
        plt.colorbar(im)
        plt.show()





if __name__ == '__main__':
    main()