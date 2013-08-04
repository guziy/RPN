import time
from crcm5.model_data import Crcm5ModelDataManager

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt


def calculate_daily_mean_fields():
    time, clim_fields = Crcm5ModelDataManager.hdf_get_daily_climatological_fields(
        hdf_db_path="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-r_spinup.hdf",
        var_name="STFL", level=None, use_grouping=True)


    ts = np.mean(clim_fields, axis=1).mean(axis=1)
    plt.plot(time, ts)
    plt.savefig("test_hdf_ts_no_grouping.png")

    pass


def main():
    manager = Crcm5ModelDataManager(
        samples_folder_path="/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-r_spinup",
        all_files_in_samples_folder=True)

    hdf_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-r_spinup.hdf"
    mean_field = manager.hdf_get_climatology_for_season(months=[6, 7, 8],
                                                        hdf_db_path= hdf_path,
                                                        var_name="TRAF", level=5)

    plt.contourf(mean_field)
    plt.show()

    pass


if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()
    t0 = time.clock()
    #main()
    calculate_daily_mean_fields()
    print "Elapsed time {0} seconds".format(time.clock() - t0)

    print "Hello world"