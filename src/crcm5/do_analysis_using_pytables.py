import time
from crcm5.model_data import Crcm5ModelDataManager

__author__ = 'huziy'

import numpy as np


def calculate_daily_mean_fields():
    clim_fields = Crcm5ModelDataManager.hdf_get_daily_climatological_fields(hdf_db_path="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-r_spinup.hdf",
                                                              var_name="PR", level=None)





    pass


def main():
    manager = Crcm5ModelDataManager(
        samples_folder_path="/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-r_spinup",
        all_files_in_samples_folder=True)

    mean_field = manager.hdf_get_climatology_for_season(months=[6, 7, 8],
                                                        hdf_db_path="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-r_spinup.hdf",
                                                        var_name="TRAF", level=5)

    import matplotlib.pyplot as plt

    plt.contourf(mean_field)
    plt.show()

    pass


if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()
    t0 = time.clock()
    main()
    print "Elapsed time {0} seconds".format(time.clock() - t0)

    print "Hello world"