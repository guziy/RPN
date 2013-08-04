import time

__author__ = 'huziy'


from model_data import Crcm5ModelDataManager


def main():

    data_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-r_spinup"
    dm = Crcm5ModelDataManager(samples_folder_path = data_folder, all_files_in_samples_folder=True)
    var_names = ["STFL"]  # , "PR", "TT", "AV", "AH", "TRAF", "TDRA", "I5", "I0", "I1", "I2", "IMAV"]
    dm.export_to_hdf(var_list = var_names,
                     file_path="/home/huziy/skynet3_rech1/hdf_store/quebec_0.1_crcm5-r_spinup.hdf")
    pass

if __name__ == "__main__":

    import application_properties
    application_properties.set_current_directory()
    t0 = time.clock()
    main()
    print "Elapsed time {0} seconds".format(time.clock() - t0)
    print "Hello world"
