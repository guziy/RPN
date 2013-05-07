from datetime import datetime
import os
from crcm5.model_data import Crcm5ModelDataManager

__author__ = 'huziy'

import numpy as np


def main():
    #params of the input data
    start_year = 1979
    end_year = 1988

    field_names = ["TT", "PR", "AU", "AH", "AV", "STFL","STFA", "TRAF", "TDRA", "I5", "GWDI"]
    file_name_prefixes = ["dm", "pm", "pm", "pm", "pm", "pm", "pm", "pm", "pm", "pm", "pm"]
    sim_name = "crcm5-hcd-rl-intfl"
    rpn_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_{0}_spinup2/Samples_all_in_one".format(sim_name)
    nc_db_folder = "/home/huziy/skynet3_rech1/crcm_data_ncdb"

    dmManager = Crcm5ModelDataManager(samples_folder_path=rpn_folder, file_name_prefix="dm", all_files_in_samples_folder=True)
    pmManager = Crcm5ModelDataManager(samples_folder_path=rpn_folder, file_name_prefix="pm", all_files_in_samples_folder=True)


    nc_path = os.path.join(nc_db_folder, sim_name)
    for field_name, fname_prefix in zip(field_names, file_name_prefixes):
        if fname_prefix == "pm":
            manager = pmManager
        elif fname_prefix == "dm":
            manager = dmManager
        else:
            raise Exception("Unknown file type...")
        manager.export_field_to_netcdf(start_year, end_year, var_name=field_name, nc_sim_folder= nc_path)
        pass


if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()

    main()

    print "Hello world"
