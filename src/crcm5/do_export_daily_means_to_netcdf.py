from datetime import datetime
from crcm5.model_data import Crcm5ModelDataManager

__author__ = 'huziy'

import numpy as np





def do_export(start_year, end_year, var_name = "STFA", dataManager = None):
    reference_date = datetime(start_year,1,1)

    assert isinstance(dataManager, Crcm5ModelDataManager)

    #TODO: implement


    pass



def main():
    start_year = 1979
    end_year = 1986

    field_names = ["TT", "PR", "AU", "AV", "STFL","STFA", "TRAF", "TDRA"]
    file_name_prefixes = ["dm", "pm", "pm", "pm", "pm", "pm", "pm", "pm"]
    sim_name = "crcm5-hcd-rl-intfl"
    rpn_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_{0}_spinup".format(sim_name)
    nc_db_folder = "/home/huziy/skynet3_rech1/crcm_data_ncdb"

    dmManager = Crcm5ModelDataManager(samples_folder_path=rpn_folder, file_name_prefix="dm", all_files_in_samples_folder=True)
    pmManager = Crcm5ModelDataManager(samples_folder_path=rpn_folder, file_name_prefix="pm", all_files_in_samples_folder=True)


    for field_name, fname_prefix in zip(field_names, file_name_prefixes):

        if fname_prefix == "pm":
            manager = pmManager
        elif fname_prefix == "dm":
            manager = dmManager
        else:
            raise Exception("Unknown file type...")

        do_export(start_year, end_year, var_name=field_name, dataManager=manager)
        pass


if __name__ == "__main__":
    main()
    print "Hello world"
  