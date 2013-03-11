from crcm5.model_data import Crcm5ModelDataManager
from rpn import level_kinds

__author__ = 'huziy'

import numpy as np

##Plot list of fields for one simulation

start_year = 1979
end_year = 1988

field_names = ["TT", "PR", "AU", "AV", "STFL"]
file_name_prefixes = ["dm", "pm", "pm", "pm", "pm"]
sim_name = "crcm5-r"
rpn_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-r_spinup"
nc_db_folder = "/home/huziy/skynet3_rech1/crcm_data_ncdb"

export_to_nc = True


def export_monthly_means_to_ncdb(data_manager, varname, level = -1, level_kind = -1):
    assert isinstance(data_manager, Crcm5ModelDataManager)

    data_manager.export_monthly_mean_fields( sim_name = sim_name, in_file_prefix = data_manager.file_name_prefix,
                                       start_year = start_year, end_year = end_year,
                                       varname = varname, nc_db_folder = nc_db_folder,
                                       level = level, level_kind = level_kind)
    pass

def plot_fields():
    pass


def main():
    dmManager = Crcm5ModelDataManager(samples_folder_path=rpn_folder, file_name_prefix="dm", all_files_in_samples_folder=True)
    pmManager = Crcm5ModelDataManager(samples_folder_path=rpn_folder, file_name_prefix="pm", all_files_in_samples_folder=True)


    #export monthly means to netcdf files if necessary
    if export_to_nc:
        for varname, prefix in zip( field_names, file_name_prefixes ):
            manager = None
            if prefix == "dm":
                manager = dmManager
            elif prefix == "pm":
                manager = pmManager

            level = -1
            level_kind = -1

            if varname == "TT":
                level = 1
                level_kind = level_kinds.HYBRID

            export_monthly_means_to_ncdb(manager, varname, level= level, level_kind= level_kind)

    #plot results
    basemap = pmManager.get_omerc_basemap()





    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  