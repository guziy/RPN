import os
import time

__author__ = 'huziy'

from crcm5.model_data import Crcm5ModelDataManager
import tables as tb


def export_static_fields_to_hdf(hdf_file="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r_spinup2.hdf",
                                data_folder="/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-r_spinup2/all_files",
                                overwrite=False
):
    dm = Crcm5ModelDataManager(samples_folder_path=data_folder, all_files_in_samples_folder=True)
    dm.export_static_fields_to_hdf(file_path=hdf_file, overwrite=overwrite)
    pass


def multiply_table_column_by():
    path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-r_spinup.hdf"
    var_name = "AV"
    h = tb.open_file(path, mode="a")
    varTable = h.get_node("/", var_name)
    coef = 3 * 60 * 60  # output step
    expr = tb.Expr("c * m", uservars={"c": varTable.cols.field, "m": coef})
    column = varTable.cols.field
    expr.set_output(column)
    expr.eval()

    varTable.flush()
    h.close()


def correct_proj_table():
    data_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-rl-intfl_spinup3"
    from rpn.rpn import RPN

    hdf_file = "/home/huziy/skynet3_rech1/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup3.hdf"
    import hdf_table_schemes

    for fName in os.listdir(data_folder):
        if not fName.startswith("dm"):
            continue

        fPath = os.path.join(data_folder, fName)
        r = RPN(fPath)
        tt = r.get_first_record_for_name("TT")
        projData = r.get_proj_parameters_for_the_last_read_rec()
        Crcm5ModelDataManager.export_grid_properties_to_hdf(file_path=hdf_file, grid_params=projData,
                                                            table_scheme=hdf_table_schemes.projection_table_scheme)

        r.close()
        return


def main():
    from_guillimin_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/"

    # data_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-r_spinup"
    #interflow experiment (crcm5-hcd-rl-intfl)
    #data_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-rl-intfl_spinup3/all_files_in_one_folder"

    #(crcm5-hcd-rl) lakes and rivers interacting no interflow
    #data_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-rl_spinup"

    #lakes are full of land (crcm5-r)
    #data_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-r_spinup"
    #hdf_file_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-r_spinup.hdf"


    #lakes and rivers not interacting
    #data_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-r_spinup2/all_files"
    #hdf_file_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r_spinup2.hdf"

    #Lake residence time decreased 5 times
    #data_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-rl-intfl-kd5_spinup/all_files_in_one_folder"

    #latest interflow
    #data_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-rl-intfl_do_not_discard_small/all_files_in_one_dir"
    #hdf_file_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_do_not_discard_small.hdf"

    #interflow and ecoclimap
    #data_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-rl-intfl_spinup_ecoclimap/all_files_in_one_dir"

    #ecoclimap and era075
    #data_folder = "quebec_0.1_crcm5-hcd-rl-intfl_spinup_ecoclimap_era075/all_files_in_one_dir"
    #data_folder = os.path.join(from_guillimin_folder, data_folder)
    #hdf_file_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup_ecoclimap_era075.hdf"

    #soil anisotropy 10000
    #data_folder = os.path.join(from_guillimin_folder, "quebec_0.1_crcm5-hcd-rl-intfl_sani-10000/all_files_in_one_dir")
    #hdf_file_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_sani-10000.hdf"

    #soil anisotropy 10000 and do not care about bulk field capacity
    #data_folder = os.path.join(from_guillimin_folder,
    #                           "quebec_0.1_crcm5-hcd-rl-intfl_sani-10000_not_care_about_thfc/all_files_in_one_dir")

    #hdf_file_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_sani-10000_not_care_about_thfc.hdf"

    #On guillimin: crcm5-r
    #data_folder = "/home/huziy/current_project/Output/quebec_0.1_crcm5-r/all_files_in_one_dir"
    #hdf_file_path = "/home/huziy/current_project/PythonProjects/hdf_store/quebec_0.1_crcm5-r.hdf5"

    #On guillimin: crcm5-hcd-r
    #data_folder = "/sb/project/xgk-345-ab/huziy/Output/quebec_0.1_crcm5-hcd-r/all_files_in_one_dir"
    #hdf_file_path = "/sb/project/xgk-345-ab/huziy/PythonProjects/hdf_store/quebec_0.1_crcm5-hcd-r.hdf5"

    #On guillimin: crcm5-hcd-rl
    #data_folder = "/home/huziy/current_project/Output/quebec_0.1_crcm5-hcd-rl/all_files_in_one_dir"
    #hdf_file_path = "/home/huziy/current_project/PythonProjects/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5"

    #Interflow spinup
    #data_folder = "/home/huziy/current_project/Output/quebec_0.1_crcm5-hcd-rl-intfl_spinup_ITFS/all_files_in_one_dir"
    #hdf_file_path = "/home/huziy/current_project/PythonProjects/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup_ITFS.hdf5"

    #Interflow 
    ##data_folder = "/home/huziy/current_project/Output/quebec_0.1_crcm5-hcd-rl-intfl_ITFS/all_files_in_one_dir"
    ##hdf_file_path = "/home/huziy/current_project/PythonProjects/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5"

    #Interflow avoiding truncation 
    data_folder = "/gs/project/ugh-612-aa/huziy/Output/quebec_0.1_crcm5-hcd-rl-intfl_ITFS_avoid_truncation/all_in_one_dir"
    hdf_file_path = "/home/huziy/current_project/PythonProjects/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS_avoid_truncation1979-1989.hdf5"

    dm = Crcm5ModelDataManager(samples_folder_path=data_folder, all_files_in_samples_folder=True)
    var_names = ["STFA", "PR", "TT", "AV", "AH", "TRAF", "TDRA", "I5", "I0", "I1", "I2", "IMAV",
                 "AS", "INTF", "QQ", "UU", "VV", "WW", "GZ", "HR", "HU", "CLDP", "LC", "LD", "AL", "L1"]
    #var_names = [ "I0", "I1", "I2", "IMAV"]
    #var_names = ["AS", ]
    #var_names = ["QQ", ]
    #var_names = ["INTF", ]
    #var_names = ["AL",]
    #var_names = ["L1", "IMAV", "I5", "I1", "I2"]

    #var_names = ["TRAF", ]

    dm.export_to_hdf(var_list=var_names, file_path=hdf_file_path, mode="w")
    export_static_fields_to_hdf(
        hdf_file=hdf_file_path, data_folder=data_folder)
    pass


if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()
    t0 = time.clock()


    # data_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-rl-intfl_do_not_discard_small/all_files_in_one_dir"
    #hdf_file_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_do_not_discard_small.hdf"

    #export_static_fields_to_hdf(
    #    hdf_file= hdf_file_path, data_folder=data_folder, overwrite= True
    #)

    #
    #correct_proj_table()
    main()
    #multiply_table_column_by()
    print "Elapsed time {0} seconds".format(time.clock() - t0)
    print "Hello world"
