__author__ = 'huziy'

from .model_data import Crcm5ModelDataManager
import tables as tb


def test_export_to_hdf():
    input_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/for_tests"
    manager = Crcm5ModelDataManager(samples_folder_path=input_folder, all_files_in_samples_folder=True)

    out_file = "/skynet3_rech1/huziy/hdf_store/test1.hdf"

    manager.export_to_hdf(var_list = ["TT", ], file_path=out_file)

    h = tb.open_file(out_file)
    the_table = h.root.TT
    assert isinstance(the_table, tb.Table)

    print(the_table.cols.year[:20])
    print(the_table.cols.day[:20])
    print(the_table.cols.level[:10])

    print(the_table.description)
    h.close()

if __name__ == "__main__":
    test_export_to_hdf()

