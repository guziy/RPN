##Fix lake fraction field that was overwritten by the LM field (Mixed layer temperature in Flake)
import os
import tables as tb
from rpn.rpn import RPN
from crcm5 import infovar

SOURCE_PATH = "/skynet3_rech1/huziy/hdf_store/pm1979010100_00000000p"


def correct(path):
    #remove lake_fraction array and create a new one from the source (rpn data)
    #data
    print "Working on {0} ...".format(path)
    h = tb.open_file(path, "a")

    #read data from the rpn file
    r = RPN(SOURCE_PATH)
    lkfr = r.get_first_record_for_name("ML")
    r.close()

    h.get_node("/", infovar.HDF_LAKE_FRACTION_NAME)[:] = lkfr

    h.close()


def main():
    hdf_folder = "/skynet3_rech1/huziy/hdf_store"
    file_list = filter(lambda x: x.endswith(".hdf5"), os.listdir(hdf_folder))

    for fname in file_list:
        correct(os.path.join(hdf_folder, fname))


if __name__ == "__main__":
    main()


