from ctypes import c_int
import os

__author__ = 'huziy'
from rpn.rpn import RPN


def main():
    folder = "/home/huziy/skynet3_exec1/modify_igs_in_rpn_file"
    in_file = "ANAL_NorthAmerica_0.44deg_MPIRCP45_B1_100_2070120100"
    out_file = in_file + "_ig_changed"
    rObjIn = RPN(os.path.join(folder, in_file))

    rObjOut = RPN(os.path.join(folder, out_file), mode="w")

    ig_to_change = [1375, 0, 56480, 56480]
    new_ig = [499, 1064, 0, 0]

    data = []
    i = 0
    while data is not None:
        data = rObjIn.get_next_record()
        if data is None:
            break
        info = rObjIn.get_current_info

        nbits = info["nbits"].value
        data_type = info["data_type"].value

        if nbits > 0:
            nbits = -nbits

        print "nbits = {0}, data_type = {1}".format(nbits, data_type)

        ips = map(lambda x: x.value, info["ip"])

        npas = info["npas"].value
        deet = info["dt_seconds"].value
        dateo = info["dateo"]

        igold = [int(ig.value) for ig in info["ig"]]

        if igold == ig_to_change:
            info["ig"] = [c_int(ig) for ig in new_ig]

        rObjOut.write_2D_field(name=info["varname"].value,
                               data=data, ip=ips,
                               ig=map(lambda x: x.value, info["ig"]),
                               npas=npas, deet=deet, label="", dateo=dateo,
                               grid_type=info["grid_type"].value, typ_var=info["var_type"].value,
                               nbits=nbits, data_type=data_type
        )
        i += 1


    #check that all fields were copied
    nRecsIn = rObjIn.get_number_of_records()
    assert i == nRecsIn, "copied {0} records, but should be {1}".format(i, nRecsIn)

    rObjIn.close()
    rObjOut.close()


if __name__ == "__main__":
    main()