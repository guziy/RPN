from rpn.rpn import RPN

__author__ = 'huziy'

import numpy as np


def select_last_year(inPath, outPath = None, label = "last 6 year", npas_range = None):
    rObjIn = RPN(inPath)

    if outPath is None:
        outPath = inPath + "_last_year"
    rObjOut = RPN(outPath, mode="w")

    data = []
    i = 0
    while data is not None:
        data = rObjIn.get_next_record()
        if data is None:
            break
        info = rObjIn.get_current_info

        nbits = info["nbits"].value
        deet = info["dt_seconds"].value
        data_type = info["data_type"].value
        npas = info["npas"].value

        varname = info["varname"].value
        #
        if (npas not in npas_range) and varname.strip() not in [">>", "^^"]  : continue
        print(npas)

        dateo = info["dateo"]
        if nbits > 0:
            nbits = -nbits

        print("nbits = {0}, data_type = {1}".format(nbits, data_type))

        rObjOut.write_2D_field(name = varname,
            data = data, ip = [x.value for x in info["ip"]],
            ig = [x.value for x in info["ig"]],
            npas = npas, deet=deet, label=label, dateo = dateo,
            grid_type=info["grid_type"].value, typ_var=info["var_type"].value,
            nbits = nbits, data_type = data_type
        )
        i += 1


    #check that all fields were copied
    nRecsIn = rObjIn.get_number_of_records()
    #assert i == nRecsIn, "copied {0} records, but should be {1}".format(i, nRecsIn)

    rObjIn.close()
    rObjOut.close()




def main():
    #TODO: implement

    select_last_year("/home/huziy/skynet3_rech1/classOff_Andrey/mpi1/temp_3d",
        label="mpi1(last year)", npas_range=list( range(2400, 2400 - 12 * 6,-1) )
    )


    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print("Hello world")
  
