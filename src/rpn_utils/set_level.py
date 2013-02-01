import os
from rpn import level_kinds
from rpn.rpn import RPN

__author__ = 'huziy'

import numpy as np

def main():
    varname = "SD"
    newLevel = 1
    oldLevel = 0
    levKind = level_kinds.ARBITRARY
    inFile = "anal_NorthAmerica_0.44deg_ERA40-Int1.5_B1_rmn13_and_Class_1979010100_dates_same1"
    outFile = inFile[:-1] + "2"
    folder = "/home/huziy/skynet3_rech1/init_cond_for_lake_infl_exp"

    inFile = os.path.join(folder, inFile)
    outFile = os.path.join(folder, outFile)

    print "input: {0}".format(inFile)
    print "output: {0}".format(outFile)

    inRObj = RPN(inFile)

    oldLevelIp = inRObj.get_ip1_from_level(oldLevel, level_kind=levKind)

    newLevelIp = inRObj.get_ip1_from_level(newLevel, level_kind =levKind)


    print "ip1(old, new) = {0}, {1}".format(oldLevelIp, newLevelIp)

    outRObj = RPN(outFile, mode="w")


    inRObj.suppress_log_messages()
    outRObj.suppress_log_messages()

    data = []
    i = 0
    while data is not None:
        data = inRObj.get_next_record()
        if data is None:
            break
        info = inRObj.get_current_info()

        ips =  map(lambda x: x.value, info["ip"])

        the_var_name = info["varname"].value.strip()
        #print "record ip: {0}".format(ips[0])
        if the_var_name == varname and ips[0] == oldLevelIp:
            ips[0] = newLevelIp
            print ips
            pass
        npas = info["npas"].value
        deet = info["dt_seconds"].value
        dateo = info["dateo"].strftime(inRObj._dateo_format)
        outRObj.write_2D_field(name = info["varname"].value,
            data = data, ip = ips,
            ig = map(lambda x: x.value, info["ig"]),
            npas = npas, deet=deet, label="IC, lake infl. exp.", dateo = dateo,
            grid_type=info["grid_type"].value, typ_var=info["var_type"].value
        )
        i += 1


    #check that all fields were copied
    nRecsIn = inRObj.get_number_of_records()
    assert i == nRecsIn, "copied {0} records, but should be {1}".format(i, nRecsIn)




    inRObj.close()
    outRObj.close()





    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print "Hello world"
  