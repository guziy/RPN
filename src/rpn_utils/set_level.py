import os
from rpn import level_kinds
from rpn.rpn import RPN

__author__ = 'huziy'

import numpy as np


def calclulate_swe(density, fraction, depth):
    return  density * fraction * depth



def main():
    varname = "SD"
    newLevel = 1
    oldLevel = 0
    levKind = level_kinds.ARBITRARY
    inFile = "anal_NorthAmerica_0.44deg_ERA40-Int1.5_B1_rmn13_and_Class_1979010100_dates_same"
    outFile = inFile + "3"
    folder = "/home/huziy/skynet3_rech1/init_cond_for_lake_infl_exp"

    inFile = os.path.join(folder, inFile)
    outFile = os.path.join(folder, outFile)

    print "input: {0}".format(inFile)
    print "output: {0}".format(outFile)

    inRObj = RPN(inFile)

    oldLevelIp = inRObj.get_ip1_from_level(oldLevel, level_kind=levKind)

    newLevelIp = inRObj.get_ip1_from_level(newLevel, level_kind =levKind)


    ipOfLevel1 = inRObj.get_ip1_from_level(1, level_kind = level_kinds.ARBITRARY)
    ipOfZeroMb = inRObj.get_ip1_from_level(0, level_kind = level_kinds.PRESSURE)

    print "ip1(old, new) = {0}, {1}".format(oldLevelIp, newLevelIp)

    outRObj = RPN(outFile, mode="w")


    inRObj.suppress_log_messages()
    outRObj.suppress_log_messages()

    data = []


    saved_vars = {}
    saved_info = None
    i = 0
    while data is not None:
        data = inRObj.get_next_record()
        if data is None:
            break
        info = inRObj.get_current_info

        ips =  map(lambda x: x.value, info["ip"])

        the_var_name = info["varname"].value.strip()

        if the_var_name == "I5":
            i += 1
            print "skipping swe from analysis, will be calculated from class data"
            continue





        conv_coef = 1
        #print "record ip: {0}".format(ips[0])
        if the_var_name == varname and ips[0] == oldLevelIp:
             #since CRCM expects snow depth in cm
            ips[0] = newLevelIp
            print ips
            pass

        if the_var_name == varname and (ips[0] == oldLevelIp or ips[0] == newLevelIp):
            conv_coef = 100.0


        if the_var_name == "DN":
            ips[0] = ipOfZeroMb

        npas = info["npas"].value
        deet = info["dt_seconds"].value
        dateo = info["dateo"].strftime(inRObj._dateo_format)
        nbits = info["nbits"].value
        data_type = info["data_type"].value
        if nbits > 0:
            nbits = -nbits
        print the_var_name, conv_coef

        if the_var_name == "SD" and ips[0] == ipOfLevel1: #get snowdepth only for soil
            saved_vars[the_var_name] = data
        elif the_var_name == "DN":
            saved_vars[the_var_name] = data
        elif the_var_name == "5P":
            saved_vars[the_var_name] = data
            saved_info = info.copy()  #this copy is not deep


        outRObj.write_2D_field(name = info["varname"].value,
            data = data * conv_coef, ip = ips,
            ig = map(lambda x: x.value, info["ig"]),
            npas = npas, deet=deet, label="IC, lake infl. exp.", dateo = dateo,
            grid_type=info["grid_type"].value, typ_var=info["var_type"].value,
            nbits = nbits, data_type = data_type
        )

        print   info["varname"].value, "->", nbits


        #calculate and write swe to the file
        if saved_vars is not None and len(saved_vars) == 3:
            print "calculating swe"
            ips = saved_info["ip"]
            print saved_info["nbits"].value

            print ips

            swe = calclulate_swe(saved_vars["DN"], saved_vars["5P"], saved_vars["SD"])
            outRObj.write_2D_field(name = "I5",
                data = swe, ip = map(lambda x: x.value, ips),
                ig = map(lambda x: x.value, saved_info["ig"]),
                npas = npas, deet=deet, label="IC, lake infl. exp.", dateo = dateo,
                grid_type=saved_info["grid_type"].value, typ_var=saved_info["var_type"].value,
                nbits = -saved_info["nbits"].value, data_type = saved_info["data_type"].value
            )

            saved_vars = None



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
  
