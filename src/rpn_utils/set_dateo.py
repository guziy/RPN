import os
from rpn.rpn import RPN

__author__ = 'huziy'


def main():
    dateo = "19580101000000"
    npas = 552240
    deet = 1200
    ip2 = 184080
    ip2old = 184086
    in_file = "anal_NorthAmerica_0.44deg_ERA40-Int1.5_B1_rmn13_and_Class_1979010100_2"
    out_file = "anal_NorthAmerica_0.44deg_ERA40-Int1.5_B1_rmn13_and_Class_1979010100_dates_same"

    folder = "/home/huziy/skynet3_rech1/init_cond_for_lake_infl_exp"

    rObjIn = RPN(os.path.join(folder, in_file))

    rObjOut = RPN(os.path.join(folder, out_file), mode="w")

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

        ips =  map(lambda x: x.value, info["ip"])
        if ips[1] == ip2old:
            ips[1] = ip2
            ips[2] = 0 #since ip3 is 0 there
        #convert soil temperature to Kelvins
        if info["varname"].value.strip() == "I0":
            data += 273.15

        rObjOut.write_2D_field(name = info["varname"].value,
            data = data, ip = ips,
            ig = map(lambda x: x.value, info["ig"]),
            npas = npas, deet=deet, label="IC, lake infl. exp.", dateo = dateo,
            grid_type=info["grid_type"].value, typ_var=info["var_type"].value,
            nbits = nbits, data_type = data_type
        )
        i += 1


    #check that all fields were copied
    nRecsIn = rObjIn.get_number_of_records()
    assert i == nRecsIn, "copied {0} records, but should be {1}".format(i, nRecsIn)

    rObjIn.close()
    rObjOut.close()


if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print "Hello world"
  
