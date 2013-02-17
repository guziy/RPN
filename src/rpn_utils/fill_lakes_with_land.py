from scipy.spatial.kdtree import KDTree
from rpn import level_kinds
from rpn.rpn import RPN
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np


import matplotlib.pyplot as plt



def interpolate(inField, inLons2D, inLats2D, kdtree):

    #TODO

    pass




def main():

    glob_fields_to_interp = ["SAND", "CLAY", "DPTH"]

    in_path = "/home/huziy/skynet3_rech1/from_guillimin/geophys_Quebec_0.1deg_260x260_with_dd_v6"
    out_path = in_path + "_no_lakes"


    #read in lake fraction field


    inRpnObj = RPN(in_path)
    lkfr = inRpnObj.get_first_record_for_name_and_level(varname="VF", level=3, level_kind=level_kinds.ARBITRARY)
    #lkfr = np.ma.masked_where(lkfr < 0.9, lkfr)



    lonsTarget, latsTarget = inRpnObj.get_longitudes_and_latitudes_for_the_last_read_rec()



    xt, yt, zt = lat_lon.lon_lat_to_cartesian(lonsTarget.flatten(), latsTarget.flatten())

    selLkfrFlat = lkfr[lkfr < 1]






    sand_glob = inRpnObj.get_2D_field_on_all_levels(name="SAND")

    lonsG, latsG = inRpnObj.get_longitudes_and_latitudes_for_the_last_read_rec()
    print lonsG.shape
    print lonsG.min(), lonsG.max(), lonsG.mean()
    sandg1 = sand_glob[1]
    print "min: {0}; max: {1}; mean: {2}".format(sandg1.min(), sandg1.max(), sandg1.mean())
    info = inRpnObj.get_current_info()
    print info[RPN.GRID_TYPE].value




    if True:
        inRpnObj.close()
        return





    plt.pcolormesh(lkfr.transpose())
    plt.show()


    outRpnObj = RPN(out_path, mode="w")

    data = []
    inRpnObj.reset_current_info()
    i = 0
    while data is not None:
        data = inRpnObj.get_next_record()
        if data is None:
            break
        info = inRpnObj.get_current_info()

        nbits = info["nbits"].value
        data_type = info["data_type"].value

        if nbits > 0:
            nbits = -nbits

        print "nbits = {0}, data_type = {1}".format(nbits, data_type)

        ips =  map(lambda x: x.value, info["ip"])
        dateo = info["dateo_rpn_format"].value


        if info["varname"].value.strip().upper() in glob_fields_to_interp:
            lonsG, latsG = inRpnObj.get_longitudes_and_latitudes_for_the_last_read_rec()
            x0, y0, z0 = lat_lon.lon_lat_to_cartesian(lonsG.flatten(), latsG.flatten())
            kdt = KDTree(zip(x0, y0, z0))
            pass





        outRpnObj.write_2D_field(name = info["varname"].value,
            data = data, ip = ips,
            ig = map(lambda x: x.value, info["ig"]),
            npas = info["npas"].value, deet=info["dt_seconds"].value,
            label="IC,lake infl.exp.(nolakes)", dateo = dateo,
            grid_type=info["grid_type"].value, typ_var=info["var_type"].value,
            nbits = nbits, data_type = data_type
        )
        i += 1


    #check that all fields were copied
    nRecsIn = inRpnObj.get_number_of_records()
    assert i == nRecsIn, "copied {0} records, but should be {1}".format(i, nRecsIn)

    inRpnObj.close()
    outRpnObj.close()



    # find points with lkfr = 1

    #read in all fields

    #for the points with lkfr = 1,


    #TODO: implement
    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print "Hello world"
  