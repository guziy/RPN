import os
from rpn.rpn import RPN

__author__ = 'huziy'

import numpy as np

from ctypes import *

def main():
    #path = "/b2_fs2/huziy/OMSC26_Can_long_new_v01/pm1958010100_02275344p"

    path = "/b2_fs2/huziy/OMSC26_Can_long_new_v01/pm1958010100_00008640p"
    r = RPN(path)


    res_date = c_int()
    res_time = c_int()
    mode = c_int(-3)

    #test1
    dateo = 10158030
    r._dll.newdate_wrapper(byref(c_int(dateo)), byref(res_date), byref(res_time), byref(mode))
    s_date = "{0:08d}{1:08d}".format(res_date.value, res_time.value)
    print "stamp: {0:09d}, result: {1}".format( dateo, s_date)


    dateo = 488069900
    r._dll.newdate_wrapper(byref(c_int(dateo)), byref(res_date), byref(res_time), byref(mode))
    s_date = "{0:08d}{1:08d}".format(res_date.value, res_time.value)
    print "stamp: {0:09d}, result: {1}".format( dateo, s_date)

    dateo = 1069261100
    r._dll.newdate_wrapper(byref(c_int(dateo)), byref(res_date), byref(res_time), byref(mode))
    s_date = "{0:08d}{1:08d}".format(res_date.value, res_time.value)
    print "stamp: {0:09d}, result: {1}".format( dateo, s_date)

    dateo = 632053700
    r._dll.newdate_wrapper(byref(c_int(dateo)), byref(res_date), byref(res_time), byref(mode))
    s_date = "{0:08d}{1:08d}".format(res_date.value, res_time.value)
    print "stamp: {0:09d}, result: {1}".format( dateo, s_date)



    ts = r.get_all_time_records_for_name("TRAF")

    times = list( sorted( ts.keys() ) )
    print times[:20]
    print times[0], times[-1]
    r.close()




    folderPath = "/b2_fs2/huziy/OMSC26_ERA40I_long_new_v02/"
    for fName in os.listdir(folderPath):
        if not fName.startswith("pm"): continue

        fPath = os.path.join(folderPath, fName)

        r = RPN(fPath)
        r.suppress_log_messages()

        data = r.get_all_time_records_for_name(varname="TDRA")
        r.close()

        print fName
        print sorted(data.keys())[:5]
        print 25 * "*"
        raw_input("press any key")




if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print "Hello world"
  