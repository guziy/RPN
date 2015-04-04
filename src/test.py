from ctypes import c_float
import os.path

from ctypes import byref
from ctypes import POINTER
import application_properties
from ctypes import c_char_p
from ctypes import *
import os
import numpy as np

import matplotlib.pyplot as plt

application_properties.set_current_directory()



def get_first_record_for_name(varname, dll, file_unit):
    dll.fstinf_wrapper.restype = c_int
    dll.fstinf_wrapper.argtypes = [c_int, POINTER(c_int), POINTER(c_int), POINTER(c_int),
                                   c_int, c_char_p, c_int, c_int, c_int, c_char_p, c_char_p ]

    ni = c_int(0)
    nj = c_int(0)
    nk = c_int(0)
    datev = c_int(-1)
    etiket = c_char_p(' ' * 16)
    ip1 = c_int(-1)
    ip2 = c_int(-1)
    ip3 = c_int(-1)
    in_typvar = c_char_p(4 * ' ')

    in_nomvar = 8 * ' '
    in_nomvar = varname + in_nomvar
    in_nomvar = in_nomvar[:8]
    in_nomvar = c_char_p(in_nomvar)

    #int fstinf_wrapper(int iun, int *ni, int *nj, int *nk, int datev,char *in_etiket,
    #             int ip1, int ip2, int ip3, char *in_typvar, char *in_nomvar)

    key = dll.fstinf_wrapper(file_unit, byref(ni), byref(nj), byref(nk), datev, etiket,
                             ip1, ip2, ip3, in_typvar, in_nomvar
                            )
    data = np.zeros((nk.value, nj.value, ni.value,), dtype = np.float32)

    dll.fstluk_wrapper.restype = c_int
    dll.fstluk_wrapper.argtypes = [POINTER(c_float), c_int, POINTER(c_int), POINTER(c_int), POINTER(c_int)]

    print(dll.fstluk_wrapper(data.ctypes.data_as(POINTER(c_float)), key, ni, nj, nk))
    return data

def main():
    dll = CDLL('lib/rmnlib.so')

    dll.get_message.restype = c_char_p # c_char_p is a pointer to a string
    print(dll.get_message())
    print(dll.get_number())

    print(os.path.isfile('data/pm1957090100_00589248p'))
    rpn_file_path = c_char_p('data/pm1957090100_00589248p')
    file_unit = c_int(1)
    options = c_char_p('RND+R/O')
    dummy = c_int(0)
    print(dll.fnom_wrapper(file_unit, rpn_file_path, options, dummy))
    options.value = 'RND'
    nrecords = dll.fstouv_wrapper(file_unit, options)

    print("nrecords = {0}".format(nrecords))

    data = get_first_record_for_name('FV', dll, file_unit)
    print(np.min(data), np.max(data))

    print(data.shape)
    #data = data.reshape((ni.value, nj.value, nk.value))
    print(np.transpose(data[0,:,:]).shape) #like this data is in the order k, i, j
    plt.imshow(data[0,:,:], origin = 'lower') #for plotting in order to see i,j we supply j,i
    plt.colorbar()
    plt.show()
    #print data[129, 1, 0]


    

if __name__ == '__main__':
    main()