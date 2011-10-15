__author__="huziy"
__date__ ="$Aug 29, 2011 10:51:32 AM$"

from rpn import RPN
import matplotlib.pyplot as plt
import numpy as np

import application_properties
def test():
    path = 'data/pm1998010100-00-00_00000000p'
    rObj = RPN(path)
    data = rObj.get_first_record_for_name('STBM')

    print data[data < 0]
    print data.min(), data.max(), data.mean()
    data = np.ma.masked_where(data < 0, data)

    print np.ma.min(data)

    plt.pcolormesh(data.transpose())
    plt.colorbar()


    plt.show()


if __name__ == "__main__":
    application_properties.set_current_directory()
    test()
    print "Hello World"
