__author__="huziy"
__date__ ="$Aug 19, 2011 3:12:27 PM$"

from rpn import RPN

import numpy as np
import application_properties

import matplotlib.pyplot as plt

def test():
    rObj = RPN('write.rpn', mode = 'w')

    nx = 20
    ny = 40

    data = np.zeros((nx, ny))
    for i in xrange(nx):
        for j in xrange(ny):
            data[i, j] = i ** 2 + j ** 2

    plt.figure()
    plt.title('before')
    plt.pcolormesh(data)
    plt.colorbar()


    rObj.write_2D_field('test', level = 1, data = data, grid_type = '')
    rObj.close()

    rObj = RPN('write.rpn')
    x = rObj.get_first_record_for_name('test')
    rObj.close()

    print x.shape


    plt.figure()
    plt.title('after')
    plt.pcolormesh(x)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    application_properties.set_current_directory()
    test()
    print "Hello World"
