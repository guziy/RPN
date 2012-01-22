__author__="huziy"
__date__ ="$19 mai 2010 12:12:24$"

from util.lat_lon_holder import LatLonHolder
import time
import application_properties

application_properties.set_current_directory()



def amno_convert_list(the_list): 
    holder = LatLonHolder()
    nx = holder.Nx
    print 'nx = ', nx
    result = []
    for pair in the_list:
        composite_index = (pair[1] - 1) * nx + (pair[0] - 1)
        new_pair = [ holder.longitudes[composite_index],
                     holder.latitudes[composite_index] ]
        result.append(new_pair)
    return result


def amno_from_index_space_to_lat_lon(ix, iy):
    '''
    ix in [1, IXMAX], iy in [1, IYMAX]
    '''
    model = LatLonHolder()
    print ix, iy
    print model.get_lon_lat(ix, iy)




if __name__ == "__main__":
    t0 = time.clock()
    amno_from_index_space_to_lat_lon(131, 101)

    the_list = [
        [131, 101], [128, 124], [113, 110],
        [120, 120], [131, 112], [130, 97],
        [117, 108]
    ]

    print amno_convert_list(the_list)

    print 'Execution time is %f seconds' % (time.clock() - t0)
