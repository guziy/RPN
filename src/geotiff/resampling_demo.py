__author__ = 'huziy'

import numpy as np
from pyresample import geometry, image

from pyresample import plot


def main():

    lons1 = range(10)
    lats1 = range(10)

    lons2 = range(0, 10, 2)
    lats2 = range(0, 10, 2)

    lats1, lons1 = np.meshgrid(lats1, lons1)
    lats2, lons2 = np.meshgrid(lats2, lons2)

    gd1 = geometry.SwathDefinition(lons=lons1, lats = lats1)
    gd2 = geometry.SwathDefinition(lons=lons2, lats= lats2)

    data = np.ones(lons1.shape)
    img1 = image.ImageContainerQuick(data, gd1, nprocs=5, fill_value=None)

    img2 = img1.resample(gd2)
    plot.show_quicklook(gd2, img2.image_data, label="test")






    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  