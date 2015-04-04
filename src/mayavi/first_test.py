__author__ = 'huziy'

import numpy as np

from mayavi.sources.builtin_surface import BuiltinSurface
from mayavi import mlab
def main():
    continents_src = BuiltinSurface(source="earth", name="Continents")
    continents = mlab.pipeline.surface(continents_src, color=(0, 0, 0))

    mlab.figure(1, bgcolor=(0.48, 0.48, 0.48), fgcolor=(0, 0, 0), size=(400, 400))
    mlab.view(63.4, 73.8, 4, [-0.05, 0, 0])
    mlab.show()




    #TODO: implement
    pass

if __name__ == "__main__":
    #main()
    mlab.figure(1, bgcolor=(0.48, 0.48, 0.48), fgcolor=(0, 0, 0), size=(400, 400))
    mlab.test_plot3d()
    mlab.view(63.4, 73.8, 4, [-0.05, 0, 0])
    mlab.show()

    print("Hello world")
  