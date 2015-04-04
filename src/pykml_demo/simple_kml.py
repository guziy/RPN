__author__ = 'huziy'

import numpy as np
from pykml.factory import KML_ElementMaker as KML
from lxml import etree

def main():
    print(dir(KML))
    pm1 = KML.Placemark(
        KML.name("name"),
        KML.Point(
            KML.coordinates("-64.5253,18.4607")
        )
    )
    print(dir(KML))
    print(etree.tostring(pm1, pretty_print=True))

    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print("Hello world")
  