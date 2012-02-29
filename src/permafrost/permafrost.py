__author__ = 'huziy'


from ccc.ccc import champ_ccc

import application_properties
import numpy as np
import os
#from matplotlib.patches import Polygon
from shapely.geometry import Point
from osgeo import ogr


name_to_abbr = {
    "alberta" : "AB",
    "saskatchewan" : "SK",
    "manitoba" : "MB",
    "newfoundland  & labrador" : "NL",
    "prince edward island": "PE",
    "nova scotia" : "NS",
    "northwest territories":"NT" ,
    "nunavut" : "NU",
    "ontario": "ON",
    "new brunswick" : "NB",
    "yukon territory" : "YT",
    "british columbia": "BC",
    "quebec" : "QC"
}

def plot_stations(the_basemap):
    folder_path = "data/permafrost"
    file_names = [ "stationNumbersCont.txt" , "stationNumbersDisc.txt", "stationNumbersSpor.txt" ]
    marks = ["o", "+", 'd']
    for fName, the_mark in zip(file_names, marks):
        path = os.path.join(folder_path, fName)
        lines = open(path).readlines()
        lons = []
        lats = []
        for line in lines:
            line = line.strip()
            if line == "": continue
            fields = line.split()
            lons.append(float(fields[1]))
            lats.append(float(fields[2]))


        lons = -np.array(lons)
        lons, lats = the_basemap(lons, lats)
        the_basemap.scatter(lons, lats, marker = the_mark, c = "none", s = 20, linewidth = 1)


def inside_region(lon, lat, geometries):
    """
    Check whether the point is inside the region of interest
    """
    point = ogr.CreateGeometryFromWkt(Point(lon, lat).wkt)

    for g in geometries:
        if g.Intersect(point):
            return True

    return False
    pass


def main():
    data_path = "data/permafrost/p1perma"
    cccObj = champ_ccc(data_path)
    pData = cccObj.charge_champs()[0]["field"]




if __name__ == "__main__":
    application_properties.set_current_directory()
    main()

