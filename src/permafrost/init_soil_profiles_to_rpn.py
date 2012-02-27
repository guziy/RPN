import os
import itertools
import re
import level_kinds
from rpn import RPN

__author__ = 'huziy'

import numpy as np
import application_properties


def read_coordinates(coords_file = "data/1950-1960-ECHO-G-profiles/coords.txt"):
    """
    returns data field dimensions
    """
    f = open(coords_file)

    lines = map(lambda x: x.strip(), f.readlines())
    id_to_hor_indices = {}
    ni = 0
    nj = 0
    land_sea_mask = {}
    for line in lines:
        if line == "": continue
        fields = line.split()
        id, j, i = map(int, fields[:3])
        if i > ni: ni = i
        if j > nj: nj = j
        j -= 1
        i -= 1
        id_to_hor_indices[id] = (i, j)
        if line.endswith("land"):
            land_sea_mask[(i,j)] = 1
        else:
            land_sea_mask[(i,j)] = 0


    land_sea = np.zeros((ni, nj))
    for i in xrange(ni):
        for j in xrange(nj):
            land_sea[i,j] = land_sea_mask[(i,j)]



    return ni, nj, id_to_hor_indices, land_sea
    pass

def read_data(folder = "data/1950-1960-ECHO-G-profiles"):
    ni, nj, id_to_hor_indices, land_sea_mask = read_coordinates()
    nz = -1
    all_data = []
    for year_folder in os.listdir(folder):
        year_folder_path = os.path.join(folder, year_folder)
        if not os.path.isdir(year_folder_path):
            continue
        data = None
        for file in os.listdir(year_folder_path):
            file_path = os.path.join(year_folder_path, file)
            f = open(file_path)
            lines = map(lambda x: x.strip(), f.readlines())
            lines = list(itertools.ifilter(lambda x: len(x) > 0, lines))

            if data is None:
                nz = len(lines)
                data = np.zeros((ni, nj, nz))

            the_id = int(re.findall("\d+", file)[0])
            i, j = id_to_hor_indices[the_id]
            profile = map(lambda x: float(x.split()[-1]), lines)
            data[i, j, :] = np.array(profile)[:]
            f.close()
        all_data.append( np.fliplr( data ))

    mean_data = np.mean(all_data, axis = 0)
    rpn_obj = RPN("data/soil_profiles/{0}.rpn".format("profile_200"), mode="w")

    rpn_obj.write_2D_field(grid_type="A", ig = [0, 0, 0, 0],data=np.fliplr(land_sea_mask), name = "MASK")

    for level in xrange(nz):
        #longitudinal grid length is 360/NI. For such a grid,
        #IG1 contains the domain of the grid: 0: Global
        #1: Northern Hemisphere
        #2: Southern Hemisphere IG2 contains the orientation of the grid:
        #0: South -> North (pt (1,1) is at the bottom of the grid)
        #1: North -> South (pt (1,1) is at the top of the grid) IG3 should be 0.
        #IG4 should be 0.

        rpn_obj.write_2D_field(grid_type="A", ig = [0, 0, 0, 0],
            data= mean_data[:,:, level], level = level, level_kind=level_kinds.HEIGHT_METERS_OVER_SEA_LEVEL,
            name="TBAR"
        )
    rpn_obj.close()

def main():
    read_data()
    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print "Hello world"
  