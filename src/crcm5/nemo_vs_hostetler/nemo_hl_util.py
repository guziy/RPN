from datetime import datetime
from pathlib import Path

import numpy as np
from rpn.domains.rotated_lat_lon import RotatedLatLon
from rpn.rpn import RPN

from crcm5.nemo_vs_hostetler import commons


def get_basemap_obj_and_coords_from_rpn_file(path=""):

    assert len(path) > 0, "The path should not be empty."
    r = RPN(path)
    vname = [v for v in r.get_list_of_varnames() if v.strip() not in [">>", "^^"]][0]
    r.get_first_record_for_name(vname)

    lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()
    projparams = r.get_proj_parameters_for_the_last_read_rec()

    rll = RotatedLatLon(**projparams)
    bmp = rll.get_basemap_object_for_lons_lats(lons2d=lons, lats2d=lats, resolution="l")
    return bmp, lons, lats


def get_monthyeardate_to_paths_map(file_prefix="pm", start_year=-np.Inf, end_year=np.Inf, samples_dir_path=None):

    if isinstance(samples_dir_path, str):
        samples_dir_path = Path(samples_dir_path)

    monthyear_to_pathlist = {}
    for monthdir in samples_dir_path.iterdir():

        y, m = commons.get_year_and_month(monthdir.name)

        if not (start_year <= y <= end_year):
            continue

        current_list = [str(p) for p in monthdir.iterdir() if p.name.startswith(file_prefix) and not p.name[-9:-1] == 8 * "0"]

        monthyear_to_pathlist[datetime(y, m, 1)] = current_list

    return monthyear_to_pathlist



class IndexRectangle(object):

    def __init__(self, ill=0, jll=0, ni=-1, nj=-1):
        """
        Represents an index selection inside a domain
        :param ill:
        :param jll:
        :param ni:
        :param nj:
        """
        self.ill = ill
        self.jll = jll
        self.ni = ni
        self.nj = nj

    def get_ur_corner(self):
        return self.ill + self.ni - 1, self.jll + self.nj - 1


    def get_2d_slice(self):
        return np.s_[self.ill: self.ill + self.ni, self.jll: self.jll + self.nj]