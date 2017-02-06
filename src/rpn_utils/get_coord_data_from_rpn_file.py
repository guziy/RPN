from rpn.rpn import RPN

from domains.rotated_lat_lon import RotatedLatLon



class IndexSubspace(object):

    def __init__(self, i_start=-1, i_end=-1, j_start=-1, j_end=-1):
        self.i_start = int(i_start)
        self.i_end = int(i_end)
        self.j_start = int(j_start)
        self.j_end = int(j_end)

    def get_islice(self):
        return slice(self.i_start, self.i_end - 1)

    def get_jslice(self):
        return slice(self.j_start, self.j_end - 1)



def get_lons_lats_basemap(rpnfile_path="", varname=None, index_subset=None):


    """
    Get longitudes, latitudes and the basemap object corresponding to the rpn file
    :param rpnfile_path:
    :param varname:
    :return:
    """
    with RPN(rpnfile_path) as r:

        assert isinstance(r, RPN)

        if varname is None:
            varname = next(v for v in r.get_list_of_varnames() if v not in [">>", "^^", "HY"])

        r.get_first_record_for_name(varname)

        lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()

        nx, ny = lons.shape

        if index_subset is None:
            index_subset = IndexSubspace(i_start=0, i_end=nx - 1, j_start=0, j_end=ny - 1)






        rll = RotatedLatLon(**r.get_proj_parameters_for_the_last_read_rec())
        bmp = rll.get_basemap_object_for_lons_lats(lons2d=lons[index_subset.get_islice(), index_subset.get_jslice()],
                                                   lats2d=lats[index_subset.get_islice(), index_subset.get_jslice()])

        return lons, lats, bmp
