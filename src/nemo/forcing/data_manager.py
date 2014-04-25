import os
import re

__author__ = 'huziy'




def _get_year_list_from_name(filename = ""):

    """
    >>> folder_path = "/skynet3_rech1/huziy/NEMO_OFFICIAL/DFS5.2/precip"
    >>> for fname in os.listdir(folder_path):
    ...        if "-" not in fname: continue
    ...        _get_year_list_from_name(fname)
    [1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978]

    >>> for fname in os.listdir(folder_path):
    ...     if "1979" not in fname: continue
    ...     _get_year_list_from_name(fname)
    [1979]

    :param filename:
    :return:
    """
    groups = re.findall(r"\d+", filename)
    if "-" not in filename:
        year = int(groups[-1])
        return [year, ]

    start, end = [int(token) for token in groups[-2:]]
    return range(start, end + 1)





class DFSDataManager(object):
    def __init__(self, folder_path = "/skynet3_rech1/huziy/NEMO_OFFICIAL/DFS5.2"):
        self.folder_path = folder_path


    def get_daily_climatology(self, start_year = None, end_year = None, var_name = "t2"):
        var_folder = os.path.join(self.folder_path, var_name)

        year_2_path = {}
        for fname in os.listdir(var_folder):
            for y in _get_year_list_from_name(fname):
                year_2_path[y] = os.path.join(var_folder, fname)
            pass

        print year_2_path
        #TODO: implement
        pass


