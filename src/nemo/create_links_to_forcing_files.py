__author__ = 'huziy'

EXP_DIR = "/home/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/EXP_0.1deg_1958-2006"
FORCING_DIR = "/home/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/EXP_0.1deg_1958-2006/" \
              "DFS4.3_interpolated"

start_year = 1958
end_year = 2006

variable_list = [
    "precip", "q2", "t2", "radsw", "snow", "u10", "v10", "radlw"
]

import os


def get_years_from_name(fname):
    import re

    groups = re.findall(r"\d+", fname)
    if "_to_" not in fname:
        return [int(groups[-1])]
    else:
        s_year = int(groups[-2])
        e_year = int(groups[-1])
        return range(s_year, e_year + 1)


def create_links(expdir = "", forcing_dir = ""):

    """
    :param expdir:
    :param forcing_dir:
    """

    the_cwd = os.getcwd()
    os.chdir(expdir)

    for the_var in variable_list:
        the_dir = os.path.join(forcing_dir, the_var)
        for fname in os.listdir(the_dir):
            fpath = os.path.join(the_dir, fname)

            years = get_years_from_name(fname)
            for y in years:
                link_name = "{0}_y{1}.nc".format(the_var, y)
                if os.path.islink(link_name):
                    os.unlink(link_name)
                os.symlink(fpath, link_name)

    #return to the previous working dir
    os.chdir(the_cwd)



if __name__ == "__main__":
    create_links(expdir = EXP_DIR, forcing_dir = FORCING_DIR)