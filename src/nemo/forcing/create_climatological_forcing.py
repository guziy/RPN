import os

__author__ = 'huziy'

from cdo import Cdo

from nemo import create_links_to_forcing_files
varnames = [
    "q2", "t2", "radsw", "radlw", "snow", "precip", "u10", "v10"
]


def main(in_folder = "/skynet1_rech3/huziy/EXP_0.1deg/DFS4.3_interpolated",
         out_folder = "/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/DFS4.3_clim"):

    create_links_to_forcing_files.create_links(expdir=out_folder, forcing_dir=in_folder)
    cdo_obj = Cdo()
    for vname in varnames:
        opath = os.path.join(out_folder, "{0}.nc".format(vname))
        inpaths = os.path.join(out_folder, "{0}_y*.nc".format(vname))
        cdo_obj.ensmean(input = inpaths, output = opath, options = "-f nc")

        print("processed: {0}".format(vname))



def main_improved(in_folder=""):
    print("Processing {}".format(in_folder))
    out_folder = in_folder + "_clim"

    if not os.path.isfile(out_folder):
        os.makedirs(out_folder)

    cdo_obj = Cdo()
    opath = os.path.join(out_folder, "mean_year.nc")
    inpaths = os.path.join(in_folder, "*")
    cdo_obj.ensmean(input=inpaths, output=opath, options="-f nc")


if __name__ == "__main__":
    # main()
    # main(in_folder="/skynet3_rech1/huziy/NEMO_OFFICIAL/DFS5.2_interpolated",
    #      out_folder="/home/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/DFS5.2_clim")


    data_root = "/home/huziy/skynet3_rech1/ERA-Interim_0.75_NEMO_pilot"
    for fname in os.listdir(data_root):
        main_improved(in_folder=os.path.join(data_root, fname))