from collections import OrderedDict

from data.convert_simulation_output_for_pcic import main_for_parallel_processing
import multiprocessing
from multiprocessing import Pool

def parallel_conversion_entry():

    """

    :return:
    """

    """
    fields = ["PR", "AD", "AV", "GIAC",
              "GIML", "GLD", "GLF", "GSAB",
              "GSAC", "GSML", "GVOL", "GWDI",
              "GWST", "GZ", "HR", "HU", "I1", "I2", "I4",
              "I5", "MS", "N3", "N4", "P0", "PN", "PR", "S6", "SD",
              "STFL", "SWSL", "SWSR", "T5", "T9", "TDRA", "TJ", "TRAF", "UD", "VD"]
    """

    # fields = ["TT", "PR", "N3", "AV", "GZ", "P0", "SN", "N4", "TJ", "HU", ]
    fields = ["LC", ]
    start_year = 1989
    end_year = 2010

    label_to_simpath = OrderedDict()
    # label_to_simpath[f"CanESM2_GL_{start_year}-{end_year}"] = "/scratch/huziy/Output/GL_CC_CanESM2_RCP85/coupled-GL-current_CanESM2/Samples"
    label_to_simpath[f"CanESM2_GL_{start_year}-{end_year}"] = "/home/huziy/data/big1/Projects/HLES_GL_NEMO_CRCM5_CC/coupled-GL-current_CanESM2/Samples"

    input = [[start_year, end_year, fname, label_to_simpath] for fname in fields]

    nprocs = min([multiprocessing.cpu_count(), 1, len(fields)])
    p = Pool(processes=min(nprocs, len(fields)))

    # do the conversion in parallel
    p.map(main_for_parallel_processing, input)


if __name__ == '__main__':
    parallel_conversion_entry()
