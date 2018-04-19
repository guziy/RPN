from collections import OrderedDict

from data.convert_simulation_output_for_pcic import main_for_parallel_processing
import multiprocessing
from multiprocessing import Pool

def parallel_conversion_entry():

    fields = ["PR", "AD", "AV", "GIAC",
              "GIML", "GLD", "GLF", "GSAB",
              "GSAC", "GSML", "GVOL", "GWDI",
              "GWST", "GZ", "HR", "HU", "I1", "I2", "I4",
              "I5", "MS", "N3", "N4", "P0", "PN", "PR", "S6", "SD",
              "STFL", "SWSL", "SWSR", "T5", "T9", "TDRA", "TJ", "TRAF", "UD", "VD"]


    fields = ["T5", "T9"]

    start_year = 1980
    end_year = 2014


    label_to_simpath = OrderedDict()
    label_to_simpath["WC011_modified"] = "/snow3/huziy/NEI/WC/NEI_WC0.11deg_Crr1/Samples/"

    input = [[start_year, end_year, fname, label_to_simpath] for fname in fields]


    nprocs = min(multiprocessing.cpu_count(), 10)
    p = Pool(processes=min(nprocs, len(fields)))

    # do the conversion in parallel
    p.map(main_for_parallel_processing, input)



if __name__ == '__main__':
    parallel_conversion_entry()
