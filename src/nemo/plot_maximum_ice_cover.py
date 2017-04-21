from datetime import datetime
import os
from nemo import nemo_commons
from nemo.nemo_output_manager import NemoOutputManager

__author__ = 'huziy'


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import maskoceans
import pandas as pd

column_to_lake_name_glsea = {
    4: "Superior",
    5: "Michigan",
    6: "Huron",
    7: "Erie",
    8: "Ontario"

}


def main():

    img_folder = "nemo/validation_1d/"
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)

    img_name_prefix = "max_ice_cover_"


    for col, lake_name in column_to_lake_name_glsea.items():
        folder_path = nemo_commons.lake_to_folder_with_sim_data_1981_2000_seplakes_exp[lake_name]
        fname_tgrid = None
        for fname in os.listdir(folder_path):
            if fname.endswith("grid_T.nc"):
                fname_tgrid = fname
                break

        if fname_tgrid is None:
            continue


        fig = plt.figure()
        ax = plt.gca()
        fpath = os.path.join(folder_path, fname_tgrid)
        nom = NemoOutputManager(
            file_path=fpath, var_name="soicecov"
        )
        ts_mod = nom.get_area_mean_timeseries()
        ts_mod = ts_mod.groupby(by=lambda d: d.year).max()
        ax.plot(ts_mod.index, ts_mod["NEMO"] * 100.0, color="r", lw=2, label="NEMO")

        ax.set_ylabel(r"%")
        ax.legend(loc=3)

        ax.set_title(lake_name)

        fig.savefig(os.path.join(img_folder, img_name_prefix + lake_name + ".png"))





    #plt.plot(dates, data[:, 4:], label = [column_to_lake_name_glsea[i + 4] for i in range(5)])



if __name__ == "__main__":
    main()