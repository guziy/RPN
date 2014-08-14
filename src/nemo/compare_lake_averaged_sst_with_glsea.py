from datetime import datetime
import os
from matplotlib.dates import date2num
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

    img_folder = "nemo/validation_1d"
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)

    img_name_prefix = "mean_surface_temp_"
    obs_data_path = "/skynet3_rech1/huziy/GLSEA/coastwatch.glerl.noaa.gov/ftp/glsea/avgtemps/1992-1998/glsea-temps1992-1998-schwab.dat"
    data = np.loadtxt(obs_data_path, skiprows=3)

    dates = [datetime(int(data[row, 0]), int(data[row, 2]), int(data[row, 3])) for row in range(data.shape[0])]





    for col, lake_name in column_to_lake_name_glsea.iteritems():
        folder_path = nemo_commons.lake_to_folder_with_sim_data_1981_2000_seplakes_exp[lake_name]
        fname_tgrid = None
        for fname in os.listdir(folder_path):
            if fname.endswith("grid_T.nc"):
                fname_tgrid = fname
                break

        if fname_tgrid is None:
            continue

        print "Processing {}".format(lake_name)
        fig = plt.figure()
        ax = plt.gca()
        fpath = os.path.join(folder_path, fname_tgrid)
        nom = NemoOutputManager(
            file_path=fpath, var_name="sosstsst"
        )
        ts_obs = pd.DataFrame(index=dates, data=data[:, col], columns=["GLSEA",])
        ts_mod = nom.get_area_mean_timeseries()

        ts_mod = ts_mod.truncate(before=ts_obs.index[0], after=ts_obs.index[-1])

        ax.plot(ts_obs.index, ts_obs["GLSEA"], color="b", lw=2, label="GLSEA")
        ax.plot(ts_mod.index, ts_mod["NEMO"], color="r", lw=2, label="NEMO")

        ax.set_ylabel(r"${\rm ^{\circ}C}$")

        if col == 5:
            ax.legend(ncol=2)

        ax.set_title(lake_name)

        fig.savefig(os.path.join(img_folder, img_name_prefix + lake_name + ".png"))





    #plt.plot(dates, data[:, 4:], label = [column_to_lake_name_glsea[i + 4] for i in range(5)])






if __name__ == "__main__":
    main()