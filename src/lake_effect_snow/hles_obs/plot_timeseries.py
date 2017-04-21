

# plot area averaged HLES for each year
from pathlib import Path

from mpl_toolkits.basemap import maskoceans
from netCDF4 import Dataset
import matplotlib.pyplot as plt

from application_properties import main_decorator
from util import plot_utils

from collections import OrderedDict
import numpy as np

@main_decorator
def main():


    label_to_hles_dir = OrderedDict(
        [
         ("Obs", Path("/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_Obs_1980-2009")),
         ("CRCM5_NEMO", Path("/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_NEMO_1980-2009")),
         ("CRCM5_HL", Path("/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_Hostetler_1980-2009"))
        ]
    )


    label_to_line_style = {
        "Obs": "k",
        "CRCM5_NEMO": "r",
        "CRCM5_HL": "b"
    }

    label_to_line_color = {
        "Obs": "k",
        "CRCM5_NEMO": "r",
        "CRCM5_HL": "b"
    }




    vname = "snow_fall"
    units = "cm"

    #vname = "lkeff_snowfall_days"
    #units = "days"

    label_to_y_to_snfl = {}

    mask = None

    plot_utils.apply_plot_params(font_size=12)

    fig = plt.figure()

    years = None
    for label, folder in label_to_hles_dir.items():

        y_to_snfl = {}
        y_to_snfldays = {}

        for the_file in folder.iterdir():
            if not the_file.name.endswith(".nc"):
                continue

            with Dataset(str(the_file)) as ds:
                print(ds)
                snfl = ds.variables[vname][:]
                year_current = ds.variables["year"][:]

                if mask is None:
                    lons, lats = [ds.variables[k][:] for k in ["lon", "lat"]]
                    lons[lons > 180] -= 360
                    mask = maskoceans(lons, lats, lons, inlands=True, resolution="i")


                y_to_snfl[year_current[0]] = snfl[0].mean()


        years_ord = sorted(y_to_snfl)

        label_to_y_to_snfl[label] = y_to_snfl




        # Calculate the correlation and add it to the line label.
        the_line_label = label
        if label not in ["Obs"]:


            arr = np.array([
                [label_to_y_to_snfl[label][y] for y in years_ord],
                [label_to_y_to_snfl["Obs"][y] for y in years_ord]
            ])

            r = np.corrcoef(arr)

            print(r)
            the_line_label = "{} (R={:.2f})".format(the_line_label, r[0, 1])




        ts_data = np.array([y_to_snfl[y] for y in years_ord])
        ts_data = (ts_data - ts_data.mean()) / ts_data.std()
        plt.plot(years_ord, ts_data, label_to_line_style[label], linewidth=2,
                 label=the_line_label)

        if years is None:
            years = years_ord



    plt.legend(loc="upper right")

    plt.ylabel(units)
    plt.xlabel("Year")
    plt.xticks(years)

    plt.grid()
    plt.gcf().autofmt_xdate()
    plt.savefig(str(label_to_hles_dir["Obs"].joinpath("area_avg_{}.png".format(vname))), bbox_inches="tight", dpi=400)
    plt.close()


    # boxplot
    plot_utils.apply_plot_params(width_cm=8, height_cm=4.5, font_size=8)
    fig = plt.figure()


    all_vals = []
    all_labels = []
    for label in label_to_hles_dir:
        y_to_snfl = label_to_y_to_snfl[label]
        vals = [v for v in y_to_snfl.values()]
        all_vals.append(vals)
        all_labels.append(label)


    ax = plt.gca()
    ax.boxplot(all_vals, labels=all_labels)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    plt.savefig(str(label_to_hles_dir["Obs"].joinpath("area_avg_boxplots_{}.png".format(vname))), bbox_inches="tight", dpi=400)
    plt.close(fig)

if __name__ == '__main__':
    main()