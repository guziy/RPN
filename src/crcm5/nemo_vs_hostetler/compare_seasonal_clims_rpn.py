import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from pathlib import Path
from rpn import level_kinds
from rpn.rpn_multi import MultiRPN

from application_properties import main_decorator
from crcm5.nemo_vs_hostetler import commons
from crcm5.nemo_vs_hostetler import nemo_hl_util
from crcm5.nemo_vs_hostetler.commons import get_year_and_month
from util import plot_utils

img_folder = "nemo_vs_hostetler"


def get_seasonal_means_from_rpn_monthly_folders(samples_dir="", season_to_months=None, start_year=-np.Inf, end_year=np.Inf,
                                                filename_prefix="pm", varname="", level=-1, level_kind=-1):

    result = OrderedDict()

    season_to_files = {s: [] for s in season_to_months}

    for fn in os.listdir(samples_dir):
        y, m = get_year_and_month(fn)

        # skip if the year is not in the selected range
        if not start_year <= y <= end_year:
            continue

        for season, months in season_to_months.items():
            if m in months:
                month_folder_path = os.path.join(samples_dir, fn)
                for filename in os.listdir(month_folder_path):

                    if filename[-9:-1] == 8 * "0":
                        continue


                    if filename.startswith(filename_prefix):
                        season_to_files[season].append(os.path.join(month_folder_path, filename))



    for season, months in season_to_months.items():
        mrpn = MultiRPN(season_to_files[season])
        date_to_field = mrpn.get_all_time_records_for_name_and_level(varname=varname, level=level, level_kind=level_kind)
        result[season] = np.mean([field for field in date_to_field.values()], axis=0)

    return result



@main_decorator
def main():

    start_year = 1980
    end_year = 1980

    HL_LABEL = "CRCM5_HL"
    NEMO_LABEL = "CRCM5_NEMO"

    # sim_label_to_path = OrderedDict(
    #     [(HL_LABEL, "/RESCUE/skynet3_rech1/huziy/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl_oneway/Samples"),
    #      (NEMO_LABEL, "/HOME/huziy/skynet3_rech1/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl/Samples")]
    # )


    sim_label_to_path = OrderedDict(
        [("Control", "/RESCUE/skynet3_rech1/huziy/CNRCWP/C5/2016/test_ouptuts/control/Samples"),
         ("Control+perftweaks", "/RESCUE/skynet3_rech1/huziy/CNRCWP/C5/2016/test_ouptuts/coupled-GL-perftest/Samples")]
    )



    #var_name_list = ["TT", "PR", "LC", "HR", "HU", "AV", "I5", "AL", "TJ"]
    var_name_list = ["TT", "PR", "LC", "HU", "I5"]

    # season_to_months = commons.season_to_months
    season_to_months = OrderedDict([("December", [12, ]),])

    vname_to_level = {
        "TT": 1, "PR": -1, "SN": -1, "LC": -1, "HR": 1, "HU": 1, "AV": -1, "I5": -1, "AL": -1, "TJ": -1
    }

    vname_to_level_kind = {
        "TT": level_kinds.HYBRID, "PR": level_kinds.ARBITRARY, "SN": level_kinds.ARBITRARY,
        "LC": level_kinds.ARBITRARY, "HR": level_kinds.HYBRID, "HU": level_kinds.HYBRID, "AV": level_kinds.ARBITRARY,
        "I5": level_kinds.ARBITRARY, "AL": level_kinds.ARBITRARY, "TJ": level_kinds.ARBITRARY
    }


    vname_to_file_prefix = {
                            "TT": "dm",
                            "PR": "pm",
                            "SN": "pm",
                            "LC": "pm",
                            "HR": "dm",
                            "HU": "dm",
                            "AV": "pm",
                            "I5": "pm",
                            "AL": "pm",
                            "TJ": "pm"
                            }

    vname_to_clevs = {
        "TT": np.arange(-5.1, 5.3, 0.2),
        "PR": np.arange(-5.1, 5.3, 0.2),
        "SN": np.arange(-5.1, 5.3, 0.2),
        "LC": [v for v in np.arange(-0.52, 0.54, 0.08)],
        "HR": [v for v in np.arange(-0.52, 0.54, 0.03)],
        "HU": np.arange(-0.5, 0.54, 0.04),
        "AV": np.arange(-150, 170, 20),
        "I5": np.arange(-30, 34, 4),
        "AL": [v for v in np.arange(-0.52, 0.54, 0.08)],
        "TJ": np.arange(-5.1, 5.3, 0.2)
    }


    vname_to_label = {
        "TT": "Air temperature, 2m",
        "PR": "Total precipitation"
    }


    vname_to_coeff = {
        "PR": 24 * 3600 * 1000,
        "HU": 1000
    }

    vname_to_units = {
        "TT": r"$^\circ$C",
        "PR": "mm/day",
        "HU": "g/kg",
        "AV": r"W/m$^2$",
        "I5": "mm"
    }



    # get a coord file ...
    coord_file = ""
    found_coord_file = False

    samples_with_coords_path = Path(next(sim_label_to_path.items())[1])

    for mdir in samples_with_coords_path.iterdir():


        if not mdir.is_dir():
            continue


        for fn in mdir.iterdir():
            print(fn)
            if fn.name[:2] not in ["pm", "dm", "pp", "dp"]:
                continue

            coord_file = str(fn)
            found_coord_file = True

        if found_coord_file:
            break


    bmp, lons, lats = nemo_hl_util.get_basemap_obj_and_coords_from_rpn_file(path=coord_file)
    xx, yy = bmp(lons, lats)

    # Do the calculations
    hl_data = OrderedDict()
    nemo_data = OrderedDict()
    for vname in var_name_list:

        field_props = dict(season_to_months=season_to_months, start_year=start_year, end_year=end_year,
                           filename_prefix=vname_to_file_prefix[vname], varname=vname, level=vname_to_level[vname],
                           level_kind=vname_to_level_kind[vname])

        hl_data[vname] = get_seasonal_means_from_rpn_monthly_folders(samples_dir=sim_label_to_path[HL_LABEL], **field_props)
        nemo_data[vname] = get_seasonal_means_from_rpn_monthly_folders(samples_dir=sim_label_to_path[NEMO_LABEL], **field_props)



    # Plotting
    plot_utils.apply_plot_params(font_size=6, width_cm=20, height_cm=20)
    fig = plt.figure()

    fig.suptitle("{} minus {}".format(NEMO_LABEL, HL_LABEL))

    nrows = len(var_name_list)
    gs = GridSpec(nrows, len(season_to_months) + 1, width_ratios=[1., ] * len(season_to_months) + [0.05, ])

    for row, vname in enumerate(hl_data):
        hl_seas = hl_data[vname]
        nemo_seas = nemo_data[vname]

        cs = None
        for col, season in enumerate(season_to_months):
            ax = fig.add_subplot(gs[row, col])

            norm = None

            if vname_to_clevs[vname] is not None:
                norm = BoundaryNorm(vname_to_clevs[vname], len(vname_to_clevs[vname]) - 1)
                cmap = cm.get_cmap("seismic", len(vname_to_clevs[vname]) - 1)
            else:
                cmap = cm.get_cmap("seismic", 11)


            cs = bmp.contourf(xx, yy, (nemo_seas[season] - hl_seas[season]) * vname_to_coeff.get(vname, 1), levels=vname_to_clevs[vname], ax=ax, extend="both", cmap=cmap, norm=norm)
            bmp.drawcoastlines(linewidth=0.3)

            if col == 0:
                ax.set_ylabel(vname_to_label.get(vname, vname))

            if row == 0:
                ax.set_title(season)

        cax = fig.add_subplot(gs[row, -1])
        plt.colorbar(cs, cax=cax)
        cax.set_title(vname_to_units.get(vname, "-"))


    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    img_file = os.path.join(img_folder, "seas_2d_diff_{}-{}.png".format(start_year, end_year))
    fig.savefig(img_file, dpi=commons.dpi, transparent=True, bbox_inches="tight")


if __name__ == '__main__':
    main()