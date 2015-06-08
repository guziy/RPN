from collections import OrderedDict
from pathlib import Path
from matplotlib.dates import MonthLocator, num2date
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from rpn.rpn import RPN
from crcm5.analyse_hdf.climate_change.plot_cc_for_each_basin_hydrographs import get_basin_to_outlet_indices_map
from crcm5.analyse_hdf.run_config import RunConfig
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis
import matplotlib.pyplot as plt
from util import plot_utils

__author__ = 'huziy'

img_folder = Path("cc_paper/bfe/1D")

import numpy as np


def plot_comparison_hydrographs(basin_name_to_out_indices_map, rea_config=None, gcm_config=None):
    """

    :type basin_name_to_out_indices_map: dict
    """
    assert isinstance(rea_config, RunConfig)
    assert isinstance(gcm_config, RunConfig)

    assert hasattr(rea_config, "data_daily")
    assert hasattr(gcm_config, "data_daily")

    bname_to_indices = OrderedDict([item for item in sorted(basin_name_to_out_indices_map.items(),
                                                            key=lambda item: item[1][1], reverse=True)])

    print(bname_to_indices)

    plot_utils.apply_plot_params(font_size=12, width_pt=None, width_cm=25, height_cm=12)
    fig = plt.figure()
    ncols = 3
    nrows = len(bname_to_indices) // ncols + int(len(bname_to_indices) % ncols != 0)
    gs = GridSpec(nrows, ncols)

    ax_last = None
    for pl_index, (bname, (i, j)) in enumerate(bname_to_indices.items()):
        row = pl_index // ncols
        col = pl_index % ncols
        ax = fig.add_subplot(gs[row, col])

        ax.plot(rea_config.data_daily[0], rea_config.data_daily[1][:, i, j], color="b", lw=2,
                label=rea_config.label)
        ax.plot(gcm_config.data_daily[0], gcm_config.data_daily[1][:, i, j], color="r", lw=2,
                label=gcm_config.label)


        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda d, pos: num2date(d).strftime("%b")[0]))

        sfmt = ScalarFormatter(useMathText=True)
        sfmt.set_powerlimits([-2, 2])
        ax.yaxis.set_major_formatter(sfmt)

        bbox_props = dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=1, alpha=0.5)
        ax.annotate(bname, (0.9, 0.1), xycoords="axes fraction", bbox=bbox_props, zorder=10,
                    alpha=0.5, horizontalalignment="right", verticalalignment="bottom")

        ax_last = ax

    ax_last.legend(loc="upper right", bbox_to_anchor=(1, -0.2), borderaxespad=0)

    img_file = img_folder.joinpath("bfe_hydrographs.png")
    with img_file.open("wb") as f:
        fig.tight_layout()
        fig.savefig(f, bbox_inches="tight", format="png")


def main():
    import application_properties
    application_properties.set_current_directory()


    # Create folder for output images
    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)


    rea_driven_path = "/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5"
    rea_driven_label = "CRCM5-L-ERAI"

    gcm_driven_path = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"
    gcm_driven_label = "CRCM5-L-CanESM2"

    start_year_c = 1980
    end_year_c = 2010

    varname = "STFL"


    params = dict(
        data_path=rea_driven_path, start_year=start_year_c, end_year=end_year_c, label=rea_driven_label)

    geo_data_file = "/skynet3_rech1/huziy/hdf_store/pm1979010100_00000000p"

    rea_driven_config = RunConfig(**params)
    params.update(dict(data_path=gcm_driven_path, label=gcm_driven_label))

    gcm_driven_config = RunConfig(**params)

    r_obj = RPN(geo_data_file)
    facc = r_obj.get_first_record_for_name("FAA")
    fldr = r_obj.get_first_record_for_name("FLDR")

    bmp_info = analysis.get_basemap_info_from_hdf(file_path=rea_driven_path)

    basin_name_to_out_indices_map, basin_name_to_basin_mask = get_basin_to_outlet_indices_map(bmp_info=bmp_info,
                                                                                              accumulation_areas=facc,
                                                                                              directions=fldr)



    # select lake rich basins
    sel_basins = ["ARN", "PYR", "LGR", "RDO", "SAG", "WAS"]
    basin_name_to_out_indices_map = {k: v for k, v in basin_name_to_out_indices_map.items() if k in sel_basins}


    rea_driven_daily = analysis.get_daily_climatology_for_rconf(rea_driven_config, var_name=varname, level=0)
    gcm_driven_daily = analysis.get_daily_climatology_for_rconf(gcm_driven_config, var_name=varname, level=0)

    rea_driven_config.data_daily = rea_driven_daily
    gcm_driven_config.data_daily = gcm_driven_daily

    plot_comparison_hydrographs(basin_name_to_out_indices_map, rea_config=rea_driven_config, gcm_config=gcm_driven_config)

if __name__ == '__main__':
    main()
