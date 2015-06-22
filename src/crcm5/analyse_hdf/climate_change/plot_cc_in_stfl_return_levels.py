import matplotlib
matplotlib.use("Agg")

from crcm5.analyse_hdf.climate_change.plot_cc_for_each_basin_hydrographs import BASIN_BOUNDARIES_FILE
from util import plot_utils


from matplotlib import cm

from mpl_toolkits.basemap import maskoceans

from crcm5.analyse_hdf.return_levels.calc_return_levels_and_unc_using_bootstrap import \
    get_return_levels_and_unc_using_bootstrap

from pathlib import Path
from rpn.rpn import RPN
from crcm5.analyse_hdf.run_config import RunConfig
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

__author__ = 'huziy'

img_folder = Path("cc_paper/return_levels")


def main():
    import application_properties

    application_properties.set_current_directory()

    # Create folder for output images
    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)

    rea_driven_path = "/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5"
    rea_driven_label = "CRCM5-L-ERAI"

    # gcm_driven_path_c = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-r-cc-canesm2-1980-2010.hdf5"
    gcm_driven_path_c = "/home/huziy/skynet3_rech1/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"
    gcm_driven_label_c = "CRCM5-L"

    start_year_c = 1980
    end_year_c = 2010

    varname = "STFL"

    future_shift_years = 90

    params = dict(
        data_path=rea_driven_path, start_year=start_year_c, end_year=end_year_c, label=rea_driven_label)

    geo_data_file = "/skynet3_rech1/huziy/hdf_store/pm1979010100_00000000p"

    rea_driven_config = RunConfig(**params)
    params.update(dict(data_path=gcm_driven_path_c, label=gcm_driven_label_c))

    gcm_driven_config_c = RunConfig(**params)
    gcm_driven_config_f = gcm_driven_config_c.get_shifted_config(shift_years=future_shift_years)

    r_obj = RPN(geo_data_file)
    facc = r_obj.get_first_record_for_name("FAA")
    fldr = r_obj.get_first_record_for_name("FLDR")

    # get basemap information
    bmp_info = analysis.get_basemap_info_from_hdf(file_path=rea_driven_path)

    rs_gcm_c = get_return_levels_and_unc_using_bootstrap(gcm_driven_config_c, varname=varname)

    rs_gcm_f = get_return_levels_and_unc_using_bootstrap(gcm_driven_config_f, varname=varname)

    plot_utils.apply_plot_params(font_size=10, width_cm=20, height_cm=18)

    # Plot return level changes
    fig = plt.figure()
    nplots = 0
    for the_type, rp_to_rl in rs_gcm_c.return_lev_dict.items():
        nplots += len(rp_to_rl)
    ncols = 2
    nrows = nplots // ncols + int(nplots % ncols != 0)
    gs = GridSpec(nrows, ncols + 1, width_ratios=[1.0, ] * ncols + [0.05, ])

    xx, yy = bmp_info.get_proj_xy()

    cmap = cm.get_cmap("bwr", 20)

    limits = {
        "high": (-50, 50),
        "low": (-150, 150)
    }

    for row, (the_type, rp_to_rl) in enumerate(sorted(rs_gcm_c.return_lev_dict.items(), key=lambda itm: itm[0])):

        for col, rp in enumerate(sorted(rp_to_rl)):

            ax = fig.add_subplot(gs[row, col])
            rl = rp_to_rl[rp]

            # Ignore 0 return levels in the current climate for percentage calculations
            rl = np.ma.masked_where(rl <= 0, rl)

            rl_future = rs_gcm_f.return_lev_dict[the_type][rp]

            # Calculate climate change signal
            diff = (rl_future - rl) / rl * 100

            diff = maskoceans(bmp_info.lons, bmp_info.lats, diff)

            std_c = rs_gcm_c.std_dict[the_type][rp]
            std_f = rs_gcm_f.std_dict[the_type][rp]

            significance = (np.ma.abs(diff) >= 1.96 * (std_c + std_f)) & (~diff.mask)
            significance = significance.astype(int)

            vmin, vmax = limits[the_type]
            im = bmp_info.basemap.pcolormesh(xx, yy, diff, vmin=vmin, vmax=vmax, cmap=cmap)

            cs = bmp_info.basemap.contourf(xx, yy, significance, levels=[0, 0.5, 1], hatches=["////", None, None],
                                           colors="none")

            if row == nrows - 1 and col == ncols - 1:
                # create a legend for the contour set
                artists, labels = cs.legend_elements()
                ax.legend([artists[0], ], ["not sign.", ], handleheight=0.5,
                          bbox_to_anchor=(1, -0.1), loc="upper right", borderaxespad=0.)

            ax.set_title("T = {}-year".format(rp))

            if col == 0:
                ax.set_ylabel("{} flow".format(the_type))

            bmp_info.basemap.drawcoastlines(ax=ax)
            bmp_info.basemap.drawmapboundary(fill_color="0.75")
            bmp_info.basemap.readshapefile(".".join(BASIN_BOUNDARIES_FILE.split(".")[:-1]).replace("utm18", "latlon"),
                                           "basin",
                                           linewidth=1.2, ax=ax)

            if col == ncols - 1:
                cax = fig.add_subplot(gs[row, -1])
                plt.colorbar(im, cax=cax, extend="both")
                cax.set_title("%")

    img_file = img_folder.joinpath("rl_cc_{}.png".format(gcm_driven_config_c.label))

    with img_file.open("wb") as f:
        fig.savefig(f, bbox_inches="tight")


if __name__ == '__main__':
    import time

    t0 = time.clock()
    main()
    print("Execution time: {}s".format(time.clock() - t0))
