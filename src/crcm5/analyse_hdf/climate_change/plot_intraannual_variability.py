from pathlib import Path
import collections
from matplotlib.gridspec import GridSpec
from rpn.rpn import RPN
from crcm5.analyse_hdf.run_config import RunConfig
from util import plot_utils
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis

import matplotlib.pyplot as plt

__author__ = 'huziy'

img_folder = Path("cc_paper/intraannual_variability")

DataConfig = collections.namedtuple("DataConfig",
                                    "basemap_info no_lakes_c no_lakes_f with_lakes_c with_lakes_f")
Data = collections.namedtuple("Data", "no_lakes_c no_lakes_f with_lakes_c with_lakes_f")


def plot_comparison_std(configs, data, varname=""):
    assert isinstance(configs, DataConfig)
    assert isinstance(data, Data)

    gs = GridSpec(3, 3)

    fig = plt.figure()

    # Calculate stds and differences
    to_plot = [
        [data.no_lakes_c.std(axis=0), data.no_lakes_f.std(axis=0)],
        [data.with_lakes_c.std(axis=0), data.with_lakes_f.std(axis=0)],
        []
    ]

    for i in range(2):
        to_plot[i].append((to_plot[i][1] - to_plot[i][0]) / to_plot[i][0])


    for i in range(3):
        denom = 1.0 if i == 2 else to_plot[0][i]
        to_plot[2].append((to_plot[1][i] - to_plot[0][i]) / denom)




    col_titles = [
        "Current ({}-{})".format(configs.no_lakes_c.start_year, configs.no_lakes_c.end_year),
        "Future ({}-{})".format(configs.no_lakes_f.start_year, configs.no_lakes_f.end_year),
        "Future - Current"
    ]

    # noinspection PyListCreation
    row_titles = [
        "{}".format(configs.no_lakes_c.label),
        "{}".format(configs.with_lakes_c.label)
    ]

    row_titles.append("{} - {}".format(*row_titles[::-1]))

    bmp = configs.basemap_info.basemap
    xx, yy = configs.basemap_info.get_proj_xy()

    # Plot values
    for row in range(3):
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])

            if row == 0:
                ax.set_title(col_titles[col])

            if col == 0:
                ax.set_ylabel(row_titles[row])

            im = bmp.pcolormesh(xx, yy, to_plot[row][col], ax=ax)
            bmp.colorbar(im, ax=ax)
            bmp.drawcoastlines(ax=ax)



    # Construct image file name
    img_file = img_folder.joinpath("{}_L_vs_NL_{}-{}_{}-{}.png".format(
        varname, configs.no_lakes_c.start_year, configs.no_lakes_c.end_year,
        configs.no_lakes_c.start_year, configs.no_lakes_c.end_year, ))

    with img_file.open("wb") as imf:
        fig.savefig(imf, format="png", bbox_inches="tight")


def main():
    import application_properties

    application_properties.set_current_directory()

    plot_utils.apply_plot_params(font_size=12, width_cm=25, height_cm=25)

    # Create folder for output images
    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)

    with_lakes_c_path = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"
    with_lakes_label = "CRCM5-L"

    no_lakes_c_path = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-r-cc-canesm2-1980-2010.hdf5"
    no_lakes_label = "CRCM5-NL"

    start_year_c = 1980
    end_year_c = 2010

    varname = "STFL"

    nyears_to_future = 75

    with_lakes_config_c = RunConfig(data_path=with_lakes_c_path, start_year=start_year_c, end_year=end_year_c,
                                    label=with_lakes_label)
    with_lakes_config_f = with_lakes_config_c.get_shifted_config(nyears_to_future)

    no_lakes_config_c = RunConfig(data_path=no_lakes_c_path, start_year=start_year_c, end_year=end_year_c,
                                  label=no_lakes_label)
    no_lakes_config_f = no_lakes_config_c.get_shifted_config(nyears_to_future)

    bmp_info = analysis.get_basemap_info_from_hdf(file_path=with_lakes_c_path)


    # Calculate daily climatologic fields
    with_lakes_c_daily = analysis.get_daily_climatology_for_rconf(with_lakes_config_c, var_name=varname, level=0)
    with_lakes_f_daily = analysis.get_daily_climatology_for_rconf(with_lakes_config_f, var_name=varname, level=0)

    no_lakes_c_daily = analysis.get_daily_climatology_for_rconf(no_lakes_config_c, var_name=varname, level=0)
    no_lakes_f_daily = analysis.get_daily_climatology_for_rconf(no_lakes_config_f, var_name=varname, level=0)

    configs = DataConfig(bmp_info, no_lakes_config_c, no_lakes_config_f, with_lakes_config_c, with_lakes_config_f)
    args = (no_lakes_c_daily, no_lakes_f_daily, with_lakes_c_daily, with_lakes_f_daily)
    data = Data(*[arg[1] for arg in args])

    plot_comparison_std(configs, data, varname=varname)


if __name__ == '__main__':
    main()
