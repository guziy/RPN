from collections import namedtuple
from pathlib import Path
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from nemo.nemo_yearly_files_manager import NemoYearlyFilesManager

__author__ = 'huziy'

# Compare 2 Nemo outputs

import matplotlib.pyplot as plt
import numpy as np


def main_compare_max_yearly_ice_conc():
    """
    ice concentration
    """
    var_name = ""

    start_year = 1979
    end_year = 1985

    SimConfig = namedtuple("SimConfig", "path label")

    base_config = SimConfig("/home/huziy/skynet3_rech1/offline_glk_output_daily_1979-2012", "ERAI-driven")
    modif_config = SimConfig("/home/huziy/skynet3_rech1/one_way_coupled_nemo_outputs_1979_1985", "CRCM5")

    nemo_manager_base = NemoYearlyFilesManager(folder=base_config.path, suffix="icemod.nc")
    nemo_manager_modif = NemoYearlyFilesManager(folder=modif_config.path, suffix="icemod.nc")

    icecov_base, icecov_ts_base = nemo_manager_base.get_max_yearly_ice_fraction(start_year=start_year,
                                                                                end_year=end_year)

    icecov_modif, icecov_ts_modif = nemo_manager_modif.get_max_yearly_ice_fraction(start_year=start_year,
                                                                                   end_year=end_year)


    lons, lats, bmp = nemo_manager_base.get_coords_and_basemap()
    xx, yy = bmp(lons.copy(), lats.copy())

    # Plot as usual: model, obs, model - obs
    img_folder = Path("nemo/{}vs{}".format(modif_config.label, base_config.label))
    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)
    img_file = img_folder.joinpath("compare_yearmax_icecov_{}_vs_{}_{}-{}.pdf".format(
        modif_config.label, base_config.label, start_year, end_year))


    fig = plt.figure()
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.05])

    cmap = cm.get_cmap("jet", 10)
    diff_cmap = cm.get_cmap("RdBu_r", 10)

    # base
    ax = fig.add_subplot(gs[0, 0])
    cs = bmp.contourf(xx, yy, icecov_base, cmap=cmap)
    bmp.drawcoastlines(ax=ax)
    ax.set_title(base_config.label)

    # modif
    ax = fig.add_subplot(gs[0, 1])
    cs = bmp.contourf(xx, yy, icecov_modif, cmap=cmap, levels=cs.levels)
    plt.colorbar(cs, cax=fig.add_subplot(gs[0, -1]))
    bmp.drawcoastlines(ax=ax)
    ax.set_title(modif_config.label)



    # difference
    ax = fig.add_subplot(gs[1, :])
    cs = bmp.contourf(xx, yy, icecov_modif - icecov_base, cmap=diff_cmap, levels=np.arange(-1, 1.2, 0.2))
    bmp.colorbar(cs, ax=ax)
    bmp.drawcoastlines(ax=ax)


    fig.tight_layout()
    fig.savefig(str(img_file), bbox_inches="tight")

    ax.set_title("{}-{}".format(modif_config.label, base_config.label))

    plt.close(fig)


    # Plot time series
    img_file = img_folder.joinpath("ts_compare_yearmax_icecov_{}_vs_{}_{}-{}.pdf".format(
        modif_config.label, base_config.label, start_year, end_year))
    fig = plt.figure()


    plt.plot(range(start_year, end_year + 1), icecov_ts_base, "b", lw=2, label=base_config.label)
    plt.plot(range(start_year, end_year + 1), icecov_ts_modif, "r", lw=2, label=modif_config.label)
    plt.legend()
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.grid()
    plt.xlabel("Year")

    fig.tight_layout()
    fig.savefig(str(img_file), bbox_inches="tight")







if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()
    main_compare_max_yearly_ice_conc()