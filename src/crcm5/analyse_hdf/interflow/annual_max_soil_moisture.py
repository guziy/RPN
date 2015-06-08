from pathlib import Path
from rpn.rpn import RPN
from crcm5.analyse_hdf.run_config import RunConfig
from util import plot_utils
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis
import matplotlib.pyplot as plt
import numpy as np

__author__ = 'huziy'

GEO_DATA_FILE = "/skynet3_rech1/huziy/hdf_store/pm1979010100_00000000p"
IMG_FOLDER = Path("impact_of_interflow/annual_max_soil_moisture")


def main():
    import application_properties
    application_properties.set_current_directory()
    if not IMG_FOLDER.exists():
        IMG_FOLDER.mkdir(parents=True)

    plot_utils.apply_plot_params(font_size=14, width_pt=None, width_cm=20, height_cm=20)
    start_year = 1980
    end_year = 2010

    varname = "TDRA"

    base_config = RunConfig(start_year=start_year, end_year=end_year,
                            data_path="/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5",
                            label="NI")

    modif_config = RunConfig(start_year=start_year, end_year=end_year,
                             data_path="/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5",
                             label="WI")


    r_obj = RPN(GEO_DATA_FILE)
    facc = r_obj.get_first_record_for_name("FAA")
    fldr = r_obj.get_first_record_for_name("FLDR")
    mg = r_obj.get_first_record_for_name("MG")

    lons, lats, bmp = analysis.get_basemap_from_hdf(file_path=base_config.data_path)

    # Calculate the daily mean fields
    dates, daily_clim_base = analysis.get_daily_climatology_for_rconf(base_config, var_name=varname, level=0)
    _, daily_clim_modif = analysis.get_daily_climatology_for_rconf(modif_config, var_name=varname, level=0)

    _, pr_base = analysis.get_daily_climatology_for_rconf(base_config, var_name="PR", level=0)
    _, pr_modif = analysis.get_daily_climatology_for_rconf(modif_config, var_name="PR", level=0)

    _, av_base = analysis.get_daily_climatology_for_rconf(base_config, var_name="AV", level=0)
    _, av_modif = analysis.get_daily_climatology_for_rconf(modif_config, var_name="AV", level=0)


    _, intf_rates = analysis.get_daily_climatology_for_rconf(modif_config, var_name="INTF", level=0)

    # Plot the difference
    fig = plt.figure()
    xx, yy = bmp(lons, lats)

    # daily_clim_base = np.array([f for d, f in zip(dates, daily_clim_base) if d.month not in range(3, 12)])
    # daily_clim_modif = np.array([f for d, f in zip(dates, daily_clim_modif) if d.month not in range(3, 12)])

    mean_base = np.mean(daily_clim_base, axis=0)
    diff = (np.mean(daily_clim_modif, axis=0) - mean_base) * 24 * 3600
    dpr = (pr_modif.sum(axis=0) - pr_base.sum(axis=0)) / pr_base.sum(axis=0)
    dav = (av_modif.sum(axis=0) - av_base.sum(axis=0)) / av_base.sum(axis=0)
    diff = np.ma.masked_where((mg <= 1.0e-3) | (dpr < 0) | (dav > 0) | (diff > 0), diff)

    print("{}-ranges: {}, {}".format(varname, daily_clim_base.min(), daily_clim_base.max()))
    print("{}-ranges: {}, {}".format(varname, daily_clim_modif.min(), daily_clim_modif.max()))

    limit_base = np.percentile(daily_clim_base, 90, axis=0)
    limit_modif = np.percentile(daily_clim_modif, 50, axis=0)
    limit_lower_intf = 1.0e-4 / (24.0 * 60.0 * 60.0)


    ndays_base = daily_clim_base > limit_base[np.newaxis, :, :]
    ndays_base = ndays_base.sum(axis=0)

    ndays_modif = daily_clim_modif > limit_base[np.newaxis, :, :]
    ndays_modif = ndays_modif.sum(axis=0)

    diff_days = np.ma.masked_where((mg <= 1.0e-4) | (intf_rates.max() < limit_lower_intf),
                                   ndays_modif - ndays_base)

    diff_days = np.ma.masked_where(diff_days > 0, diff_days)

    print("diff_days ranges: {} to {}".format(diff_days.min(), diff_days.max()))


    im = bmp.pcolormesh(xx, yy, diff)
    bmp.colorbar(im)
    bmp.drawcoastlines()




    img_file = IMG_FOLDER.joinpath("{}_{}-{}.png".format(varname, start_year, end_year))
    with img_file.open("wb") as imf:
        fig.savefig(imf, bbox_inches="tight")

    plt.show()


if __name__ == '__main__':
    main()