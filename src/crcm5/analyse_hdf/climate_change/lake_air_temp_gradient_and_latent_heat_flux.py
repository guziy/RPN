from collections import OrderedDict

from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec

from application_properties import main_decorator
from crcm5 import infovar
from crcm5.analyse_hdf.run_config import RunConfig
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis
from util import plot_utils

import matplotlib.pyplot as plt
import os
import numpy as np

img_folder = "cc-paper-comments"

@main_decorator
def main():
    lkfr_limit = 0.05
    model_data_current_path = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/" \
                         "quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"


    modif_label = "CanESM2-CRCM5-L"

    start_year_c = 1980
    end_year_c = 2010

    future_shift_years = 90

    params = dict(
        start_year=start_year_c, end_year=end_year_c
    )

    params.update(
        dict(data_path=model_data_current_path, label=modif_label)
    )

    model_config_c = RunConfig(**params)
    model_config_f = model_config_c.get_shifted_config(future_shift_years)



    bmp_info = analysis.get_basemap_info(r_config=model_config_c)


    specific_cond_heat = 0.250100e7  # J/kg
    water_density = 1000.0  # kg/m**3

    season_to_months = OrderedDict([
        ("Summer", [6, 7, 8]),
    ])

    lkfr = analysis.get_array_from_file(path="/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5", var_name=infovar.HDF_LAKE_FRACTION_NAME)

    assert lkfr is not None, "Could not find lake fraction in the file"

    # Current climate
    traf_c = analysis.get_seasonal_climatology_for_runconfig(run_config=model_config_c, varname="TRAF", level=5, season_to_months=season_to_months)
    pr_c = analysis.get_seasonal_climatology_for_runconfig(run_config=model_config_c, varname="PR", level=0, season_to_months=season_to_months)

    lktemp_c = analysis.get_seasonal_climatology_for_runconfig(run_config=model_config_c, varname="L1", level=0, season_to_months=season_to_months)
    airtemp_c = analysis.get_seasonal_climatology_for_runconfig(run_config=model_config_c, varname="TT", level=0, season_to_months=season_to_months)

    lhc = OrderedDict([
        (s, specific_cond_heat * (pr_c[s] * water_density - traf_c[s])) for s, traf in traf_c.items()
    ])



    avc = analysis.get_seasonal_climatology_for_runconfig(run_config=model_config_c, varname="AV", level=0, season_to_months=season_to_months)


    plt.figure()
    lhc["Summer"] = np.ma.masked_where(lkfr < lkfr_limit, lhc["Summer"])
    print("min: {}, max: {}".format(lhc["Summer"].min(), lhc["Summer"].max()))
    cs = plt.contourf(lhc["Summer"].T)
    plt.title("lhc")
    plt.colorbar()

    plt.figure()
    cs = plt.contourf(avc["Summer"].T, levels=cs.levels, norm=cs.norm, cmap=cs.cmap)
    plt.title("avc")
    plt.colorbar()

    # Future climate
    traf_f = analysis.get_seasonal_climatology_for_runconfig(run_config=model_config_f, varname="TRAF", level=5, season_to_months=season_to_months)
    pr_f = analysis.get_seasonal_climatology_for_runconfig(run_config=model_config_f, varname="PR", level=0,
                                                           season_to_months=season_to_months)

    lktemp_f = analysis.get_seasonal_climatology_for_runconfig(run_config=model_config_f, varname="L1", level=0,
                                                             season_to_months=season_to_months)
    airtemp_f = analysis.get_seasonal_climatology_for_runconfig(run_config=model_config_f, varname="TT", level=0,
                                                              season_to_months=season_to_months)

    lhf = OrderedDict([
        (s, specific_cond_heat * (pr_f[s] * water_density - traf_f[s])) for s, traf in traf_f.items()
    ])

    plt.figure()
    plt.pcolormesh(traf_c["Summer"].T)
    plt.title("TRAF over lakes current")
    plt.colorbar()



    avf = analysis.get_seasonal_climatology_for_runconfig(run_config=model_config_f, varname="AV", level=0,
                                                          season_to_months=season_to_months)

    plt.figure()
    cs = plt.contourf(avf["Summer"].T)
    plt.title("avf")
    plt.colorbar()


    plt.figure()
    cs = plt.contourf(avf["Summer"].T - avc["Summer"].T, levels=np.arange(-40, 45, 5))
    plt.title("d(av)")
    plt.colorbar()


    plt.figure()
    plt.contourf(lhf["Summer"].T - lhc["Summer"].T, levels=cs.levels, cmap=cs.cmap, norm=cs.norm)
    plt.title("d(lh)")
    plt.colorbar()



    # plotting
    plot_utils.apply_plot_params(width_cm=15, height_cm=15, font_size=10)
    gs = GridSpec(2, 2)




    # tair_c_ts = analysis.get_area_mean_timeseries(model_config_c.data_path, var_name="TT", level_index=0,
    #                                   start_year=model_config_c.start_year, end_year=model_config_c.end_year,
    #                                   the_mask=lkfr >= lkfr_limit)
    #
    # tair_f_ts = analysis.get_area_mean_timeseries(model_config_f.data_path, var_name="TT", level_index=0,
    #                                   start_year=model_config_f.start_year, end_year=model_config_f.end_year,
    #                                   the_mask=lkfr >= lkfr_limit)
    #
    #
    # tlake_c_ts = analysis.get_area_mean_timeseries(model_config_c.data_path, var_name="TT", level_index=0,
    #                                   start_year=model_config_c.start_year, end_year=model_config_c.end_year,
    #                                   the_mask=lkfr >= lkfr_limit)
    #
    # tlake_f_ts = analysis.get_area_mean_timeseries(model_config_f.data_path, var_name="TT", level_index=0,
    #                                   start_year=model_config_f.start_year, end_year=model_config_f.end_year,
    #                                   the_mask=lkfr >= lkfr_limit)






    for season in season_to_months:
        fig = plt.figure()


        lktemp_c[season] -= 273.15
        dT_c = np.ma.masked_where(lkfr < lkfr_limit, lktemp_c[season] - airtemp_c[season])


        lktemp_f[season] -= 273.15
        dT_f = np.ma.masked_where(lkfr < lkfr_limit, lktemp_f[season] - airtemp_f[season])

        d = np.round(max(np.ma.abs(dT_c).max(), np.ma.abs(dT_f).max()))

        vmin = -d
        vmax = d

        clevs = np.arange(-12, 13, 1)
        ncolors = len(clevs) - 1
        bn = BoundaryNorm(clevs, ncolors=ncolors)
        cmap = cm.get_cmap("seismic", ncolors)




        ax_list = []

        fig.suptitle(season)

        xx, yy = bmp_info.get_proj_xy()

        # Current gradient
        ax = fig.add_subplot(gs[0, 0])
        ax.set_title(r"current: $T_{\rm lake} - T_{\rm atm}$")
        cs = bmp_info.basemap.pcolormesh(xx, yy, dT_c, ax=ax, norm=bn, cmap=cmap)
        bmp_info.basemap.colorbar(cs, ax=ax, extend="both")
        ax_list.append(ax)



        # Future Gradient
        ax = fig.add_subplot(gs[0, 1])
        ax.set_title(r"future: $T_{\rm lake} - T_{\rm atm}$")
        cs = bmp_info.basemap.pcolormesh(xx, yy, dT_f, ax=ax, norm=cs.norm, cmap=cs.cmap, vmin=vmin, vmax=vmax)
        bmp_info.basemap.colorbar(cs, ax=ax, extend="both")
        ax_list.append(ax)


        # Change in the gradient
        ax = fig.add_subplot(gs[1, 0])
        ax.set_title(r"$\Delta T_{\rm future} - \Delta T_{\rm current}$")

        ddT = dT_f - dT_c
        d = np.round(np.ma.abs(ddT).max())
        clevs = np.arange(-3, 3.1, 0.1)
        ncolors = len(clevs) - 1
        bn = BoundaryNorm(clevs, ncolors=ncolors)
        cmap = cm.get_cmap("seismic", ncolors)
        cs = bmp_info.basemap.pcolormesh(xx, yy, ddT, norm=bn, cmap=cmap)
        bmp_info.basemap.colorbar(cs, ax=ax, extend="both")
        ax_list.append(ax)



        # Change in the latent heat flux
        # ax = fig.add_subplot(gs[1, 1])
        # ax.set_title(r"$LE_{\rm future} - LE_{\rm current}$")
        # dlh = np.ma.masked_where(lkfr < lkfr_limit, lhf[season] - lhc[season])
        #
        # d = np.round(np.ma.abs(dlh).max() // 10) * 10
        # clevs = np.arange(0, 105, 5)
        # bn = BoundaryNorm(clevs, ncolors=ncolors)
        # cmap = cm.get_cmap("jet", ncolors)
        #
        # cs = bmp_info.basemap.pcolormesh(xx, yy, dlh, norm=bn, cmap=cmap)
        # bmp_info.basemap.colorbar(cs, ax=ax, extend="max")  # Change in the latent heat flux
        # ax_list.append(ax)


        for the_ax in ax_list:
            bmp_info.basemap.drawcoastlines(linewidth=0.3, ax=the_ax)


        fig.tight_layout()
        fig.savefig(os.path.join(img_folder, "lake_atm_gradients_and_fluxes_{}-{}_{}-{}.png".format(model_config_f.start_year, model_config_f.end_year, start_year_c, end_year_c)),
                    dpi=800,
                    bbox_inches="tight")







if __name__ == '__main__':
    main()