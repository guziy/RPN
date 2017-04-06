from collections import OrderedDict
from pathlib import Path
from matplotlib.gridspec import GridSpec
from util import plot_utils

__author__ = 'huziy'

from nemo.nemo_yearly_files_manager import NemoYearlyFilesManager
import matplotlib.pyplot as plt

import cartopy.feature as cfeature
import numpy as np


def main():
    # path_to_folder = "/home/huziy/skynet3_rech1/offline_glk_output_daily_1979-2012"

    path_to_folder = "/home/huziy/skynet3_rech1/one_way_coupled_nemo_outputs_1979_1985"

    season_to_months = OrderedDict([
    #    ("Winter", (12, 1, 2)),
    #    ("Spring", (3, 4, 5)),
    #    ("Summer", (6, 7, 8)),
        ("Fall", (9, 10, 11))
    ])

    start_year = 1990
    end_year = 2010

    u_manager = NemoYearlyFilesManager(folder=path_to_folder, suffix="_U.nc")
    v_manager = NemoYearlyFilesManager(folder=path_to_folder, suffix="_V.nc")

    u_clim = u_manager.get_seasonal_clim_field(start_year=start_year, end_year=end_year,
                                               season_to_months=season_to_months, varname="vozocrtx")
    v_clim = v_manager.get_seasonal_clim_field(start_year=start_year, end_year=end_year,
                                               season_to_months=season_to_months, varname="vomecrty")

    lons, lats, bmp = u_manager.get_coords_and_basemap()
    lons, lats, rpole = u_manager.get_cartopy_proj_and_coords()

    img_folder = Path("nemo/circulation")
    if not img_folder.is_dir():
        img_folder.mkdir()

    fig = plt.figure()

    gs = GridSpec(len(season_to_months), 1, width_ratios=[1, 0.05])

    xx, yy = bmp(lons.copy(), lats.copy())
    lons_1d = lons.mean(axis=0)
    lats_1d = lats.mean(axis=1)

    import cartopy.crs as ccrs

    tr_xy = rpole.transform_points(ccrs.Geodetic(), lons, lats)

    xx = tr_xy[:, :, 0]
    yy = tr_xy[:, :, 1]

    for i, season in enumerate(season_to_months):
        ax = fig.add_subplot(gs[i, 0], projection=rpole)
        u = u_clim[season]
        v = v_clim[season]
        # im = bmp.contourf(xx, yy, (u ** 2 + v ** 2) ** 0.5)

        # bmp.colorbar(im)

        # bmp.quiver(xx, yy, u, v, ax=ax)

        # uproj, vproj, xx, yy = bmp.transform_vector(u, v, lons_1d, lats_1d, lons.shape[0], lons.shape[1],
        # returnxy=True, masked=True)

        # uproj, vproj = bmp.rotate_vector(u, v, lons.copy(), lats.copy())


        c = (u ** 2 + v ** 2) ** 0.5 * 100

        print(c.shape, u.shape, v.shape)

        # Convert to cm
        u *= 100
        v *= 100

        # u = np.ma.masked_where(u > 10, u)
        # v = np.ma.masked_where(v > 10, v)


        step = 2
        q = ax.quiver(xx[::step, ::step], yy[::step, ::step], u[::step, ::step], v[::step, ::step], width=0.001,
                      pivot="middle", headwidth=5, headlength=5, scale=100)

        qk = ax.quiverkey(q, 0.1, 0.1, 10, r'$10 \frac{cm}{s}$', fontproperties={'weight': 'bold'}, linewidth=1)

        # ax.streamplot(xx, yy, u, v, linewidth=2, density=5, color="k")
        # im = ax.pcolormesh(xx, yy, c)
        # plt.colorbar(im, ax=ax)
        ax.set_extent([xx[0, 0], xx[-1, -1], yy[0, 0], yy[-1, -1]], crs=rpole)
        ax.coastlines("10m")
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', '50m',
                                                    edgecolor='k',
                                                    facecolor="none"), linewidth=0.5)

        ax.add_feature(cfeature.RIVERS, edgecolor="k", facecolor="none", linewidth=0.5)

        ax.grid()
        ax.set_title(season)

    fig.savefig(str(img_folder.joinpath(
        "NEMO-CRCM5-seasonal_surf_circulation_{}_arrows.pdf".format("_".join(season_to_months.keys())))),
                bbox_inches="tight")


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()
    plot_utils.apply_plot_params(font_size=12, width_pt=None, width_cm=20)
    main()