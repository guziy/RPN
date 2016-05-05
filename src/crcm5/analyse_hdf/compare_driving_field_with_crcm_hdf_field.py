import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
from rpn import level_kinds
from rpn.rpn_multi import MultiRPN

from application_properties import main_decorator
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis



def get_files_for_season(in_folder, start_year=-np.Inf, end_year=np.Inf, months=range(1, 13)):
    """
    Select the files corresponding to a season in a folder withing the [start_year, end_year] interval
    :param in_folder:
    :param start_year:
    :param end_year:
    :param months:
    :return:
    """
    flist_for_season = []

    for fn in os.listdir(in_folder):

        cmonth = int(fn[-2:])
        year = int(fn.split("_")[-1][:-2])

        if year < start_year or year > end_year:
            continue

        if cmonth not in months:
            continue

        flist_for_season.append(os.path.join(in_folder, fn))

    return flist_for_season


@main_decorator
def main():

    vname = "VV"
    start_year = 1980
    end_year = 2010

    crcm_data_path = "/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5"

    months_of_interest = [6, 7, 8]  # summer


    summer_crcm = analysis.get_seasonal_climatology(hdf_path=crcm_data_path, start_year=start_year, end_year=end_year,
                                                    level=0, var_name=vname, months=months_of_interest)

    bmp_info = analysis.get_basemap_info_from_hdf(file_path=crcm_data_path)


    erainterim_15_folder = "/RECH/data/Driving_data/Pilots/ERA-Interim_1.5/Pilots/"


    flist_for_season = get_files_for_season(erainterim_15_folder, start_year=start_year, end_year=end_year, months=months_of_interest)

    rpf = MultiRPN(flist_for_season)

    date_to_hu_erai15 = rpf.get_all_time_records_for_name_and_level(varname=vname, level=1000, level_kind=level_kinds.PRESSURE)
    summer_era15 = np.mean([field for field in date_to_hu_erai15.values()], axis=0)

    lons_era, lats_era = rpf.get_longitudes_and_latitudes_of_the_last_read_rec()

    # plotting

    # ERA-Interim
    plt.figure()
    b = Basemap(lon_0=180)
    xxg, yyg = b(lons_era, lats_era)
    im = b.contourf(xxg, yyg, summer_era15, 40, zorder=1)


    lonsr = bmp_info.lons.copy()
    lonsr[lonsr < 180] += 360
    xxr, yyr = b(lonsr, bmp_info.lats)
    b.contourf(xxr, yyr, summer_crcm, levels=im.levels, norm=im.norm, cmap=im.cmap, zorder=2)

    b.drawcoastlines(zorder=3)
    plt.colorbar(im)

    # CRCM (plot both crcm and era on the same plot)
    fig = plt.figure()
    xx, yy = bmp_info.get_proj_xy()

    margin = 20
    bext = bmp_info.basemap_for_extended_region(marginx=10 * margin, marginy=10 * margin)
    bmiddle = bmp_info.basemap_for_extended_region(marginx=9 * margin, marginy=9 * margin)

    xxg, yyg = bext(lons_era, lats_era)

    outer_domain = (xxg <= bext.urcrnrx) & (xxg >= bext.llcrnrx) & (yyg <= bext.urcrnry) & (yyg >= bext.llcrnry)



    summer_era15 = np.ma.masked_where(~outer_domain, summer_era15)

    im = bext.contourf(xx, yy, summer_crcm, levels=im.levels, norm=im.norm, cmap=im.cmap, zorder=2)
    bmiddle.contourf(xxg, yyg, summer_era15, levels=im.levels, norm=im.norm, cmap=im.cmap, zorder=1)
    bext.drawcoastlines()
    plt.colorbar(im)


    # Add a polygon
    ax = plt.gca()
    coords = np.array([
        [xx[0, 0], yy[0, 0]], [xx[0, -1], yy[0, -1]], [xx[-1, -1], yy[-1, -1]], [xx[-1, 0], yy[-1, 0]]
    ])
    ax.add_patch(Polygon(coords, facecolor="none", lw=3, zorder=3, edgecolor="k"))




    img_folder = "cc-paper-comments"
    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    fig.savefig(os.path.join(img_folder, "{}_era_1.5_and_crcm.png".format(vname)), bbox_inches="tight", transparent=True)

    plt.show()






if __name__ == '__main__':
    main()