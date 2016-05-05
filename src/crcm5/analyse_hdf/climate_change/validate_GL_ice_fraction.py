from datetime import datetime, timedelta

from matplotlib.dates import DateFormatter
from matplotlib.gridspec import GridSpec

from application_properties import main_decorator
from crcm5.analyse_hdf.run_config import RunConfig
from util import plot_utils
from util.geo import mask_from_shp
import matplotlib.pyplot as plt
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis
from data import GL_obs_timeseries
import os
from crcm5.analyse_hdf import common_plot_params
import pandas as pd

OBS_DATA_FOLDER = "/HOME/huziy/skynet3_rech1/obs_data/Lake_ice_concentration_Great_lakes_timeseries"

# IDs of the lakes of interest
LAKE_IDS = ["HUR", "ERI", "ONT"]


LAKE_ID_TO_SHP_POLYGON_NAME = {
    "HUR": "Lake Huron",
    "ERI": "Lake Erie",
    "ONT": "Lake Ontario"
}

# GL coastlines (shp)
GL_COASTLINES_SHP = "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/data/shp/ne_10m__Great_Lakes/ne_10m_lakes.shp"

img_folder = "cc-paper-comments"


def get_lake_masks(lons2d, lats2d):
    """
    Get a mask for each lake in the LAKE_IDS list
    :param lons2d:
    :param lats2d:
    """

    res = {}

    for lid, lid_shp_name in LAKE_ID_TO_SHP_POLYGON_NAME.items():
        the_mask = mask_from_shp.get_mask(lons2d=lons2d, lats2d=lats2d, shp_path=GL_COASTLINES_SHP, polygon_name=lid_shp_name)
        res[lid] = the_mask

    return res


@main_decorator
def main():

    start_year = 1980
    end_year = 2003

    months_of_obs = [12, 1, 2, 3, 4, 5]

    r_config = RunConfig(
        data_path="/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5",
        start_year=start_year, end_year=end_year, label="ERAI-CRCM5-L"
    )

    var_name = "LC"
    bmp_info = analysis.get_basemap_info(r_config=r_config)
    lkid_to_mask = get_lake_masks(bmp_info.lons, bmp_info.lats)


    cell_area_m2 = analysis.get_array_from_file(path=r_config.data_path, var_name="cell_area_m2")


    # read the model data
    lkid_to_ts_model = {}
    for lkid, the_mask in lkid_to_mask.items():
        lkid_to_ts_model[lkid] = analysis.get_area_mean_timeseries(r_config.data_path, var_name=var_name, the_mask=the_mask * cell_area_m2,
                                                                   start_year=start_year, end_year=end_year)

        df = lkid_to_ts_model[lkid]

        # remove the last December
        df = df.select(lambda d: not (d.year == end_year and d.month == 12))

        # remove the first Jan and Feb
        df = df.select(lambda d: not (d.year == start_year and d.month in [1, 2]))

        # remove the Feb 29th
        df = df.select(lambda d: not (d.month == 2 and d.day == 29))

        # select months of interest
        df = df.select(lambda d: d.month in months_of_obs)

        # calculate the climatology
        df = df.groupby(lambda d: datetime(2001 if d.month == 12 else 2002, d.month, d.day)).mean()
        df.sort_index(inplace=True)


        lkid_to_ts_model[lkid] = df * 100


    # read obs data and calculate climatology
    lkid_to_ts_obs = {}
    for lkid in LAKE_IDS:
        lkid_to_ts_obs[lkid] = GL_obs_timeseries.get_ts_from_file(path=os.path.join(OBS_DATA_FOLDER, "{}-30x.TXT".format(lkid)),
                                                                  start_year=start_year, end_year=end_year - 1)

        # get the climatology
        dfm = lkid_to_ts_obs[lkid].mean(axis=1)

        dfm.index = [datetime(2001, 1, 1) + timedelta(days=int(jd - 1)) for jd in dfm.index]

        lkid_to_ts_obs[lkid] = dfm


    # plotting
    plot_utils.apply_plot_params(font_size=10)
    fig = plt.figure()
    gs = GridSpec(nrows=len(lkid_to_ts_model), ncols=2)

    for row, lkid in enumerate(lkid_to_ts_model):

        ax = fig.add_subplot(gs[row, 0])

        mod = lkid_to_ts_model[lkid]
        obs = lkid_to_ts_obs[lkid]

        print(obs.index)
        print(obs.values)

        ax.plot(mod.index, mod.values, label=r_config.label, color="r", lw=2)
        ax.plot(obs.index, obs.values, label="NOAA NIC/CIS", color="k", lw=2)

        if row == 0:
            ax.legend()

        ax.set_title(lkid)

        ax.xaxis.set_major_formatter(DateFormatter("%b"))


    fig.tight_layout()
    fig.savefig(os.path.join(img_folder, "GL_ice-cover-validation.png"), bbox_inches="tight", dpi=common_plot_params.FIG_SAVE_DPI)




if __name__ == '__main__':
    main()