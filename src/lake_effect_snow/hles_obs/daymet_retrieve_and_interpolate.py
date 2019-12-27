
"""
Retrieves only needed data from the official thredds server and interpolates it to the
model grid
"""
from datetime import datetime
from pathlib import Path
import xarray
from pyresample.kd_tree import KDTree
import numpy as np

from rpn.domains import lat_lon

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



from domains.grid_config import GridConfig, gridconfig_from_gemclim_settings_file


THREDDS_URL = "https://thredds.daac.ornl.gov/thredds/dodsC/ornldaac/1328/{year}/daymet_v3_{vname}_{year}_{domain}.nc4"


def main(vname="tmin",
         beg_time=None,
         end_time=None, target_grid_config: GridConfig = None,
         coarsen_before_interp_by=10):
    """
    :param vname: can be tmin, tmax, prcp
    """

    year_list = list(range(beg_time.year, end_time.year + 1))

    params = dict(
        vname=vname, domain="na"
    )

    url_list = []
    for y in year_list:
        params["year"] = y
        url_list.append(THREDDS_URL.format(**params))

    logger.debug(["retrieving ", ] + url_list)


    source_lons = None
    source_lats = None


    agg_lons = None
    agg_lats = None


    subset_list = []
    for url in url_list:
        with xarray.open_dataset(url) as ds:
            logger.debug(ds)

            if source_lons is None:
                source_lons = ds["lon"]
                source_lats = ds["lat"]

                target_lons, target_lats = target_grid_config.get_lons_and_lats_of_gridpoint_centers()

                xs, ys, zs = lat_lon.lon_lat_to_cartesian(source_lons.data.flatten(), source_lats.data.flatten())
                xt, yt, zt = lat_lon.lon_lat_to_cartesian(target_lons.flatten(), target_lats.flatten())

                logger.debug("Creating kdtree ...")
                kdtree = KDTree(np.array(list(zip(xs, ys, zs))))
                logger.debug("Created kdtree")

                dists, inds = kdtree.query(np.array(list(zip(xt, yt, zt)), dtype=np.float32),
                                           k=coarsen_before_interp_by ** 2)


                logger.debug(["inds.shape = ", inds.shape])

            # TODO: implement


def entry_GL01deg_452x260():
    gc = gridconfig_from_gemclim_settings_file(
        fpath=Path("config_bundle/GL_coupling_configs/Config_current/gemclim_settings.nml"))

    beg_time = datetime(1989, 1, 1)
    end_time = datetime(1989, 1, 1)

    vlist = ["tmin", ]
    for vn in vlist:
        main(vname=vn, beg_time=beg_time, end_time=end_time, target_grid_config=gc)


if __name__ == '__main__':
    import time
    t0 = time.clock()
    entry_GL01deg_452x260()
    logger.debug(f"Execution time {time.clock() - t0} seconds")