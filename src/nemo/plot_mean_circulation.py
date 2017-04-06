from collections import OrderedDict
from pathlib import Path

import xarray as xr
from mpl_toolkits.basemap import Basemap

from lake_effect_snow import common_params

import matplotlib.pyplot as plt


def main():
    dirpath = "/HOME/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/zdf_gls_dt_and_sbc_5min"
    dirpath = Path(dirpath)



    ds_u = xr.open_mfdataset(str(dirpath / "*_grid_U.nc"))
    ds_v = xr.open_mfdataset(str(dirpath / "*_grid_V.nc"))

    level = 0


    season_to_months = OrderedDict()


    uu = ds_u["vozocrtx"][:, level, :, :].mean(dim="time_counter")

    vv = ds_v["vomecrty"][:, level, :, :].mean(dim="time_counter")


    wndspd = (uu ** 2 + vv ** 2) ** 0.5

    uu1 = uu / wndspd
    vv1 = vv / wndspd


    b = Basemap(lon_0=180,
                    llcrnrlon=common_params.great_lakes_limits.lon_min,
                    llcrnrlat=common_params.great_lakes_limits.lat_min,
                    urcrnrlon=common_params.great_lakes_limits.lon_max,
                    urcrnrlat=common_params.great_lakes_limits.lat_max,
                    resolution="i")


    lons, lats = ds_u["nav_lon"][:].values, ds_v["nav_lat"][:].values

    xx, yy = b(lons, lats)


    im = b.pcolormesh(xx, yy, wndspd)
    b.colorbar(im)
    b.quiver(xx, yy, uu1, vv1)

    b.drawcoastlines()

    plt.savefig("nemo/circ_annual_mean.png", bbox_inches="tight", dpi=300)







if __name__ == '__main__':
    main()