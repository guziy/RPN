from mpl_toolkits.basemap import Basemap
from crcm5.model_data import Crcm5ModelDataManager
from domains.rotated_lat_lon import RotatedLatLon

__author__ = 'huziy'

import numpy as np


Grd_ni_name = "Grd_ni"
Grd_nj_name = "Grd_nj"
Grd_dx_name = "Grd_dx"
Grd_dy_name = "Grd_dy"
Grd_iref_name = "Grd_iref"
Grd_jref_name = "Grd_jref"
Grd_latr_name = "Grd_latr"
Grd_lonr_name = "Grd_lonr"
Grd_xlat1_name = "Grd_xlat1"
Grd_xlat2_name = "Grd_xlat2"
Grd_xlon1_name = "Grd_xlon1"
Grd_xlon2_name = "Grd_xlon2"

not_relevent_params = [
    "Grd_typ_S"
]

import matplotlib.pyplot as plt


def _parse_parameters(path):
    # parse config file
    result = {}
    with open(path) as f:
        lines = f.readlines()

        select_in_progress = False
        for line in lines:
            line = line.strip()

            # skip empty lines
            if line == "":
                continue

            if select_in_progress and line.endswith("/"):
                break

            # parse the values of the parameters
            if select_in_progress:
                fields = line.split(",")
                print(line)
                for the_field in fields:
                    # skip fields that do not contain "="
                    if "=" not in the_field:
                        continue

                    name, value = [f.strip() for f in the_field.split("=")]
                    if name in [Grd_ni_name, Grd_nj_name, Grd_iref_name, Grd_jref_name]:
                        result[name] = int(value)
                    elif name in not_relevent_params:
                        continue
                    else:
                        result[name] = float(value)

            if line.startswith("&grid"):
                select_in_progress = True

    return result


def main(path="/skynet3_rech1/huziy/gemclim_settings.nml"):
    params = _parse_parameters(path)
    print(params)

    ni, nj = 140, 140  # params[Grd_ni_name], params[Grd_nj_name]
    dx, dy = params[Grd_dx_name], params[Grd_dy_name]
    iRef, jRef = params[Grd_iref_name] - 1, params[Grd_jref_name] - 1
    lonRef, latRef = params[Grd_lonr_name], params[Grd_latr_name]

    lon1, lat1 = params[Grd_xlon1_name], params[Grd_xlat1_name]
    lon2, lat2 = params[Grd_xlon2_name], params[Grd_xlat2_name]

    lons_rot = np.arange(lonRef + (0 - iRef) * dx, lonRef + (ni - iRef) * dx, dx)
    lats_rot = np.arange(latRef + (0 - jRef) * dy, latRef + (nj - jRef) * dy, dy)

    lats_rot, lons_rot = np.meshgrid(lats_rot, lons_rot)
    print(lats_rot.shape)
    # lons_rot[lons_rot > 180] -= 360

    rll = RotatedLatLon(lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2)

    truepole_lonr, truepole_latr = rll.get_true_pole_coords_in_rotated_system()
    rotpole_lon, rotpole_lat = rll.get_north_pole_coords()

    llcrnrlon, llcrnrlat = rll.toGeographicLonLat(lons_rot[0, 0], lats_rot[0, 0])
    urcrnrlon, urcrnrlat = rll.toGeographicLonLat(lons_rot[-1, -1], lats_rot[-1, -1])

    b = Basemap(projection="rotpole", lon_0=truepole_lonr - 180, o_lat_p=rotpole_lat, o_lon_p=rotpole_lon,
                llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
    print(lons_rot[0, 0], lats_rot[0, 0], lons_rot[-1, -1], lats_rot[-1, -1])
    b.contourf(lons_rot, lats_rot, lons_rot)
    b.colorbar()

    b.drawcoastlines()
    # b.drawmeridians(np.arange(160, 200, 10))
    plt.show()


if __name__ == "__main__":
    main()
    print("Hello world")
  
