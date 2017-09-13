


# extract soundings for the variables
from pathlib import Path

from rpn import level_kinds
from rpn.rpn import RPN
from scipy.spatial import KDTree

from util.geo import lat_lon

import pandas as pd
import numpy as np




def get_multiplier(vname):

    if vname.lower() == "hu":
        return 1000
    elif vname.lower() == "pr":
        return 1000 * 24 * 3600  # convert to mm/day
    elif vname.lower() == "qc":
        return  1000 # convert to g/kg
    elif vname.lower() == "gz":
        return 10


    return 1


def main(in_dir="/HOME/data/Driving_data/Pilots/ERA-Interim_0.75/Pilots",
         out_dir="/HOME/huziy/skynet3_rech1/hail/soundings_from_erai", npoints=1,
         var_list=None):

    # var_list = ["TT", "HU", "UU", "VV"]
    # var_list = ["HU", ]

    if var_list is None:
        var_list = ["TT", "HU", "UU", "VV", "GZ"]

    out_dir_p = Path(out_dir)


    if not out_dir_p.exists():
        out_dir_p.mkdir(parents=True)


    in_dir_p = Path(in_dir)

    lon0 = -114.0708
    lat0 = 51.0486



    spatial_ind = None


    varname_to_list_of_frames = {vname: [] for vname in var_list}


    for fin in in_dir_p.iterdir():

        if fin.name.lower().endswith("ret"):
            continue

        if fin.name.lower().endswith("verif"):
            continue



        try:
            vname_to_data = {}


            with RPN(str(fin)) as r:

                assert isinstance(r, RPN)

                for vname in var_list:
                    vname_to_data[vname] = r.get_4d_field(name=vname, level_kind=level_kinds.PRESSURE)


                if spatial_ind is None:

                    lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()

                    x, y, z = lat_lon.lon_lat_to_cartesian(lons.flatten(), lats.flatten())

                    ktree = KDTree(list(zip(x, y, z)))

                    x0, y0, z0 = lat_lon.lon_lat_to_cartesian(lon0, lat0)

                    dist, spatial_ind = ktree.query((x0, y0, z0), k=npoints)



            for vname, data in vname_to_data.items():

                dates = sorted([d for d in data])
                vertical_levels = sorted([lev for lev in data[dates[0]]])

                values = np.array([[data[d][lev].flatten()[spatial_ind].mean() for lev in vertical_levels] for d in dates])

                print("levels={}".format(vertical_levels))
                print("date range={}...{}".format(dates[0], dates[-1]))
                print(values.shape)


                varname_to_list_of_frames[vname].append(pd.DataFrame(index=dates, columns=vertical_levels, data=values))

        except Exception as e:
            print("Warning: {}".format(e))



    for vname in var_list:
        df = pd.concat(varname_to_list_of_frames[vname])



        assert isinstance(df, pd.DataFrame)

        df = df * get_multiplier(vname)

        df.sort_index(inplace=True)



        df.to_csv(str(out_dir_p.joinpath("{}.csv".format(vname))), float_format="%.3f", index_label="Time")




if __name__ == '__main__':
    # main()
    # main(in_dir="/HOME/data/Driving_data/Offline/ERA-Interim_0.75/6h_Analysis.PR0")
    main(in_dir="/HOME/data/Driving_data/Offline/ERA-Interim_0.75/3h_Forecast",
         out_dir="/HOME/huziy/skynet3_rech1/hail/soundings_from_erai/3h",
         var_list=["PR"])

    main(out_dir="/HOME/huziy/skynet3_rech1/hail/soundings_from_erai/4points_mean",
         var_list=["TT", "HU", "UU", "VV", "GZ"], npoints=4)
