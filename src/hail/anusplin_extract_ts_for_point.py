from pathlib import Path

from netCDF4 import Dataset, datetime
from rpn.domains import lat_lon
from scipy.spatial import KDTree
import numpy as np
import pandas as pd


def get_ym_from_path(fpath):

    if not isinstance(fpath, Path):
        fpath = Path(fpath)

    fields = fpath.stem.split("_")
    return int(fields[-2]), int(fields[-1])



def main(in_dir="/RESCUE/skynet3_rech1/huziy/anusplin_links", out_dir="/HOME/huziy/skynet3_rech1/hail/anusplin_ts"):

    out_dir_p = Path(out_dir)

    in_dir_p = Path(in_dir)

    lon0 = -114.0708
    lat0 = 51.0486


    vname = "daily_precipitation_accumulation"
    vname_alternatives = ["daily_accumulation_precipitation"]
    vname_alternatives.append(vname)

    var_list = [vname]
    fname_hint = "pcp"

    spatial_ind = None


    varname_to_list_of_frames = {vname: [] for vname in var_list}


    for fin in in_dir_p.iterdir():

        if fin.name.lower().endswith("ret"):
            continue

        if fin.name.lower().endswith("verif"):
            continue


        if fname_hint not in fin.name.lower():
            continue



        if not fin.name.endswith(".nc"):
            continue


        print(fin)

        year, month = get_ym_from_path(fin)
        with Dataset(str(fin)) as ds:



            if spatial_ind is None:

                lons, lats = ds.variables["lon"][:], ds.variables["lat"][:]

                x, y, z = lat_lon.lon_lat_to_cartesian(lons.flatten(), lats.flatten())

                ktree = KDTree(list(zip(x, y, z)))

                x0, y0, z0 = lat_lon.lon_lat_to_cartesian(lon0, lat0)

                dist, spatial_ind = ktree.query((x0, y0, z0))



            for vname_alt in vname_alternatives:
                try:
                    values = ds[vname_alt][:]
                    values = [field.flatten()[spatial_ind] for field in values]
                    break
                except IndexError as ierr:
                    pass


            dates = [datetime(year, month, int(d)) for d in ds["time"][:]]


            varname_to_list_of_frames[vname].append(pd.DataFrame(index=dates, data=values))




    for vname in var_list:
        df = pd.concat(varname_to_list_of_frames[vname])

        assert isinstance(df, pd.DataFrame)

        df.sort_index(inplace=True)

        df.to_csv(str(out_dir_p.joinpath("{}.csv".format(vname))), float_format="%.3f", index_label="Time")



if __name__ == '__main__':
    main()
