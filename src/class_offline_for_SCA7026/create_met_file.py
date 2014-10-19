import os
from datetime import timedelta

from rpn.rpn import RPN
from scipy.spatial.ckdtree import cKDTree
from util.geo import lat_lon

import pandas as pd

__author__ = 'huziy'

import util_class_offline

def main():
    source_data_path = "/skynet1_rech3/camille/ERA_1958-2010"
    fname_pattern = "ERA-Interim_1979-2010_{}.rpn"
    #fname_pattern = "ERA_1958-2010_{}.rpn"

    varnames = [
        "TT", "UV", "SD", "AD", "HU", "P0", "PR"
    ]

    ignore_leap_year = True

    point_name = "YUMA_erai"
    dest_lat, dest_lon = 32.58, 245.61

    #point_name = "FLINT"
    #dest_lat, dest_lon = 43.27, 276.83

    #longitude should be in the range [-180, 180]
    dest_lon = dest_lon - 360 if dest_lon > 180 else dest_lon



    x0, y0, z0 = lat_lon.lon_lat_to_cartesian(dest_lon, dest_lat)

    start_year = 1980
    end_year = 1988
    out_step_min = 30  # interpolate in time if needed

    out_folder = "SCA7026"
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)


    #Determine the index of the closest point
    tt_path = os.path.join(source_data_path, fname_pattern.format("TT"))

    robj = RPN(tt_path)
    tt = robj.get_first_record_for_name("TT")
    lons, lats = robj.get_longitudes_and_latitudes_for_the_last_read_rec()

    robj.close()

    lons_1d, lats_1d = lons.flatten(), lats.flatten()
    x, y, z = lat_lon.lon_lat_to_cartesian(lons_1d, lats_1d)

    ktree = cKDTree(zip(x, y, z))
    dist, ind = ktree.query((x0, y0, z0))

    met_format = " {:>2}{:>3}{:>5}{:>6}" + 2 * "{:>9.2f}" + \
                 "{:>14.4E}{:>9.2f}{:>12.3E}{:>8.2f}{:>12.2f}" + "\n"


    shift_min = timedelta(minutes=int(dest_lon / 15.0 * 60))

    #write data to the text file
    with open(os.path.join(out_folder, "{}.MET".format(point_name)), "w") as f:
        df = pd.DataFrame()
        dates = None
        for vname in varnames:
            print "Reading {}".format(vname)
            r = RPN(os.path.join(source_data_path, fname_pattern.format(vname)))
            data = r.get_time_records_iterator_for_name_and_level(varname=vname)
            #select only dates from the range of interest and for the position of interest

            data = {k: v.flatten()[ind] for k, v in data if start_year <= k.year <= end_year
                and not (k.month == 2 and k.day == 29 and ignore_leap_year)  # Skip feb 29
            }

            if dates is None:
                dates = list(sorted(data.keys()))

            ts = pd.DataFrame(data=[data[k] for k in dates], index=dates, columns=[vname, ])



            ts = ts.asfreq(pd.DateOffset(minutes=out_step_min), method=None)
            if vname in ["AD", "SD", "PR"]:
                ts.values[ts.values < 0] = 0.0
                method = "nearest"
            else:
                method = "linear"

            ts = ts.interpolate(method=method)


            df = pd.concat([df, ts], axis=1)

            r.close()

        print "Finished reading data into memory"




        for d, row in df.iterrows():

            t_local = d + shift_min
            cos_zen = util_class_offline.get_cos_of_zenith_angle(dest_lat, t_local)
            #swrad = 0.0
            #if cos_zen > 0.0:
            swrad = row["SD"]

            line = met_format.format(
                d.hour, d.minute, d.timetuple().tm_yday, d.year,
                swrad, row["AD"], row["PR"], row["TT"], row["HU"], row["UV"],

                #Convert pressure to hPa => Pa
                row["P0"] * 100
            )
            f.write(line)

    print dist, ind


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()
    main()