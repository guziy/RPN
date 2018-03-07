

# Interpolate data to selected points for Craig
from collections import defaultdict, OrderedDict
from pathlib import Path

from rpn.domains.rotated_lat_lon import RotatedLatLon
from rpn.rpn import RPN
from rpn.rpn_multi import MultiRPN
import pandas as pd
from scipy.spatial import KDTree

from application_properties import main_decorator
import itertools as itt

from util.geo import lat_lon
import numpy as np


def get_points_of_interest(path: str = "data/NEI/selected_points.txt") -> pd.DataFrame:
    pts = pd.read_csv(path, sep="\s+")

    # west longitudes, convert to negative
    pts.iloc[:, -1] *= -1

    return pts


def save_data_to_csv_files(out_dir="data/NEI/crcm5_hostetler", pts_to_vn_to_vals=None, dates=None):
    out_dir_p = Path(out_dir)

    if not out_dir_p.is_dir():
        out_dir_p.mkdir()


    for pt, vn_to_vals in pts_to_vn_to_vals.items():
        df = pd.DataFrame(data=vn_to_vals, index=dates)
        df.to_csv(str(out_dir_p.joinpath("{}.csv".format(pt))), index_label="Date", float_format="%10.2f")


@main_decorator
def main():

    # in_folder = "/RECH2/huziy/coupling/GL_440x260_0.1deg_GL_with_Hostetler/Samples_selected"
    # in_folder = "/b4_fs1/huziy/from_guillimin_important_sim_results/quebec_0.1_crcm5-hcd-rl-cc/Samples"

    # in_folder = "/HOME/huziy/skynet3_rech1/CRCM5_outputs/coupled-GL-NEMO1h/selected_fields"

    in_folder = "/snow3/huziy/NEI/GL/GL_CC_CanESM2_RCP85/coupled-GL-future_CanESM2/Samples"
    in_folder_p = Path(in_folder)


    out_dir = Path("data/NEI/crcm5_glnemo/canesm2_driven_additional_2018/" + in_folder_p.parent.name)
    # create the output directory
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)


    # vars_of_interest = ["TT", "HR", "P0", "PN", "UU", "VV", "PR", "N4", "AD"]
    vars_of_interest = ["TT", "HR", "UU", "VV", "PR", "N4", "AD", "GZ"]


    varname_to_fname_prefix = {
       "TT": "dm",
       "HR": "dm",
       "UU": "dm",
       "VV": "dm",
       "PR": "pm",
       "N4": "pm",
       "AD": "pm",
       "GZ": "dm"
    }

    # varname_to_fname_prefix = {
    #      k: None for k in vars_of_interest
    # }


    vname_to_multiplier = defaultdict(lambda: 1)

    mps_per_knot = 0.514444
    vname_to_multiplier["UU"] = mps_per_knot
    vname_to_multiplier["VV"] = mps_per_knot
    vname_to_multiplier["PR"] = 3600 * 24 * 1000.0  # M/s -> mm/day


    # for testing
    # vars_of_interest = vars_of_interest[0:2]

    points = get_points_of_interest(
        path="data/NEI/selected_points_201801.csv"
    )
    print(points.head(15))



    flist = itt.chain(*[[f for f in monthdir.iterdir() if not f.is_dir()] for monthdir in in_folder_p.iterdir() if monthdir.is_dir()])
    flist = list(flist)



    for f in flist:
        assert f.exists()
        assert not f.is_dir()

        print(f)


    pts_to_vname_to_vals = {}
    pts_to_lon = {}
    pts_to_lat = {}

    # for each point there is a dict of vname => value_list
    # as well as coordinate mappings
    for row, pt in enumerate(points.iloc[:, 0]):

        pts_to_vname_to_vals[pt] = defaultdict(OrderedDict)
        for v in vars_of_interest:
            # pts_to_vname_to_vals[pt][v] = {}
            pts_to_lon[pt] = points.iloc[row, -1]
            pts_to_lat[pt] = points.iloc[row, -2]



    # model indices corresponding to each point
    pts_to_model_indices = {}


    # Get the path to the file with coordinates
    coord_file = in_folder_p.parent.joinpath("pm1979010100_00000000p")
    print(coord_file.exists())
    print(flist)
    if not coord_file.exists():
        for f in flist:
            if f.name.startswith("pm"):
                coord_file = f
                print("Using {} as coord file".format(f))

                break

        if not coord_file.exists():
            coord_file = [f for f in flist if ("PR" in f.name or "TT" in f.name)][0]




    # get the corresponding gridpoints for selected positions
    r = RPN(str(coord_file))

    vlist = r.get_list_of_varnames()

    if "PR" in vlist:
        coord_vname = "PR"
    elif "TT" in vlist:
        coord_vname = "TT"
    elif len([v for v in vlist if v not in [">>", "^^", "HY"]]) > 0:
        coord_vname = [v for v in vlist if v not in [">>", "^^", "HY"]][0]
    else:
        coord_vname = "ML"


    print(coord_vname)

    pr = r.get_first_record_for_name(coord_vname)
    lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()
    x, y, z = lat_lon.lon_lat_to_cartesian(lons.flatten(), lats.flatten())
    ktree = KDTree(list(zip(x, y, z)))
    r.close()


    for pt in pts_to_vname_to_vals:
        x0, y0, z0 = lat_lon.lon_lat_to_cartesian(pts_to_lon[pt], pts_to_lat[pt])
        dist, ind = ktree.query((x0, y0, z0))
        pts_to_model_indices[pt] = ind
        if coord_vname == "ML":
            print("{}: lkfr={} in the model, dist={}".format(pt, pr.flatten()[ind], dist))
        else:
            print("{}: dist={}".format(pt, pr.flatten()[ind], dist))

    # for each variable
    for vi, vn in enumerate(vars_of_interest):
        print("Processing {}".format(vn))

        fname_prefix = None
        if vn in varname_to_fname_prefix:
            fname_prefix = varname_to_fname_prefix[vn]

        # select data for each point
        for fp in flist:
            # skip files which are not supposed to contain the variable
            if fname_prefix is not None:
                if not fp.name.startswith(fname_prefix):
                    continue


            with RPN(str(fp)) as r:
                vnames_in_file = list(r.get_list_of_varnames())

                if vn not in vnames_in_file:
                    continue

                print("reading {} ...".format(fp))
                data = r.get_4d_field(vn)

                for pt in pts_to_vname_to_vals:

                    for d, field in data.items():
                        pts_to_vname_to_vals[pt][vn][d] = list(field.items())[0][1].flatten()[pts_to_model_indices[pt]] * vname_to_multiplier[vn]




    # initialize sorted dates
    dates = None
    for pt, vname_to_datevals in pts_to_vname_to_vals.items():

        if dates is not None:
            break

        for vname, datevals in vname_to_datevals.items():
            dates = list(sorted(datevals))
            break

    for pt, vname_to_datevals in pts_to_vname_to_vals.items():
        for vn in vname_to_datevals:

            vname_to_datevals[vn] = [vname_to_datevals[vn][d] for d in dates]

    # write data in a csv file per point
    # rows -  {date => value}
    # columns - variables
    # save_data_to_csv_files(pts_to_vn_to_vals=pts_to_vname_to_vals, dates=dates, out_dir="data/NEI/crcm5_hostetler/cc_canesm2_rcp85_additional")
    save_data_to_csv_files(pts_to_vn_to_vals=pts_to_vname_to_vals, dates=dates, out_dir=str(out_dir))

if __name__ == '__main__':
    main()
