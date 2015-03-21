from rpn.rpn import RPN
from scipy.spatial.ckdtree import cKDTree
from nemo.generate_grid import nemo_domain_properties
from util.geo import lat_lon

__author__ = 'huziy'
import os
import itertools as itt
from netCDF4 import Dataset, date2num
import numpy as np


def get_target_lon_lat_from_coordinates_file(
        path="~/skynet3_rech1/NEMO_fields_to_interpolated_210x130_0.1deg/coordinates_rotpole_nx210_ny130_dx0.1_dy0.1.nc"):

    path = os.path.expanduser(path)
    ds = Dataset(path)
    lons, lats = ds.variables["glamt"][:], ds.variables["gphit"][:]

    lons[lons < 0] += 360

    return lons, lats


def _year_from_file_path(path):
    return int(path.split("_")[-1].split(".")[0][:-2])


def create_yearly_from_rpn(in_folder, out_folder, varnames=None, multipliers=None,
                           units=None, offsets=None):

    if multipliers is None:
        multipliers = [1] * len(varnames)
    else:
        assert len(varnames) == len(multipliers)

    if offsets is None:
        offsets = [0] * len(varnames)
    else:
        assert len(varnames) == len(offsets)


    if units is None:
        units = ["unknown"] * len(varnames)
    else:
        assert len(varnames) == len(units)


    in_files = [os.path.join(in_folder, fname) for fname in os.listdir(in_folder) if not fname.endswith(".txt")]
    in_files = list(sorted(in_files))

    lonst, latst = get_target_lon_lat_from_coordinates_file()
    xt, yt, zt = lat_lon.lon_lat_to_cartesian(lonst.flatten(), latst.flatten())

    rll_target = nemo_domain_properties.known_domains["GLK_210x130_0.1deg"]
    basemap_params_dict = rll_target.get_basemap_params(llcrnrlon=lonst[0, 0], llcrnrlat=latst[0, 0],
                                                        urcrnrlon=lonst[-1, -1], urcrnrlat=latst[-1, -1])

    indices = None

    for year, paths in itt.groupby(in_files, key=lambda xtemp: _year_from_file_path(xtemp)):

        paths = list(paths)
        if len(paths) < 12:  # Consider only years with complete dataset
            continue

        print year

        # loop over all var names
        for varname, multiplier, var_units, offset in zip(varnames, multipliers, units, offsets):
            var_folder = os.path.join(out_folder, varname)

            if not os.path.isdir(var_folder):
                os.makedirs(var_folder)

            var_nc_path = os.path.join(var_folder, "{}_{}.nc".format(varname, year))
            if os.path.isfile(var_nc_path):
                print "{} -- already exists, remove in order to regenerate".format(var_nc_path)
                continue

            ds = Dataset(var_nc_path, "w", format="NETCDF3_CLASSIC")
            ds.createDimension("lat", lonst.shape[0])
            ds.createDimension("lon", lonst.shape[1])
            ds.createDimension("time")
            time_var = ds.createVariable("time", "i4", dimensions=("time", ))
            ncvar = ds.createVariable(varname, "f4", dimensions=("time", "lat", "lon"))
            basemap_params_var = ds.createVariable("basemap_params", "c")

            year_dates = []
            data = []
            for apath in paths:
                r = RPN(apath)
                data_dict = r.get_4d_field(name=varname)
                dkeys = sorted(data_dict.keys())
                year_dates.extend(dkeys)

                for k in dkeys:
                    if indices is None:
                        lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()
                        xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons.flatten(), lats.flatten())
                        tree = cKDTree(zip(xs, ys, zs))
                        d, indices = tree.query(zip(xt, yt, zt))

                    data.append(data_dict[k].items()[0][1].flatten()[indices].reshape(lonst.shape))

                r.close()

            time_var.units = "hours since {:%Y-%m-%d %H:%M:%S}".format(year_dates[0])
            ncvar.units = var_units
            time_var[:] = date2num(year_dates, time_var.units)
            ncvar[:] = np.asarray(data) * multiplier + offset

            # assert isinstance(basemap_params_var, Variable)
            basemap_params_var.setncatts(basemap_params_dict)

            ds.close()



def main():
    out_folder = os.path.expanduser("~/skynet3_rech1/ERA-Interim_0.75_NEMO_pilot")

    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # file names are in the following form: ERA_Interim_0.75d_6h_analysis_199010
    path_to_rpn_files = "/RECH/data/Driving_data/Offline/ERA-Interim_0.75/6h_Analysis"

    varnames = ["PR", "TT", "HU", "UU", "VV", "N4", "AD", "SN"]
    offsets = [0, 273.15, 0, 0, 0, 0, 0, 0]
    units = ["mm/s", "K", "kg/kg", "m/s", "m/s", "W/m**2", "W/m**2", "mm/s"]

    mpers_per_knot = 0.514444444
    multipliers = [1.0e3, 1., 1., mpers_per_knot, mpers_per_knot, 1.0, 1.0, 1.0e3]
    create_yearly_from_rpn(path_to_rpn_files, out_folder, varnames=varnames,
                           multipliers=multipliers, units=units, offsets=offsets)

    # Interpolate snowfall
    # varnames = ["SN", ]
    # units = ["mm/s", ]
    # multipliers = [1.0e3, ]
    # path_to_rpn_files = "/home/huziy/skynet3_rech1/ERAI075_snowfall_rpn/6h"
    # create_yearly_from_rpn(path_to_rpn_files, out_folder, varnames=varnames,
    #                        multipliers=multipliers, units=units)


    pass


if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()
    main()