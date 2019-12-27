from pathlib import Path

from rpn import level_kinds
from rpn.rpn import RPN
from scipy.spatial import KDTree

from domains import grid_config

from netCDF4 import Dataset, date2num
from datetime import datetime

from domains.grid_config import GridConfig
from util.geo import lat_lon
import xarray
import numpy as np



def file_sort_key(fobj):

    """
    For sorting of the files by their names containing year and month at the end
    :param fobj:
    :return:
    """
    tok = fobj.name.split("_")[-1]
    try:
        d = datetime.strptime(tok, "%Y%m%d%H%M")
    except Exception as e:
        d = datetime.strptime(tok, "%Y%m")
        d = d.replace(day=15)

    return d


def interpolate_wind_components_from_rpn_files(data_dir: Path = "", out_dir: Path = "", target_grid_config=None,
                                               wind_level=1., wind_level_kind=level_kinds.HYBRID):
    """
    Interpolate wind component fields and save to a netcdf file
    :param data_dir:
    :param out_dir:
    :param target_grid_config:
    """

    # sort files to be in the chronological order
    files_sorted = list(sorted((mfile for mfile in data_dir.iterdir()), key=file_sort_key))

    out_file_name = "erai0.75_interpolated_uu_vv_knots.nc"

    n_records_written = 0

    uu_var = None
    vv_var = None
    time_var = None

    indices_in_source_field = None
    lon_t = None

    with Dataset(out_dir.joinpath(out_file_name), "w") as ds:

        assert isinstance(ds, Dataset)

        for in_file in files_sorted:

            print("Processing {}".format(in_file))

            with RPN(str(in_file)) as r:
                assert isinstance(r, RPN)
                uu = r.get_all_time_records_for_name_and_level("UU", level=wind_level, level_kind=wind_level_kind)
                vv = r.get_all_time_records_for_name_and_level("VV", level=wind_level, level_kind=wind_level_kind)

                # create dimensions, initialize variables and coordiates
                if uu_var is None:
                    lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()

                    xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons.flatten(), lats.flatten())
                    ktree = KDTree(list(zip(xs, ys, zs)))

                    #
                    lon_t, lat_t = target_grid_config.get_lons_and_lats_of_gridpoint_centers()
                    xt, yt, zt = lat_lon.lon_lat_to_cartesian(lon_t.flatten(), lat_t.flatten())

                    # nearest neighbour interpolation
                    dists, indices_in_source_field = ktree.query(list(zip(xt, yt, zt)), k=1)

                    ds.createDimension("time")
                    ds.createDimension("x", lon_t.shape[0])
                    ds.createDimension("y", lon_t.shape[1])

                    lon_var = ds.createVariable("lon", "f4", dimensions=("x", "y"))
                    lat_var = ds.createVariable("lat", "f4", dimensions=("x", "y"))

                    lon_var[:] = lon_t
                    lat_var[:] = lat_t

                    start_date = file_sort_key(in_file)

                    time_var = ds.createVariable("time", "i4", dimensions=("time",))
                    time_var.units = "hours since {:%Y-%m-%d %H:%M:%S}".format(start_date)

                    uu_var = ds.createVariable("UU", "f4", dimensions=("time", "x", "y"),
                                               zlib=True,
                                               least_significant_digit=3)
                    uu_var.units = "knots"
                    uu_var.coordinates = "lon lat"

                    vv_var = ds.createVariable("VV", "f4",
                                               dimensions=("time", "x", "y"),
                                               zlib=True,
                                               least_significant_digit=3)
                    vv_var.units = "knots"
                    vv_var.coordinates = "lon lat"

                t_vals = list(sorted(uu))

                uu_vals = [uu[t].flatten()[indices_in_source_field].reshape(lon_t.shape) for t in t_vals]
                vv_vals = [vv[t].flatten()[indices_in_source_field].reshape(lon_t.shape) for t in t_vals]

                uu_var[n_records_written:, :, :] = uu_vals
                vv_var[n_records_written:, :, :] = vv_vals
                time_var[n_records_written:] = date2num(t_vals, time_var.units)

                n_records_written += len(t_vals)


def merge_and_interpolate_temperature_and_precip(data_dir_anusplin: Path = None, out_dir: Path = None,
                                                 additional_data_file_paths: dict = None,
                                                 target_grid_config: GridConfig = None, start_year: int=1980):
    """
    Merges anusplin and nldas and interpolates to the GL grid
    the interpolated daily fields are saved to a netcdf file (PR and TT are both saved to the same file)
    :param data_dir_anusplin:
    :param out_dir:
    :param additional_data_file_paths:
    :param target_grid_config:
    """
    out_file_name = "anusplin+_interpolated_tt_pr.nc"

    # sort files to be in the hronological order
    files_sorted_tmax = list(sorted(mfile for mfile in data_dir_anusplin.iterdir() if "stmx" in mfile.name))
    files_sorted_tmin = list(sorted(mfile for mfile in data_dir_anusplin.iterdir() if "stmn" in mfile.name))
    files_sorted_pcp = list(sorted(mfile for mfile in data_dir_anusplin.iterdir() if "pcp" in mfile.name))

    # make sure that the tmin and tmax files correspond well to each other
    assert len(files_sorted_tmax) == len(files_sorted_tmin) == len(files_sorted_pcp)
    assert files_sorted_tmax[0].name.replace("stmx", "") == files_sorted_tmin[0].name.replace("stmn", "")
    assert files_sorted_tmax[-1].name.replace("stmx", "") == files_sorted_tmin[-1].name.replace("stmn", "")
    assert files_sorted_pcp[-1].name.replace("pcp", "") == files_sorted_tmin[-1].name.replace("stmn", "")
    assert files_sorted_pcp[0].name.replace("pcp", "") == files_sorted_tmin[0].name.replace("stmn", "")


    # To convert to the usual units used by CRCM5, i.e. M/s rather than mm/day
    precip_conversion_factor = 1.0 / (24 * 3600.0 * 1000.0)

    n_records_written = 0

    indices_in_source_field = None




    with Dataset(str(out_dir.joinpath(out_file_name)), "w") as ds_out:
        lon_t, lat_t = target_grid_config.get_lons_and_lats_of_gridpoint_centers()
        # layout of the output netcdf file
        ds_out.createDimension("time")
        ds_out.createDimension("x", lon_t.shape[0])
        ds_out.createDimension("y", lon_t.shape[1])

        lon_var = ds_out.createVariable("lon", "f4", dimensions=("x", "y"))
        lat_var = ds_out.createVariable("lat", "f4", dimensions=("x", "y"))

        lon_var[:] = lon_t
        lat_var[:] = lat_t

        start_date = datetime(1970, 1, 1)

        time_var = ds_out.createVariable("time", "i4", dimensions=("time",))
        time_var.units = "days since {:%Y-%m-%d %H:%M:%S}".format(start_date)

        tt_var = ds_out.createVariable("TT", "f4", dimensions=("time", "x", "y"),
                                       zlib=True,
                                       least_significant_digit=3)
        tt_var.units = "degrees C"
        tt_var.coordinates = "lon lat"

        pr_var = ds_out.createVariable("PR", "f4",
                                       dimensions=("time", "x", "y"),
                                       zlib=True)
        pr_var.units = "M/s"
        pr_var.coordinates = "lon lat"



        ds_additional_tt = xarray.open_dataset(additional_data_file_paths["TT"])
        ds_additional_pr = xarray.open_dataset(additional_data_file_paths["PR"])
        print("Opened {} and {} as additional datasets".format(additional_data_file_paths["TT"],
                                                               additional_data_file_paths["PR"]))


        print(ds_additional_pr["pr"])

        # ds_additional = xarray.open_mfdataset(list(additional_data_file_paths.values()))

        # the masks of good values
        good_vals_additional = None
        good_vals_anusplin = None


        # calculate daily means from anusplin data.
        for fp_tmax, fp_tmin, fp_pcp in zip(files_sorted_tmax, files_sorted_tmin, files_sorted_pcp):


            year, month = [int(token) for token in fp_tmax.name.split(".")[-2].split("_")[-2:]]

            # Skip the dates earlier than the start year
            if year < start_year:
                print("Ignoring data before {}: skipping {}/{:02d}".format(start_year, year, month))
                continue


            ds_tmax = Dataset(fp_tmax)
            ds_tmin = Dataset(fp_tmin)
            ds_pcp = Dataset(fp_pcp)

            t_vals = [datetime(year, month, int(day)) for day in ds_tmax.variables["time"][:]]
            # t_vals = date2num(t_vals, time_var.units)

            tmax_anusplin = ds_tmax.variables["daily_maximum_temperature"][:]
            tmin_anusplin = ds_tmin.variables["daily_minimum_temperature"][:]
            tmean_anusplin = 0.5 * (tmax_anusplin + tmin_anusplin)
            pcp_anusplin = ds_pcp.variables["daily_precipitation_accumulation"][:]


            # get the indices for interpolation
            if indices_in_source_field is None:
                lons_anusplin = ds_tmax.variables["lon"][:]
                lats_anusplin = ds_tmax.variables["lat"][:]

                lons_additional = ds_additional_pr.variables["longitude"][:]
                lats_additional = ds_additional_pr.variables["latitude"][:]

                lons_additional, lats_additional = np.meshgrid(lons_additional, lats_additional)

                # get the interpolation indices

                good_vals_anusplin = ~np.isnan(tmean_anusplin[0, :, :])
                good_vals_additional = ~ds_additional_pr["pr"][0].to_masked_array(copy=False).mask


                lons_all = list(lons_anusplin[good_vals_anusplin]) + list(lons_additional[good_vals_additional])
                lats_all = list(lats_anusplin[good_vals_anusplin]) + list(lats_additional[good_vals_additional])

                xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons_all, lats_all)

                ktree = KDTree(list(zip(xs, ys, zs)))

                xt, yt, zt = lat_lon.lon_lat_to_cartesian(lon_t.flatten(), lat_t.flatten())
                dists, indices_in_source_field = ktree.query(list(zip(xt, yt, zt)))


            ds_additional_pr_sel = None
            ds_additional_tt_sel = None
            try:
                # align anusplin and NLDAS in time and interpolate
                ds_additional_pr_sel = ds_additional_pr.sel(time="{}-{:02d}".format(year, month))
                print("Selected {} pr fields from the additional dataset".format(ds_additional_pr_sel.variables["pr"].shape[0]))

                ds_additional_tt_sel = ds_additional_tt.sel(time="{}-{:02d}".format(year, month))
                print("Selected {} tas fields from the additional dataset".format(ds_additional_tt_sel.variables["tas"].shape[0]))


            except KeyError:
                print("Could not find additional fields for {}-{:02d}, skipping ...".format(year, month))


            # close anusplin datasets for the current month
            ds_tmax.close()
            ds_tmin.close()



            # Skip if there is no additional data for the month
            if ds_additional_pr_sel is None or len(ds_additional_pr_sel) == 0:
                continue



            # interpolate and write the data to the netcdf file
            tt_interpolated = None
            for t, tt_anusplin, pr_anusplin in zip(t_vals, tmean_anusplin, pcp_anusplin):
                tt_additional = ds_additional_tt_sel.sel(method="nearest", time=t)["tas"]
                pr_additional = ds_additional_pr_sel.sel(method="nearest", time=t)["pr"]



                tt_all = list(tt_anusplin[good_vals_anusplin]) + list(tt_additional.data[good_vals_additional])
                pr_all = list(pr_anusplin[good_vals_anusplin]) + list(pr_additional.data[good_vals_additional])

                # convert to numpy arrays for easier indexing
                tt_all = np.asarray(tt_all)
                pr_all = np.asarray(pr_all)

                tt_interpolated = tt_all[indices_in_source_field].reshape(lon_t.shape)
                pr_interpolated = pr_all[indices_in_source_field].reshape(lon_t.shape)


                tt_var[n_records_written, :, :] = tt_interpolated
                pr_var[n_records_written, :, :] = pr_interpolated * precip_conversion_factor  # convert to M/s
                time_var[n_records_written] = date2num(t, time_var.units)

                n_records_written += 1




def get_mean_gridcell_area(lons, lats):

    dlons = (lons[1:, :-1] - lons[:-1, :-1]) * np.sin(np.radians(lats[:-1, :-1]))
    dlats = lats[:-1, 1:] - lats[:-1, :-1]

    area = np.abs(dlons * dlats).mean()
    return area



def interpolate_ice_fractions(input_file_path: Path=None, out_dir: Path=None, target_grid_config: GridConfig=None):
    """


    As, At - mean gridcell areas of the source and target grids, respectively.

    alpha_s, alpha_t - ice fractions of the source and target grids, respectively.


    if At/As = n >= 1:
        alpha_t = sum(alpha_s_i)/n
    else:
        alpha_t = alpha_s


    :param input_file_path:
    :param out_dir:
    :param target_grid_config:
    """
    out_file_name = "cis_nic_glerl_interpolated_lc_fix.nc"


    ds_in = Dataset(input_file_path)
    lon_s, lat_s = [ds_in.variables[k][:] for k in ["lon", "lat"]]

    time_var_in = ds_in.variables["time"]
    time_var_in_data = time_var_in[:]

    ice_cover_var_in = ds_in.variables["ice_cover"]



    with Dataset(out_dir.joinpath(out_file_name), "w") as ds_out:

        lon_t, lat_t = target_grid_config.get_lons_and_lats_of_gridpoint_centers()
        # layout of the output netcdf file
        ds_out.createDimension("time")
        ds_out.createDimension("x", lon_t.shape[0])
        ds_out.createDimension("y", lon_t.shape[1])

        lon_var = ds_out.createVariable("lon", "f4", dimensions=("x", "y"))
        lat_var = ds_out.createVariable("lat", "f4", dimensions=("x", "y"))

        lon_var[:] = lon_t
        lat_var[:] = lat_t


        time_var = ds_out.createVariable("time", "i4", dimensions=("time",))
        time_var.units = time_var_in.units
        time_var[:] = time_var_in_data

        lc_var = ds_out.createVariable("LC", "f4", dimensions=("time", "x", "y"),
                                       zlib=True,
                                       least_significant_digit=3)
        lc_var.units = "-"
        lc_var.coordinates = "lon lat"



        # compare areas in degrees

        area_s = get_mean_gridcell_area(lon_s, lat_s)
        area_t = get_mean_gridcell_area(lon_t, lat_t)


        radius_of_influence = max(area_s, area_t) ** 0.5 * (np.pi / 180.0) * lat_lon.EARTH_RADIUS_METERS

        print("area_s={}, area_t={}".format(area_s, area_t))

        # spatial interpolation
        n_neighbours = max(int(area_t / area_s + 0.5), 1)
        print("nneighbours = {}".format(n_neighbours))


        xs, ys, zs = lat_lon.lon_lat_to_cartesian(lon_s.flatten(), lat_s.flatten())
        ktree = KDTree(list(zip(xs, ys, zs)))


        xt, yt, zt = lat_lon.lon_lat_to_cartesian(lon_t.flatten(), lat_t.flatten())

        dists, inds = ktree.query(list(zip(xt, yt, zt)), k=n_neighbours)

        for i in range(len(time_var_in_data)):
            the_field = ice_cover_var_in[i, :, :]


            if np.any(~the_field.mask):

                the_field[the_field.mask] = np.nan

                if n_neighbours > 1:
                    lc_interpolated = np.mean(the_field.flatten()[inds], axis=1)
                else:
                    lc_interpolated = the_field.flatten()[inds]


                lc_interpolated = np.ma.masked_where(np.isnan(lc_interpolated), lc_interpolated)

                # reshape back to the 2d array
                lc_interpolated.shape = lon_t.shape
            else:
                lc_interpolated = np.ma.masked_all_like(lon_t)

            lc_var[i, :, :] = lc_interpolated


    # close the input dataset
    ds_in.close()



def interpolate_snow_water_equivalent(input_file_path: Path=None, out_dir: Path=None, target_grid_config: GridConfig=None):
    out_file_name = "ross_brown_interpolated_i5.nc"

    ds_in = Dataset(input_file_path)
    lon_s, lat_s = [ds_in.variables[k][:] for k in ["longitude", "latitude"]]

    time_var_in = ds_in.variables["time"]
    time_var_in_data = time_var_in[:]

    swe_var_in = ds_in.variables["SWE"]


    with Dataset(out_dir.joinpath(out_file_name), "w") as ds_out:
        lon_t, lat_t = target_grid_config.get_lons_and_lats_of_gridpoint_centers()
        # layout of the output netcdf file
        ds_out.createDimension("time")
        ds_out.createDimension("x", lon_t.shape[0])
        ds_out.createDimension("y", lon_t.shape[1])

        lon_var = ds_out.createVariable("lon", "f4", dimensions=("x", "y"))
        lat_var = ds_out.createVariable("lat", "f4", dimensions=("x", "y"))

        lon_var[:] = lon_t
        lat_var[:] = lat_t

        time_var = ds_out.createVariable("time", "i4", dimensions=("time",))
        time_var.units = time_var_in.units
        time_var[:] = time_var_in_data

        i5_var = ds_out.createVariable("I5", "f4", dimensions=("time", "x", "y"),
                                       zlib=True)
        i5_var.units = "mm"
        i5_var.coordinates = "lon lat"



        # prepare parameters for interpolation
        xs, ys, zs = lat_lon.lon_lat_to_cartesian(lon_s.flatten(), lat_s.flatten())
        ktree = KDTree(list(zip(xs, ys, zs)))

        xt, yt, zt = lat_lon.lon_lat_to_cartesian(lon_t.flatten(), lat_t.flatten())
        dists, inds = ktree.query(list(zip(xt, yt, zt)))


        for i in range(len(time_var_in_data)):
            the_field = swe_var_in[i, :, :]


            if hasattr(the_field, "mask"):
                the_field[the_field.mask] = np.nan


            if (not hasattr(the_field, "mask")) or np.any(~the_field.mask):

                swe_interpolated = the_field.flatten()[inds]


                swe_interpolated = np.ma.masked_where(np.isnan(swe_interpolated), swe_interpolated)

                # reshape back to the 2d array
                swe_interpolated.shape = lon_t.shape
            else:
                swe_interpolated = np.ma.masked_all_like(lon_t)

            i5_var[i, :, :] = swe_interpolated


    # close the input dataset
    ds_in.close()





def main_for_wc_domain():
    import sys

    # target grid for interpolation
    nml_path = "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/sim_configs/NEI_WC0.11deg_Crr1_gemclim_settings.nml"
    target_grid_config = grid_config.gridconfig_from_gemclim_settings_file(nml_path)
    print(target_grid_config)

    # the output folder
    out_folder = Path("/HOME/huziy/skynet3_rech1/obs_data/anuspl_uw_0.11_wc_domain")

    interpolate_tt_pr = True


    if interpolate_tt_pr:
        # Source data for precip and temperature
        # a) Anusplin data manager
        # b) Additional sources (nldas)
        data_dir_anusplin = Path("/RESCUE/skynet3_rech1/huziy/anusplin_links")
        additional_data_sources = {
            "PR": "/HOME/huziy/skynet3_rech1/obs_data_for_HLES/initial_data/nldas_obs_daily.pr_1980_2010.nc",
            "TT": "/HOME/huziy/skynet3_rech1/obs_data_for_HLES/initial_data/nldas_met_update.obs.daily.tas_1980_2010.nc"
        }
        merge_and_interpolate_temperature_and_precip(data_dir_anusplin=data_dir_anusplin,
                                                     out_dir=out_folder,
                                                     target_grid_config=target_grid_config,
                                                     additional_data_file_paths=additional_data_sources,
                                                     start_year=1980)


def main():
    import sys

    # target grid for interpolation
    nml_path = "/HOME/huziy/skynet3_rech1/obs_data_for_HLES/interploated_to_the_same_grid/GL_0.1_452x260/gemclim_settings.nml"
    target_grid_config = grid_config.gridconfig_from_gemclim_settings_file(nml_path)
    print(target_grid_config)

    # the output folder
    out_folder = Path(nml_path).parent

    interpolate_tt_pr = False
    interpolate_uu_vv = False
    interpolate_lc = False
    interpolate_i5 = False

    if len(sys.argv) > 1:
        args = [arg.lower() for arg in sys.argv[1:]]
        interpolate_uu_vv = "wind" in args
        interpolate_tt_pr = "tt_pr" in args
        interpolate_lc = "lc" in args
        interpolate_i5 = "i5" in args


    if interpolate_tt_pr:
        # Source data for precip and temperature
        # a) Anusplin data manager
        # b) Additional sources (nldas)
        data_dir_anusplin = Path("/RESCUE/skynet3_rech1/huziy/anusplin_links")
        additional_data_sources = {
            "PR": "/HOME/huziy/skynet3_rech1/obs_data_for_HLES/initial_data/nldas_obs_daily.pr_1980_2010.nc",
            "TT": "/HOME/huziy/skynet3_rech1/obs_data_for_HLES/initial_data/nldas_met_update.obs.daily.tas_1980_2010.nc"
        }
        merge_and_interpolate_temperature_and_precip(data_dir_anusplin=data_dir_anusplin,
                                                     out_dir=out_folder,
                                                     target_grid_config=target_grid_config,
                                                     additional_data_file_paths=additional_data_sources,
                                                     start_year=1980)

    if interpolate_lc:
        # Source for the lake ice
        # ice_fraction_data_file_cis_nic = "/HOME/huziy/skynet3_rech1/obs_data_for_HLES/initial_data/glerl_icecov1.nc"
        ice_fraction_data_file_cis_nic = "/RESCUE/skynet3_rech1/huziy/nemo_obs_for_validation/glerl_icecov1_fix.nc"

        interpolate_ice_fractions(input_file_path=Path(ice_fraction_data_file_cis_nic),
                                  out_dir=out_folder,
                                  target_grid_config=target_grid_config)



    if interpolate_uu_vv:
        # Source for u and v components of the 10m wind
        # /RECH/data/Driving_data/Offline/ERA-Interim_0.75/6h_Analysis ()
        winds_dir = "/RECH/data/Driving_data/Offline/ERA-Interim_0.75/6h_Analysis"
        interpolate_wind_components_from_rpn_files(data_dir=Path(winds_dir), out_dir=out_folder,
                                                   target_grid_config=target_grid_config)



    if interpolate_i5:
        # SWE is not really used directly in the hles analysis, but still very hady to have
        swe_in_path = Path("/HOME/huziy/skynet3_rech1/obs_data_for_HLES/initial_data/swe.nc")
        interpolate_snow_water_equivalent(input_file_path=swe_in_path,
                                          out_dir=out_folder,
                                          target_grid_config=target_grid_config)


if __name__ == '__main__':
    # main()
    main_for_wc_domain()
