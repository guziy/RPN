from collections import defaultdict
from pathlib import Path

from rpn.rpn import RPN
from scipy.spatial import KDTree

from application_properties import main_decorator
from crcm5.mh_domains import default_domains
from domains.grid_config import GridConfig

from netCDF4 import Dataset, OrderedDict

from util.geo import lat_lon


import numpy as np

def vname_map(vname):
    if vname in ["8L", "DPTH"]:
        return "DPTH_BDRCK"
    return vname


def select_data_to_nc(gconfig: GridConfig, in_rpn_path:str, in_directions_path:str, selection=None, out_dir: Path=None, label="mh"):

    out_file = Path("{}_sand_clay_dpth_{}x{}_{}.nc".format(label, gconfig.ni, gconfig.nj, gconfig.dx))

    if out_dir is not None:
        out_file = out_dir.joinpath(out_file)


    lon_t, lat_t = gconfig.get_lons_and_lats_of_gridpoint_centers()
    inds = None


    roughness = [0.2, 1.4, 0.8]
    dpth_to_bedrock = None
    sand = None
    clay = None

    xt, yt, zt = None, None, None

    mg = None
    dpth_to_bedrock_vname = None

    vname_to_data = {}
    with Dataset(str(out_file), "w", format="NETCDF3_CLASSIC") as ds:
        assert isinstance(ds, Dataset)

        ds.createDimension("longitude", lon_t.shape[0])
        ds.createDimension("latitude", lon_t.shape[1])

        with RPN(in_rpn_path) as r:
            assert isinstance(r, RPN)
            for vname, lev in selection.items():
                if lev is not None:
                    data = r.get_first_record_for_name_and_level(varname=vname, level=lev)
                else:
                    data = r.get_first_record_for_name(varname=vname)


                # Create the dimensions
                lon_s, lat_s = r.get_longitudes_and_latitudes_for_the_last_read_rec()

                xs, ys, zs = lat_lon.lon_lat_to_cartesian(lon_s.flatten(), lat_s.flatten())

                xt, yt, zt = lat_lon.lon_lat_to_cartesian(lon_t.flatten(), lat_t.flatten())

                ktree = KDTree(list(zip(xs, ys, zs)))

                dists, inds = ktree.query(list(zip(xt, yt, zt)), k=1)


                # get the subset for the grid
                # do the interpolation
                data = data.flatten()[inds]
                data.shape = lon_t.shape




                if vname in ["8L", "DPTH"]:
                    dpth_to_bedrock = data
                    dpth_to_bedrock_vname = vname_map(vname)


                elif vname.lower() in ["sand"]:
                    sand = data
                elif vname.lower() in ["clay"]:
                    clay = data

                    mg = r.get_first_record_for_name("MG").flatten()[inds].reshape(lon_t.shape)

                vname_to_data[vname_map(vname)] = data



        # write the roughness
        manning = -np.ones_like(dpth_to_bedrock)

        # rock or ice
        manning[(clay < -1) & (clay > -4)] = 0.01

        # ocean =  (np.abs(sand) < 0.1) & (np.abs(clay) < 0.1)
        ocean = (mg < 0.01)

        good = ~ocean



        for vname, data in vname_to_data.items():
            data[ocean] = -1
            # create the variable in the output file
            v = ds.createVariable(vname_map(vname), "f4", dimensions=("longitude", "latitude"))
            v[:] = data

            if vname in ["8L", "DPTH"]:
                v.units = "m"



        manning[good] = clay[good] * roughness[0] + sand[good] * roughness[2] + (100 - sand[good] - clay[good]) * roughness[1]

        manning[good] /= 100.0
        manning[ocean] = -1
        v = ds.createVariable("RManningBF", "f4", dimensions=("longitude", "latitude"))
        v.description = "Bankfull channel roughness "
        v[:] = manning


        #write the residence time
        # sand, silt, clay
        res_time_days = [10, 30, 60]
        v = ds.createVariable("GWdelay", "f4", dimensions=("longitude", "latitude"))
        restime = -np.ones_like(dpth_to_bedrock)
        restime[clay < 0] = res_time_days[1]
        restime[good] = sand[good] * res_time_days[0] + clay[good] * res_time_days[2] + (100 - sand[good] - clay[good]) * res_time_days[2]
        restime[good] /= 100.0

        restime[ocean] = -1

        v[:] = restime
        v.units = "days"
        v.description = "Ground water residence time"


        # write the cross-section and bankfull volume
        ds_dir = Dataset(in_directions_path)
        lons_dir, lats_dir = [ds_dir.variables[k][:] for k in ["lon", "lat"]]
        xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons_dir.flatten(), lats_dir.flatten())
        ktree_dir = KDTree(list(zip(xs, ys, zs)))

        dists_dir, inds_dir = ktree_dir.query(list(zip(xt, yt, zt)))


        v = ds.createVariable("ChCrossSection", "f4", dimensions=("longitude", "latitude"))
        a2 = 11.0
        a3 = 0.43
        a4 = 1.0
        faa = ds_dir.variables["accumulation_area"][:]
        chlen = ds_dir.variables["channel_length"][:]

        #interpolate (flow accumulation area and channel length)
        faa = faa.flatten()[inds_dir].reshape(lon_t.shape)
        reg = faa >= 0
        ccsection = -np.ones_like(faa)
        ccsection[reg] = a2 + a3 * faa[reg] ** a4
        v[:] = ccsection
        v.units = "m ** 2"


        chlen = chlen.flatten()[inds_dir].reshape(lon_t.shape)
        v = ds.createVariable("BankfullVolume", "f4", dimensions=("longitude", "latitude"))

        reg = (chlen >= 0) & (faa >= 0)
        bf = -np.ones_like(chlen)
        bf[reg] = faa[reg] * chlen[reg]
        v[:] = bf
        v.units = "m ** 3"







@main_decorator
def main():

    """
    Prepare sand clay and depth to bedrock in netcdf for MH
    """


    gconfig_to_infile = OrderedDict([
        (default_domains.bc_mh_011, "/RESCUE/skynet3_rech1/huziy/directions_for_ManitobaHydro/20170320/rpn/geophys_CORDEX_NA_0.11deg_695x680_filled_grDes_barBor_Crop2Gras_peat"),
        (default_domains.bc_mh_022, "/RESCUE/skynet3_rech1/huziy/directions_for_ManitobaHydro/20170320/rpn/geophys_CORDEX_NA_0.22deg_filled_grDes_barBor_Crop2Gras_peat"),
        (default_domains.bc_mh_044, "/RESCUE/skynet3_rech1/huziy/directions_for_ManitobaHydro/20170320/rpn/geophys_CORDEX_NA_0.44d_filled_grDes_barBor_Crop2Gras_peat"),
    ])


    # gconfig_to_infile = {
    #     default_domains.bc_mh_011: "/HOME/data/Simulations/CRCM5/North_America/NorthAmerica_0.11deg_ERA40-Int0.75_B1/Samples/NorthAmerica_0.11deg_ERA40-Int0.75_B1_195801/pm1958010100_00000000p",
    #     default_domains.bc_mh_022: "/HOME/data/Simulations/CRCM5/North_America/NorthAmerica_0.22deg_ERA40-Int0.75_B1/Samples/NorthAmerica_0.22deg_ERA40-Int0.75_B1_195801/pm1958010100_00000000p",
    #     default_domains.bc_mh_044: "/HOME/data/Simulations/CRCM5/North_America/NorthAmerica_0.44deg_ERA40-Int0.75_B1/Samples/NorthAmerica_0.44deg_ERA40-Int0.75_B1_195801/pm1958010100_00000000p",
    # }

    gconfig_to_dirfile = {
        default_domains.bc_mh_011: "/HOME/huziy/skynet3_rech1/directions_for_ManitobaHydro/20170310/netcdf/directions_bc-mh_0.11deg_new_hsfix.nc",
        default_domains.bc_mh_022: "/HOME/huziy/skynet3_rech1/directions_for_ManitobaHydro/20170310/netcdf/directions_bc-mh_0.22deg_new_hsfix.nc",
        default_domains.bc_mh_044: "/HOME/huziy/skynet3_rech1/directions_for_ManitobaHydro/20170310/netcdf/directions_bc-mh_0.44deg_new_hsfix.nc",

    }


    out_dir = Path("mh/sand_clay_dpth_fields")
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)


    # selection criteria
    selection = {
        "SAND": 1,
        "CLAY": 1,
        "DPTH": None
    }


    # ----
    print(gconfig_to_infile)
    print(gconfig_to_dirfile)


    for gc, rpn_path in gconfig_to_infile.items():
        select_data_to_nc(gc, in_rpn_path=rpn_path, in_directions_path=gconfig_to_dirfile[gc], selection=selection, out_dir=out_dir)
        break




if __name__ == '__main__':
    main()