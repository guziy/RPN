
#Author: Huziy
#Object: interpolate forcing fields to the model grid

import sys
import os
import shutil

from netCDF4 import Dataset
import numpy as np
from scipy.spatial.ckdtree import cKDTree


from multiprocessing import Pool
from util.geo.lat_lon import lon_lat_to_cartesian


def interpolate_bathymetry(in_data, kdtree,
                           source_coords = None,
                           target_coords = None,
                           out_data_shape = None):
    """
    Interpolate bathymetry data so no ocean points are lost
    :param in_data:
    :param kdtree:
    """
    assert isinstance(kdtree, cKDTree)
    #determine characteristic lengths of the source and target grids
    xt2d, yt2d, zt2d = [s.reshape(out_data_shape) for s in target_coords]
    xs2d, ys2d, zs2d = [s.reshape(in_data.shape) for s in source_coords]


    #determine approximate dimensions of the target grid
    pt1 = np.asarray([xt2d[0, 0], yt2d[0, 0], zt2d[0, 0]])
    pt2 = np.asarray([xt2d[1, 0], yt2d[1, 0], zt2d[1, 0]])
    pt3 = np.asarray([xt2d[0, 1], yt2d[0, 1], zt2d[0, 1]])

    d_target_x = np.sqrt(sum((pt1 - pt2) ** 2))
    d_target_y = np.sqrt(sum((pt1 - pt3) ** 2))

    #determine approximate dimensions of the source grid
    d_source_x = (xs2d[:-1, :] - xs2d[1:, :]) ** 2 + \
                 (ys2d[:-1, :] - ys2d[1:, :]) ** 2 + \
                 (zs2d[:-1, :] - zs2d[1:, :]) ** 2

    d_source_y = (xs2d[:, :-1] - xs2d[:, 1:]) ** 2 + \
                 (ys2d[:, :-1] - ys2d[:, 1:]) ** 2 + \
                 (zs2d[:, :-1] - zs2d[:, 1:]) ** 2


    d_source_x = np.median(np.sqrt(d_source_x))
    d_source_y = np.median(np.sqrt(d_source_y))


    print d_source_x, d_source_y, d_target_x, d_target_y
    nneighbours = int((d_target_x * d_target_y) / (d_source_x * d_source_y))

    print "nneighbours = ", nneighbours

    in_data_flat = in_data.flatten()
    if nneighbours <= 1:
        dists, inds = kdtree.query(zip(*target_coords))
        return in_data_flat[inds].reshape(out_data_shape)
    else:
        dists, inds = kdtree.query(zip(*target_coords), k = nneighbours)
        w = 1.0 / dists ** 2
        print w.shape
        out_data_flat = np.sum(in_data_flat[inds] * w, axis=1) / w.sum(axis = 1)

        #count 0
        depth_lower_limit_m = 1e-8  # m
        valpoints = in_data_flat[inds] > depth_lower_limit_m
        mask_apl = valpoints.astype(int).sum(axis = 1) >= max(1, nneighbours // 2)
        mask_apl = mask_apl.astype(int)
        out_data_flat *= mask_apl
        return out_data_flat.reshape(out_data_shape)



class Interpolator(object):
    """
    Object: Efficient nearest neighbor interpolator
    Does not reinterpolate if the output file already exists
    """

    def __init__(self, coord_file = "coordinates.nc"):
        ds = Dataset(coord_file)
        self.target_lons = ds.variables["glamt"][:]
        self.target_lats = ds.variables["gphit"][:]

        print "target lons shape = ", self.target_lons.shape
        ds.close()
    
    
    def interpolate_file(self, inpath, outpath):
        """
        Interpolate data in the file, save the result to a new
        file in the same folder, with the name of the interpolated variable 
        and time variable unchanged.
        """
        #check if the output file already exists
        if os.path.isfile(outpath):
            print "{0} already exists, remove to recreate ...".format(outpath)
            return

        ds_in = Dataset(inpath)
        print "working on {0}".format(inpath)
      
        lon_ncatts = {}
        lat_ncatts = {}

        #read in and calculate coordinates of source grid

        if "lon" in ds_in.variables:
            in_lon_var_name = "lon"
            in_lat_var_name = "lat"
        elif "nav_lon" in ds_in.variables:
            in_lon_var_name = "nav_lon"
            in_lat_var_name = "nav_lat"
        else:
            raise Exception("The file does not contain conventional lat/lon information: {0}".format(inpath))


        in_lon_var = ds_in.variables[in_lon_var_name]
        in_lat_var = ds_in.variables[in_lat_var_name]

         
        source_lons, source_lats = in_lon_var[:], in_lat_var[:]
        if in_lon_var.ndim == 1:
           source_lons, source_lats = np.meshgrid(source_lons, source_lats)


        for attname in in_lon_var.ncattrs():
            lon_ncatts[attname] = in_lon_var.getncattr(attname)

        for attname in in_lat_var.ncattrs():
           lat_ncatts[attname] = in_lat_var.getncattr(attname)


        #find the name of the field to be interpolated, and read it into memory
        varnames = ds_in.variables.keys()        
        
        #write interpolated data
        ds_out = Dataset(outpath, "w", format = "NETCDF3_CLASSIC")
        
        #copy and create dimensions
        ds_out.createDimension("time", None)        
        ds_out.createDimension("x", self.target_lons.shape[1])
        ds_out.createDimension("y", self.target_lons.shape[0])

 

        #copy and interpolate variables
        lonVar = ds_out.createVariable(in_lon_var_name, "f4", ("y", "x"))
        latVar = ds_out.createVariable(in_lat_var_name, "f4", ("y", "x"))
        #set the attributes 
        if len(lon_ncatts):
            lonVar.setncatts(lon_ncatts)
            latVar.setncatts(lat_ncatts)


        good_points = np.abs(source_lons.flatten()) < 360        

        xs, ys, zs = lon_lat_to_cartesian(source_lons.flatten(), source_lats.flatten())
        xt, yt, zt = lon_lat_to_cartesian(self.target_lons.flatten(), self.target_lats.flatten())
        
        ktree = cKDTree(data = zip(xs[good_points], ys[good_points], zs[good_points]))
        dists, inds = ktree.query(zip(xt, yt, zt), k = 1)
        
        for varname in varnames:
            outVar = None
            if varname.lower() in ["time", "time_counter"]:
                time_var_in = ds_in.variables[varname] 
                time_var_out = ds_out.createVariable(varname, "f4", ("time",))
                time_var_out[:] = time_var_in[:]
                if hasattr(time_var_in, "units"):
                    time_var_out.units = time_var_in.units
                continue
            
            inVar = ds_in.variables[varname]
            #Interpolate only 3d variables (time, lat, lon) and some 2d variables
            if inVar.ndim == 3: 
                outVar = ds_out.createVariable(varname, "f4", ("time", "y", "x"))
                outVar.units = inVar.units
                in_data = inVar[:]
                #reshape to 2d
                in_data.shape = (in_data.shape[0], -1)
                out_data = in_data[:, inds]
                out_data.shape = (inVar.shape[0], ) + self.target_lons.shape
                #print out_data.shape
                outVar[:] = out_data

            elif varname.lower() == "bathymetry":
                outVar = ds_out.createVariable(varname, "f4", ("y", "x"))
                outVar[:] = interpolate_bathymetry(inVar[:], ktree, source_coords=(xs, ys, zs),
                                                   target_coords=(xt, yt, zt),
                                                   out_data_shape = self.target_lons.shape)

            elif inVar.ndim == 2 and varname.lower() in ["socoefr"]:
                outVar = ds_out.createVariable(varname, "f4", ("y", "x"))

                in_data = inVar[:]
                #reshape to 2d
                in_data = in_data.flatten()
                out_data = in_data[inds]
                out_data.shape = self.target_lons.shape
                #print out_data.shape
                outVar[:] = out_data

            if outVar is not None:
                #Set attributes of the interpolated fields
                if hasattr(outVar, "long_name"):
                    outVar.long_name = inVar.long_name

                if hasattr(inVar, "units"):
                    outVar.units = inVar.units

                if hasattr(inVar, "missing_value"):
                    outVar.missing_value = inVar.missing_value

                if hasattr(inVar, "coordinates"):
                    outVar.coordinates = inVar.coordinates




        lonVar[:] = self.target_lons
        latVar[:] = self.target_lats
        #close netcdf files
        ds_out.close()
        ds_in.close() 


def ignore_copy_func(dirpath, files):
    return filter(lambda f: os.path.isfile(os.path.join(dirpath, f)) and not f.endswith(".nc"), files)


def apply_interpolator(arg):
    worker, inpath, outpath = arg
    worker.interpolate_file(inpath, outpath)
    return 0


def main(infolder = "DFS4.3", coord_file = "", outfolder = None):
    if outfolder is None:
        outfolder = infolder + "_interpolated"
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)
 
    worker = Interpolator(coord_file=coord_file)
    #in_file = "DFS4.3_interpolated/t2/t2_DFS4.3_1985_sht.nc" 
    #worker.interpolate_file(in_file)    

    ##interpolate sst and salinity
    in_ncpaths = []
    out_ncpaths = []

    #worker.interpolate_file(in_ncpaths[0], out_ncpaths[0])


    ##interpolate runoffs
    #in_ncpaths += ["runoff_1m_nomask.nc"]
    #out_ncpaths += ["runoff_1m_nomask_interpolated.nc"]

    #find all netcdf files interpolate and save to the same structure, but with adding
    # _interpolated to the topmost folder
    for root, dirs, files in os.walk(infolder):
        root_name = os.path.basename(root)
        if root_name.lower() == "masks":
            continue
     
        dir_paths = [os.path.join(outfolder, d) for d in dirs]
        out_root = root.replace(infolder, outfolder)
        for dpath in dir_paths:
            if not os.path.isdir(dpath):
                os.mkdir(dpath)

        ncfiles = filter(lambda f: f.endswith(".nc") and not f.startswith("interpolated"), files)
        in_ncpaths += [os.path.join(root, f) for f in ncfiles]
        out_ncpaths += [os.path.join(out_root, f) for f in ncfiles] 



    print len(in_ncpaths), len(out_ncpaths) 
    pool = Pool()
    workers = [worker] * len(in_ncpaths)
    res = pool.map(apply_interpolator, zip(workers, in_ncpaths, out_ncpaths))
    assert sum(res) == 0
    #for in_path, out_path in zip(in_ncpaths, out_ncpaths):
    #    worker.interpolate_file(in_path, out_path)


def interpolate_10km_grid_glk():
    main(infolder="/home/huziy/skynet3_rech1/NEMO/WORK_GRTLKS/DFS4.3",
         coord_file="/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/nemo_grids/coordinates_rotpole_nx170_ny90_dx0.1_dy0.1.nc",
         outfolder="/home/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/EXP_0.1deg/DFS4.3_interpolated")

if __name__ == "__main__":    
    #main()
    interpolate_10km_grid_glk()