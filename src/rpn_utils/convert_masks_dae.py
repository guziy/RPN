import os
from rpn.rpn import RPN

from netCDF4 import Dataset

__author__ = 'huziy'

def combine():
    out_folder = "/skynet1_rech3/huziy/Converters/NetCDF_converter/"
    path1 = "/skynet1_rech3/huziy/Converters/NetCDF_converter/mappe.rpnw"
    r = RPN(path1)
    mask1 = r.get_first_record_for_name("MK")[10:-10, 10:-10]
    r.close()


    path2 = "/skynet1_rech3/huziy/Converters/NetCDF_converter/champs_st.rpnw"
    r = RPN(path2)
    mk = r.get_first_record_for_name("MK")
    print r.get_dateo_of_last_read_record()
    lons2d, lats2d = r.get_longitudes_and_latitudes_for_the_last_read_rec()
    r.close()


    #combine the masks
    mk = mk * 7 + mask1


    # Write to netcdf file
    ds = Dataset(os.path.join(out_folder, "mask_combined.nc"), mode="w")

    #subset
    vars_list = [mk, lons2d, lats2d]
    ni, nj = vars_list[0].shape
    ds.createDimension("lon", ni)
    ds.createDimension("lat", nj)

    var_names = ["MK", "longitude", "latitude"]
    for the_name, field in zip(var_names, vars_list):
        ncVar = ds.createVariable(the_name, "f4", ("lat", "lon"))
        ncVar[:] = field.transpose()
    ds.close()



def main():
    path = "/skynet1_rech3/huziy/Converters/NetCDF_converter/mappe.rpnw"
    #path = "/skynet1_rech3/huziy/Converters/NetCDF_converter/champs_st.rpnw"
    r = RPN(path)
    mk = r.get_first_record_for_name("MK")
    print r.get_dateo_of_last_read_record()
    lons2d, lats2d = r.get_longitudes_and_latitudes_for_the_last_read_rec()
    r.close()


    # Write to netcdf file
    ds = Dataset(path + ".nc", mode="w")

    #subset
    vars_list = [mk, lons2d, lats2d]
    vars_list = [v[10:-10, 10:-10] for v in vars_list]
    ni, nj = vars_list[0].shape
    ds.createDimension("lon", ni)
    ds.createDimension("lat", nj)

    var_names = ["MK", "longitude", "latitude"]
    for the_name, field in zip(var_names, vars_list):
        ncVar = ds.createVariable(the_name, "f4", ("lat", "lon"))
        ncVar[:] = field.transpose()
    ds.close()

if __name__ == "__main__":
    #main()
    combine()