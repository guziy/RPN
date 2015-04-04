import os
from rpn.rpn import RPN

__author__ = 'huziy'

from pyhdf.SD import SD
import matplotlib.pyplot as plt

def main():

    varname_to_rpn_name = {
        "precipitation": "PR",
        "relativeError": "RERR"
    }

    varnames = list(varname_to_rpn_name.keys())

    target_dir = "/skynet3_rech1/huziy/from_hdf4"
    source_dir = "/st1_fs2/winger/Validation/TRMM/HDF_format"

    for f_name in os.listdir(source_dir):
        if not f_name.endswith("HDF"):
            continue

        path = os.path.join(source_dir, f_name)
        ds = SD(path)
        print(ds.datasets())
        target_path = os.path.join(target_dir, f_name + ".rpn")
        r_obj = RPN(target_path, mode="w")
        for varname in varnames:
            var_data = ds.select(varname)[0, :, :]
            r_obj.write_2D_field(
                name=varname_to_rpn_name[varname],
                data=var_data, label=varname, grid_type="L",
                ig = [25, 25, 4013, 18012])
        r_obj.close()









if __name__ == "__main__":
    main()