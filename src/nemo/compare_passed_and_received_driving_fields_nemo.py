from application_properties import main_decorator

# Objective: To check if what passed to NEMO from CRCM, is handled OK

from rpn.rpn_multi import MultiRPN
from rpn import level_kinds

import numpy as np

from netCDF4 import Dataset

@main_decorator
def main():
    path_to_nemo_outputs = "/RESCUE/skynet3_rech1/huziy/one_way_coupled_nemo_outputs_1979_1985/GLK_1d_grid_T.nc"
    vname_nemo = "sosstsst"

    path_to_crcm5_outputs = "/home/huziy/skynet3_rech1/glk-oneway-coupled-crcm5-outputs/coupled-GL/Samples"
    vname_crcm5 = "TT"

    # month of interest
    month = 4
    year = 1981

    mrpn = MultiRPN("{}/*_{}{}/dm*".format(path_to_crcm5_outputs, year, month))
    data = mrpn.get_all_time_records_for_name_and_level(varname=vname_crcm5, level=1, level_kind=level_kinds.HYBRID)

    # Calculate the monthly mean fields in both cases
    assert isinstance(data, dict)
    mm_crcm5 = np.array(list(data.values())).mean(axis=0)

    print("crcm5-out-shape = ", mm_crcm5.shape)
    with Dataset(path_to_nemo_outputs) as ds:
        mm_nemo = ds.variables[vname_nemo][:].mean(axis=0)

    print("nemo-out-shape = ", mm_nemo.shape)




if __name__ == '__main__':
    main()