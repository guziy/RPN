

from rpn.rpn import RPN
from rpn.domains.rotated_lat_lon import RotatedLatLon
import numpy as np


import matplotlib.pyplot as plt


def main():
    path = "/RESCUE/skynet3_rech1/huziy/from_guillimin/new_outputs/current_climate_30_yr_sims/quebec_0.1_crcm5-hcd-rl-intfl_ITFS/Samples/quebec_crcm5-hcd-rl-intfl_198806/pm1979010100_00999072p"
    # path = "/RESCUE/skynet3_rech1/huziy/temp/pm1979010100_00054144p"

    r = RPN(path)


    water_density = 1000.0
    specific_cond_heat = 0.250100e7  # J/kg

    av = np.mean([field for field in r.get_all_time_records_for_name_and_level(varname="AV").values()], axis=0)

    traf = np.mean([field for field in r.get_all_time_records_for_name_and_level(varname="TRAF", level=6).values()], axis=0)

    traf_agg = np.mean([field for field in r.get_all_time_records_for_name_and_level(varname="TRUN", level=5).values()], axis=0)

    pr = np.mean([field for field in r.get_all_time_records_for_name_and_level(varname="PR").values()], axis=0)


    # fv = np.mean([field for field in r.get_all_time_records_for_name_and_level(varname="FV", level=6).values()], axis=0)

    # mask = fv < 1e-4
    #
    #
    # plt.figure()
    # fv = np.ma.masked_where(mask, fv)
    # cs = plt.contourf(fv.T)
    # plt.title("Output (fv)")
    # plt.colorbar()

    plt.figure()
    av1 = (pr * water_density - traf / 2) * specific_cond_heat
    # av1 = np.ma.masked_where(mask, av1)
    cs = plt.contourf(av1.T, 40)
    plt.title("Calculated (from traf(6))")
    plt.colorbar()

    plt.figure()
    # av = np.ma.masked_where(mask, av)
    cs = plt.contourf(av.T, levels=cs.levels, cmap=cs.cmap, norm=cs.norm)
    plt.title("Output (av)")
    plt.colorbar()




    plt.show()



if __name__ == '__main__':
    main()