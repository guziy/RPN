from rpn.rpn import RPN

__author__ = 'huziy'

import numpy as np

import matplotlib.pyplot as plt

def main():
    path = "/home/huziy/skynet3_rech1/init_cond_for_lake_infl_exp/anal_NorthAmerica_0.44deg_ERA40-Int1.5_B1_rmn13_and_Class_1979010100_dates_same3"
    #path = "/home/huziy/skynet3_rech1/init_cond_for_lake_infl_exp/from_Analysis/anal_NorthAmerica_0.44deg_ERA40-Int1.5_B1_rmn13_1979010100"
    rObj = RPN(path)

    depth = rObj.get_first_record_for_name("SD")
    dens = rObj.get_first_record_for_name("DN")
    snwfr = rObj.get_first_record_for_name("5P")
    snoma = rObj.get_first_record_for_name("I5")
    tsurf = rObj.get_first_record_for_name_and_level(varname="I0",level=1)

    plt.figure()


    #to_plot = np.ma.masked_where(depth != 1 > 0, depth)
    #print to_plot.min(), to_plot.max()
    plt.title("depth")
    plt.pcolormesh(depth.transpose(), vmax = 150)
    plt.colorbar()

    plt.figure()
    plt.title("snow fraction")
    plt.pcolormesh(snwfr.transpose())
    plt.colorbar()



    plt.figure()
    plt.title("swe - rho*h*frac")
    plt.pcolormesh(snoma.transpose() - (depth * 1e-2 * dens * snwfr).transpose())
    plt.colorbar()


    plt.figure()
    plt.title("tsurf")
    plt.pcolormesh(tsurf.transpose())
    plt.colorbar()




    plt.show()



    #TODO: implement
    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print("Hello world")
  