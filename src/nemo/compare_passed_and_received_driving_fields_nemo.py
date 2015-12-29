from application_properties import main_decorator

# Objective: To check if what passed to NEMO from CRCM, is handled OK

from rpn.rpn_multi import MultiRPN

@main_decorator
def main():
    path_to_nemo_outputs = "/RESCUE/skynet3_rech1/huziy/one_way_coupled_nemo_outputs_1979_1985/GLK_1d_grid_T.nc"
    vname_nemo = "sosstsst"

    path_to_crcm5_outputs = "/home/huziy/skynet3_rech1/glk-oneway-coupled-crcm5-outputs/coupled-GL/Samples"
    vname_crcm5 = "TT"

    # month of interest
    month = 4
    year = 1981

    mrpn = MultiRPN("{}/*_{}{}".format(path_to_crcm5_outputs, year, month))





if __name__ == '__main__':
    main()