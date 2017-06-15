


# Do a quick plot for temperature and precip biases for ~0.44 and 0.11 simulations
from collections import OrderedDict
from pathlib import Path

from util.seasons_info import MonthPeriod


def main():

    img_folder = Path("nei_validation")

    if not img_folder.exists():
        img_folder.mkdir()



    seasons = OrderedDict([
        ("DJF", MonthPeriod(12, 3)),
        ("JJA", MonthPeriod(6, 3)),
    ])

    sim_paths = OrderedDict()


    start_year = 1980
    end_year = 1988

    sim_paths["WC_0.44deg_default"] = Path("/HOME/huziy/skynet3_rech1/CRCM5_outputs/NEI/diags/NEI_WC0.44deg_default/Diagnostics")
    sim_paths["WC_0.44deg_ctem+frsoil+dyngla"] = Path("/HOME/huziy/skynet3_rech1/CRCM5_outputs/NEI/diags/debug_NEI_WC0.44deg_Crr1/Diagnostics")
    sim_paths["WC_0.11deg_ctem+frsoil+dyngla"] = Path("/HOME/huziy/skynet3_rech1/CRCM5_outputs/NEI/diags/NEI_WC0.11deg_Crr1/Diagnostics")











if __name__ == '__main__':
    main()
