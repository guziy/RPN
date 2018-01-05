

import matplotlib.pyplot as plt
from pathlib import Path
from rpn.rpn import RPN
import numpy as np


vname_to_prefix = {
    "TT": "dm",
    "TJ": "pm",
    "UU": "dm",
    "UD": "pm"
}





def __get_monthly_mean(vname="", month_dir: Path = None):
    for f in month_dir.iterdir():
        if not f.name.startswith(vname_to_prefix[vname]):
            continue

        with RPN(str(f)) as r:
            v = r.variables[vname][:].squeeze()
            return v.mean(axis=0)



def main():
    month_dir = Path("/snow3/huziy/NEI/WC/debug_NEI_WC0.44deg_Crr1/Samples/debug_NEI_WC0.44deg_Crr1_198409")


    # compare temperatures
    tt = __get_monthly_mean("TT", month_dir=month_dir)
    tj = __get_monthly_mean("TJ", month_dir=month_dir) - 273.15

    dtt = tt - tj
    plt.figure()

    im = plt.pcolormesh(dtt.T)
    plt.colorbar(im)

    print(f"np.abs(dtt).max() = {np.abs(dtt).max()}")

    # compare winds
    uu = __get_monthly_mean("UU", month_dir=month_dir) * 0.514444
    ud = __get_monthly_mean("UD", month_dir=month_dir)

    duu = uu - ud
    plt.figure()

    im = plt.pcolormesh(duu.T)
    plt.colorbar(im)

    print(f"np.abs(duu).max() = {np.abs(duu).max()}")

    plt.show()

    print(month_dir)



if __name__ == '__main__':
    main()