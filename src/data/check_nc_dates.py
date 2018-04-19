from pathlib import Path
from netCDF4 import Dataset

import numpy as np

def main():
    data_dir = Path("/snow3/huziy/NEI/WC/NEI_WC0.11deg_Crr1/Netcdf_exports_WC011_modified")


    files_not_ok = []
    for nf in data_dir.iterdir():
        if not nf.name.endswith(".nc"):
            continue

        print(f"Checking {nf}")
        with Dataset(str(nf)) as ds:
            t = ds.variables["t"][:]

            not_ok = False
            if np.any(t < 0):
                print(f"ERROR: negative time values encountered in {nf}")
                not_ok = True

            if len(np.unique(t)) < len(t):
                print(f"ERROR: duplicate time values encountered in {nf}")
                not_ok = True


            t_prev = 0
            for ti in t:
                if ti < t_prev:
                    print(f"ERROR: decreasing time values encountered in {nf}(t={ti}; tprev={t_prev})")
                    not_ok = True
                    break
                t_prev = ti

            if not not_ok:
                print(f"{nf} is OK")
                files_not_ok.append(str(nf))


    print(f"Summary: {len(files_not_ok)} have problems with time axis")
    print("Found problems in: ")
    for f in files_not_ok:
        print(f)



if __name__ == '__main__':
    main()