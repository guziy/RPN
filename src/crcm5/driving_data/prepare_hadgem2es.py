
# Following the scripts written by B. Dugas (UQAM)


# TM and LG are daily data => 1 file per month, level probably does not matter, make 1mb as in other cases
# TT, HU, UU, VV, P0, GZ - 6hourly data => 1 folder per month and inside 1 file per time step, levels hybrid-pressure interpolated
# from the original hybrid heights

# Note: for both types of the output files make sure that the YYYYMMdd00 is in the YYYY$prevmonth(MM)$
from pathlib import Path

from netCDF4 import Dataset

hdgem_to_crcm_vname = {
    "tos": "TM",  # surface temperature
    "sic": "LG",  # ice fraction
    "hus": "HU",
    "ua": "UU",
    "va": "VV",
    "ta": "TT",
    "ps": "P0" # surface pressure
}

hadgem_to_crcm_mult = {
    "LG": 1,
}




def get_orography_meters(in_data_dir, vname="tas"):
    p = Path(in_data_dir)

    for folder in p.iterdir():

        if not folder.name.startswith(vname):
            continue


        if not folder.is_dir():
            continue

        for fpath in folder.iterdir():

            with Dataset(str(fpath)) as ds:
                return ds.variables["orog"][:]




def main():

    in_data_dir = "/b10_fs1/huziy/driving_data/hadgem2_es/r1i1p1/rcp85/NetCDF"

    orog_m = get_orography_meters(in_data_dir)




    pass


if __name__ == '__main__':
    main()