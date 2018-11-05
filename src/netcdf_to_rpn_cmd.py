
"""
Command line interface to the netcdf_to_rpn.py script

Usage: Should be launched from the RPN folder.
the PYTHONPATH should contain the path to ...RPN/src
Assuming you are in the RPN folder, the PYTHONPATH could be updated as follows

    export PYTHONPATH=./src:${PYTHONPATH}

"""

import argparse

from domains import grid_config
from netcdf_to_rpn import convert


def main():
    parser = argparse.ArgumentParser(description="Convert a netcdf file with directions created by DDM to a standard file.")

    parser.add_argument("--nc", help="Paht to the input netcdf file")
    parser.add_argument("--fst", help="Path to the output standard file")
    parser.add_argument("--nml", help="Path to the gemclim_settings.nml")

    args = parser.parse_args()
    gc = grid_config.gridconfig_from_gemclim_settings_file(args.nml)
    convert(nc_path=args.nc, out_path=args.fst, gc=gc)


if __name__ == '__main__':
    main()