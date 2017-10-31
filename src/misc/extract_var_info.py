

# extract variable info for a simulation

import re
from pathlib import Path

from rpn.rpn import RPN
from rpn.variable import RPNVariable


def __parse_line(line):
    cols = line.split("|")
    print(f"line={line}")
    name = cols[2].split("\"")[1].strip()
    descr = cols[3].strip()

    print(f"{name} ===> {descr}")

    return name, descr


def __parse_line_phy_ini(line, pattern=None):
    cols = line.split(";")

    if pattern is None:
        pattern = re.compile(r".*\=\s*(.*)")

    name = pattern.search(cols[1]).group(1).strip()
    name_internal = pattern.search(cols[0]).group(1).strip()
    descr = pattern.search(cols[2]).group(1).strip()

    print(f"{name} ===> {descr}")

    return [name, name_internal], descr


def save_info(var_freqs, var_desc, out_file_name, ignore_vars=None):
    with Path(out_file_name).open("w") as f:

        f.write("{:<8s}{:<8s}{:<160s}\n".format("Name", "Freq", "Description"))

        for vn in sorted(var_freqs):

            if vn in ignore_vars:
                continue

            descr = var_desc[vn].upper().replace("MOYHR", var_freqs[vn][:-1])
            f.write(f"{vn:<8s}{var_freqs[vn]:<8s}{descr:<160s}\n")


def parse_var_descriptions(listing_file: Path):
    extract_flag = False

    res = {}

    with listing_file.open() as f:
        for line in f:
            print(line)

            if line.count("+") == 10:
                extract_flag = not extract_flag
                continue

            if extract_flag:
                name, descr = __parse_line(line)
                res[name] = descr

    return res


def parse_var_descriptions_from_phy_ini(source_file: Path, pattern=None):
    res = {}

    with source_file.open() as f:
        for line in f:

            line = line.upper().strip()

            if line.startswith("!"):
                continue

            extract_flag = "GESDICT" in line

            if extract_flag:
                names, descr = __parse_line_phy_ini(line, pattern=pattern)

                for name in names:
                    res[name] = descr.strip()

    return res


def __parse_year_and_month(fname):
    my_s = fname.split("_")[-1]
    m = int(my_s[-2:])
    y = int(my_s[:-2])
    return y, m


def __get_num_hours_in_month(fname):
    y, m = __parse_year_and_month(fname)
    import calendar
    d1, ndays = calendar.monthrange(y, m)
    return ndays * 24


def extract_save_freqency_hours(month_out_dir: Path)-> dict:

    checked_prefixes = []

    res = {}
    for afile in month_out_dir.iterdir():

        # do not check the same file type twice
        if afile.name[:2] in checked_prefixes:
            continue

        print(f"Examining {afile}")

        with RPN(str(afile)) as r:
            for vname, v in r.variables.items():
                assert isinstance(v, RPNVariable)

                dt = v.sorted_dates[-1] - v.sorted_dates[0]

                if len(v.sorted_dates) > 1:
                    res[vname] = dt.total_seconds() / 3600 / (len(v.sorted_dates) - 1)
                    res[vname] = f"{res[vname]:.1f}h"
                else:
                    prefix = afile.name[:2]
                    num_files = len([fi for fi in month_out_dir.iterdir() if fi.name.startswith(prefix)])

                    if num_files > 1:
                        res[vname] = __get_num_hours_in_month(month_out_dir.name) / num_files
                        res[vname] = f"{res[vname]:.1f}h"
                    else:
                        res[vname] = "monthly"

    return res


def expand_vnames_in_descriptions(vdescriptions: dict):

    ignored_words = ["AT", "IN", "FROM", "OF", "BY", "WITH", "VIS", "SW", "LW", "FOR", "BARE"
                     "WRT", "S", "C", "KM", "KG", "K", "T", "IR", "WEQ", "M3", "M",
                     "SHAL", "HRS", "PFT", "VOL", "SOIL", "LAKE", "AVG", "MEAN", "OR", "AND",
                     "PFTS", "RUNOFF", "BASE", "MIN", "STREAMFLOW", "GLACIER", "BIN", "SOIL",
                     "VIA", "LID", "ICE", "SNOW"
                     ]
    for vn in vdescriptions:
        descr = vdescriptions[vn]

        words = [w for w in descr.split() if (w not in ignored_words and w in vdescriptions)]

        if len(words) > 0:
            for w in words:

                if "." in w:
                    continue

                if "/" in w:
                    continue

                if "(" in w:
                    continue

                if ")" in w:
                    continue

                descr = descr.replace(w, f"{w}({vdescriptions[w]})")

        vdescriptions[vn] = descr




def main():
    out_dir = Path("var_info_nei")
    out_dir.mkdir(parents=True, exist_ok=True)

    phy_ini_current = "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Fortran/gemclim/phy_ini.ftn90"
    phy_ini_current = Path(phy_ini_current)

    simulations = {
        "WC044_modified":
            ("/snow3/huziy/NEI/WC/debug_NEI_WC0.44deg_Crr1/Listings/debug_NEI_WC0.44deg_Crr1_201110_M_32575.1",
             "/snow3/huziy/NEI/WC/debug_NEI_WC0.44deg_Crr1/Samples/debug_NEI_WC0.44deg_Crr1_201401"),
        "WC044_default":
            ("/snow3/huziy/NEI/WC/NEI_WC0.44deg_default/Listings/NEI_WC0.44deg_default_201401_M_29642.1",
             "/snow3/huziy/NEI/WC/NEI_WC0.44deg_default/Samples/NEI_WC0.44deg_default_201312"),
        "WC011_modified":
            ("/snow3/huziy/NEI/WC/NEI_WC0.11deg_Crr1/Listings/NEI_WC0.11deg_Crr1_201205_M_27681.1",
             "/snow3/huziy/NEI/WC/NEI_WC0.11deg_Crr1/Samples/NEI_WC0.11deg_Crr1_201111")
    }


    # L1 - is ignored when running with FLake
    # INTF - is ignored when running without interflow
    ignore_vars = ["INTF", "L1"]

    for label, (listing_file_path, month_out_dir) in simulations.items():
        listing_file_path = Path(listing_file_path)
        month_out_dir = Path(month_out_dir)

        # vdescriptions = parse_var_descriptions(listing_file_path)
        vdescriptions = parse_var_descriptions_from_phy_ini(phy_ini_current,
                                                            pattern=re.compile(r".*\=\s*(.*)\s*"))
        vdescriptions.update({
            "GZ": "Geopotential height",
            "HR": "Relative humidity",
            "HU": "Specific humidity",
            "P0": "Surface pressure",
            "PN": "Sea level pressure",
            "QQ": "Absolute vorticity (s**-1)",
            "TT": "Air temperature (C)",
            "UU": "Zonal wind component (knots)",
            "VV": "Meridional wind component (knots)",
            "WW": "Vertical motion (Pa/s)"
        })



        expand_vnames_in_descriptions(vdescriptions)

        save_info(
            extract_save_freqency_hours(month_out_dir),
            vdescriptions,
            out_dir / f"{label}.txt", ignore_vars=ignore_vars
        )


if __name__ == '__main__':
    main()