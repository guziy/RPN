from pathlib import Path

from collections import defaultdict
from rpn import level_kinds
from rpn.rpn import RPN
from rpn.rpn_multi import MultiRPN
import pandas as pd

from application_properties import main_decorator
import matplotlib.pyplot as plt


# Calculate spatial average of snowdepth [cm] over lakes


def get_lake_fraction(sim_path, fname="pm1979010100_00000000p"):
    static_data_file = Path(sim_path).joinpath("..").joinpath(fname)

    with RPN(str(static_data_file)) as r:
        assert isinstance(r, RPN)
        try:
            lkfr = r.get_first_record_for_name("ML")
        except Exception:
            # Try another field
            lkfr = r.get_first_record_for_name("FU")
    return lkfr


@main_decorator
def main(sim_path="/RECH2/huziy/coupling/GL_440x260_0.1deg_GL_with_Hostetler/Samples_selected/",
         fpath_pattern="var_per_file",
         fname_prefix=None, coords_filename="pm1979010100_00000000p"):



    start_year = 1980
    end_year = 1982

    vname = "SD"

    level = 6
    level_kind = level_kinds.ARBITRARY

    lkfr = get_lake_fraction(sim_path, fname=coords_filename)


    data_series = []

    year_to_paths_cache = defaultdict(list)

    data = None
    for y in range(start_year, end_year + 1):

        if fpath_pattern == "var_per_file":
            r = MultiRPN("{}/*{}*/*{}*".format(sim_path, y, vname))
            data = r.get_all_time_records_for_name_and_level(vname, level=level, level_kind=level_kind)
        elif fpath_pattern == "default":

            sim_dir = Path(sim_path)

            # Create the map in order to reuse
            if len(year_to_paths_cache) == 0:
                for month_dir in sim_dir.iterdir():
                    year_to_paths_cache[y].append(month_dir)

            for month_dir in year_to_paths_cache[y]:
                for fpath in month_dir.iterdir():

                    # print(fpath)

                    # Check if the prefix is OK
                    if not fpath.name.startswith(fname_prefix):
                        continue


                    # Check if this is not a timestep 0
                    if fpath.name[:-1].endswith(8 * "0"):
                        continue

                    with RPN(str(fpath)) as r:
                        data_tmp = r.get_all_time_records_for_name_and_level(vname, level=level, level_kind=level_kind)
                        if data is None:
                            data = data_tmp
                        else:
                            data.update(data_tmp)


        else:
            raise Exception("Unknown file path pattern: {}".format(fpath_pattern))



        ts_for_year = {}
        for d, field in data.items():
            ts_for_year[d] = field[lkfr > 0.5].mean()

        data_series.append(pd.Series(ts_for_year))

    total_series = pd.concat(data_series)
    assert isinstance(total_series, pd.Series)
    ax = total_series.plot(title="{}-{}".format(start_year, end_year))
    plt.show()

if __name__ == '__main__':
    # main()
    # main(
    #     sim_path="/RECH2/huziy/TestGeophyNEI/NorthAmerica_0.44deg_test_SOIL_OM_MODIS_FLAKE/Samples",
    #     fpath_pattern="default", fname_prefix="pm", coords_filename="coords.rpn"
    # )

    # main(
    #     sim_path="/HOME/data/Simulations/CRCM5/North_America/NorthAmerica_0.44deg_ERA40-Int0.75_B1/Samples/",
    #     fpath_pattern="default",
    #     fname_prefix="pm",
    #     coords_filename="Samples/NorthAmerica_0.44deg_ERA40-Int0.75_B1_195801/pm1958010100_00000000p"
    # )


    main(
        sim_path="/RECH2/huziy/TestGeophyNEI/NorthAmerica_0.44deg_test_SOIL_OM_MODIS_HSSDFIX/Samples",
        fpath_pattern="default",
        fname_prefix="pm",
        coords_filename="Samples/NorthAmerica_0.44deg_test_SOIL_OM_MODIS_HSSDFIX_198001/pm1980010100_00000000p"
    )


