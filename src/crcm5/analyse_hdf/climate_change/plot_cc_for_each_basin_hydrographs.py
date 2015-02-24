__author__ = 'huziy'

import matplotlib
#matplotlib.use("Agg")
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

from crcm5.analyse_hdf.run_config import RunConfig
from crcm5.analyse_hdf import do_analysis_using_pytables

BASIN_BOUNDARIES_FILE = "data/shape/contour_bv_MRCC/Bassins_MRCC_latlon"

from osgeo import ogr



def calculate_climate_change_hydrographs():
    pass


def main():
    base_current_path = "/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5"
    base_future_path = ""
    base_label = "CRCM5-R"

    start_year_c = 1980
    end_year_c = 2010

    future_shift_years = 61

    params = dict(
        data_path=base_current_path, start_year=start_year_c, end_year=end_year_c, label=base_label
    )

    base_config_c = RunConfig(**params)
    base_config_f = base_config_c.get_shifted_config(future_shift_years)

    print base_config_f


    b = Basemap()
    b.readshapefile(BASIN_BOUNDARIES_FILE, "basin")
    print b.basin_info

    b.drawcoastlines()

    plt.show()
    

    pass

if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()
    main()