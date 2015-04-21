from collections import OrderedDict
from datetime import datetime
from nemo.glerl_icecov_data2d_interface import GLERLIceCoverManager
from nemo.nemo_yearly_files_manager import NemoYearlyFilesManager
import matplotlib.pyplot as plt

__author__ = 'huziy'


def main():
    season_to_months = OrderedDict([
        ("Dec", [12, ]),
        ("Jan", [1, ]),
        ("Feb", [2, ]),
        ("Mar", [3, ]),
        ("Apr", [4, ])])

    start_year = 2003
    end_year = 2012


    nemo_manager = NemoYearlyFilesManager(folder="/home/huziy/skynet3_rech1/offline_glk_output_daily_1979-2012")
    # nemo_manager = NemoYearlyFilesManager(folder="/Users/san/NEMO/outputs")

    # glerl_manager = GLERLIceCoverManager(data_folder="/Users/san/NEMO/validation/glerl_ice_data")
    glerl_manager = GLERLIceCoverManager()
    glerl_manager.get_data_for_day(the_date=datetime(2005, 1, 3))
    obs_ice_cover_interp = glerl_manager.get_icecover_interpolated_to(
        lons2d_target=nemo_manager.lons, lats2d_target=nemo_manager.lats, the_date=datetime(2013, 4, 3))


    fig = plt.figure()
    xx, yy = nemo_manager.basemap(nemo_manager.lons, nemo_manager.lats)
    im = nemo_manager.basemap.pcolormesh(xx, yy, obs_ice_cover_interp)
    nemo_manager.basemap.colorbar(im)
    nemo_manager.basemap.drawcoastlines()
    plt.show()

if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()
    main()