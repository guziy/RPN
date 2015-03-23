from collections import OrderedDict
from nemo.glerl_icecov_data2d_interface import GLERLIceCoverManager
from nemo.nemo_yearly_files_manager import NemoYearlyFilesManager

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

    glerl_manager = GLERLIceCoverManager()
    # glerl_manager.get_data_for_day(the_date=datetime(2005, 1, 3))
    glerl_manager.get_clim_of_max_icecover_interpolated_to(
        lons2d_target=nemo_manager.lons, lats2d_target=nemo_manager.lats, start_year=2003, end_year=2004, month=2
    )


if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()
    main()