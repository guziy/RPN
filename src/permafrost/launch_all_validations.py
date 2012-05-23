from util import plot_utils

__author__ = 'huziy'

import temperature_validation as tmp_val
import swe_validation as swe_val
import compare_mean_alt_and_from_climatology as alt_val

from multiprocessing import Process

def main():
    """
    Launch all the validators in parallel
    """
    plot_utils.apply_plot_params(width_pt=None, width_cm=27, height_cm=40, font_size=25)
    p1 = Process(target=tmp_val.validate_using_monthly_diagnostics)
    p1.start()

    p2 = Process(target=swe_val.validate_using_monthly_diagnostics)
    p2.start()

    p3 = Process(target=alt_val.plot_current_alts)
    p3.start()

    p1.join()
    p2.join()
    p3.join()
    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print "Hello world"
  