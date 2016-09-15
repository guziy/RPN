from pathlib import Path

from application_properties import main_decorator
from data.swe import SweDataManager


@main_decorator
def main():

    # Monthly means from diagnostics
    model_data_path = Path("/RECH2/huziy/BC-MH/bc_mh_044deg/Diagnostics")





    swe_obs_manager = SweDataManager(var_name="SWE")

    swe_obs_manager.get_daily_clim_fields_interpolated_to()


    pass


if __name__ == '__main__':
    pass