__author__ = 'huziy'

import os

import pandas as pd

# Validate temperature profiles, flow profiles at given points





def main(folder_path="~/skynet3_rech1/nemo_obs_for_validation/data_from_Ram_Yerubandi"):
    folder_path = os.path.expanduser(folder_path)

    temperature_profile_file_prefixes = [
        "08-01T-004A024.120.290", "08-01T-013A054.120.290"
    ]



    pass


if __name__ == '__main__':
    # For tests at home
    main(folder_path="/Users/san/NEMO/validation/from_Ram_Yerubandi")