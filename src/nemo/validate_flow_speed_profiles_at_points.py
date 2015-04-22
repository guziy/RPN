from pathlib import Path

__author__ = 'san'


import matplotlib.pyplot as plt

from nemo.nemo_yearly_files_manager import NemoYearlyFilesManager

def get_img_folder():
    img_dir = Path("nemo/adcp_comparisons")
    if not img_dir.is_dir():
        img_dir.mkdir(parents=True)

def plot_profiles():
    obs_base_dir = Path("/home/huziy/skynet3_rech1/nemo_obs_for_validation/data_from_Ram_Yerubandi/ADCP-profiles")
    obs_dir_list = [
        str(obs_base_dir.joinpath("105.317")),
        str(obs_base_dir.joinpath("155.289"))
    ]

    model_folder = "/home/huziy/skynet3_rech1/offline_glk_output_daily_1979-2012"

    manager_nemo_u = NemoYearlyFilesManager(folder=model_folder, suffix="_U.nc")
    manager_nemo_v = NemoYearlyFilesManager(folder=model_folder, suffix="_V.nc")


    # TODO: compare observed and modelled profiles of the flow velocity module





def plot_point_positions():
    fig = plt.figure()



def main():
    pass



if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()
    main()