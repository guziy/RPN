from crcm5.model_data import Crcm5ModelDataManager

__author__ = 'huziy'

from data import cehq_station
import matplotlib.pyplot as plt


def main():
    stations = cehq_station.load_from_hydat_db(natural = True, province = "QC")

    dm = Crcm5ModelDataManager(
        samples_folder_path="/skynet3_rech1/huziy/from_guillimin/new_outputs/quebec_0.1_crcm5-r_spinup",
        all_files_in_samples_folder=True)

    basemap = dm.get_rotpole_basemap()

    lons = [s.longitude for s in stations]
    lats = [s.latitude for s in stations]

    n_cont_years = [len(s.get_list_of_complete_years()) for s in stations]


    x, y = basemap(lons, lats)
    basemap.scatter(x, y, c = n_cont_years)
    basemap.drawcoastlines()
    basemap.colorbar()
    plt.show()



if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
