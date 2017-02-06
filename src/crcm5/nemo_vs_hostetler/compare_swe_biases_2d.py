from collections import OrderedDict

from pathlib import Path

from data.swe import SweDataManager


def main():

    start_year = 1979
    end_year = 1988


    img_folder = "nemo_vs_hostetler_GL_extended_domain"

    # create the image folder if necessary
    img_folder_p = Path(img_folder)
    if not img_folder_p.is_dir():
        img_folder_p.mkdir()



    HL_LABEL = "CRCM5_HL"
    NEMO_LABEL = "CRCM5_NEMO"


    sim_label_to_path = OrderedDict(
        [(HL_LABEL, "/RECH2/huziy/coupling/GL_440x260_0.1deg_GL_with_Hostetler/Samples_selected"),
         (NEMO_LABEL, "/RECH2/huziy/coupling/coupled-GL-NEMO1h_30min/Samples")]
    )



    obs_manager = SweDataManager()

    




    pass


if __name__ == '__main__':
    main()