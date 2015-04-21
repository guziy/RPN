__author__ = 'huziy'

# due to a bug in LIM3 the ice surface temperature is named: isnotem2

from nemo.nemo_yearly_files_manager import NemoYearlyFilesManager

def main():
    model_data_manager = NemoYearlyFilesManager(
        folder="/home/huziy/skynet3_rech1/offline_glk_output_daily_1979-2012"
    )




if __name__ == '__main__':
    main()
