from collections import OrderedDict

from mpl_toolkits.basemap import Basemap

from application_properties import main_decorator
from crcm5.global_simulation import commons

from rpn.rpn_multi import MultiRPN


def plot_row(row=0, season_to_mean=commons.default_seasons, difference=False, label="", basemap=None, xx=None, yy=None):
    pass



@main_decorator
def main():
    seasons = commons.default_seasons


    label_to_folder = OrderedDict([
        ("ERA-Interim", "/home/huziy/skynet3_rech1/CNRCWP/Calgary_flood/SST_SeaIce/I_SST_SeaIce"),
        ("PreI-CanESM2", "/RESCUE/skynet3_rech1/huziy/CNRCWP/Calgary_flood/SST_SeaIce/PreI_SST_SeaIce"),
        ("PreI-GFDL", "/RESCUE/skynet3_rech1/huziy/CNRCWP/Calgary_flood/SST_SeaIce/PreI_SST_SeaIce"),
        ("PreI-GISS", "/RESCUE/skynet3_rech1/huziy/CNRCWP/Calgary_flood/SST_SeaIce/PreI_SST_SeaIce"),

    ])

    bmp = Basemap(projection="robin", lon_0=0)


if __name__ == '__main__':
    main()