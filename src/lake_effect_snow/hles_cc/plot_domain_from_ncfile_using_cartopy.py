from collections import OrderedDict
from pathlib import Path

from matplotlib.axes import Axes

from crcm5.nemo_vs_hostetler.main_for_lake_effect_snow import get_mask_of_points_near_lakes
from lake_effect_snow.hles_cc import common_params
from lake_effect_snow.hles_cc.plot_cc_2d_all_variables_for_all_periods import get_gl_mask
import xarray
import matplotlib.pyplot as plt


def plot_domain_and_interest_region(ax: Axes, topo_nc_file_path: Path, region_mask, region_mask_lons, region_mask_lats):
    """
    :param region_mask_lats: latitudes corresponding to the region mask
    :param region_mask_lons:
    :param ax:
    :param topo_nc_file_path:
    :param region_mask:

    Note: below is the expected structure of the input netcdf file

    $ ncdump -h geophys_452x260_me.nc
    netcdf geophys_452x260_me {
    dimensions:
        x = 452 ;
        y = 260 ;
    variables:
        float ME(x, y) ;
        float lon(x, y) ;
        float lat(x, y) ;
        int proj_params ;
            proj_params:grid_type = "E" ;
            proj_params:lat1 = 0. ;
            proj_params:lon1 = 180. ;
            proj_params:lat2 = 1. ;
            proj_params:lon2 = 276. ;
    }
    """

    # read the model topography from the file
    with xarray.Dataset(topo_nc_file_path) as topo_ds:
        topo_lons, topo_lats, topo = [topo_ds[k].values for k in ["lon", "lat", "ME"]]


    # TODO:

    pass


def test():
    data_root = common_params.data_root

    label_to_datapath = OrderedDict([
        (common_params.crcm_nemo_cur_label, data_root / "lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_1989-2010_1989-2010"),
        (common_params.crcm_nemo_fut_label, data_root / "lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_2079-2100_2079-2100"),
    ])

    gl_mask = get_gl_mask(label_to_datapath[common_params.crcm_nemo_cur_label] / "merged")
    hles_region_mask = get_mask_of_points_near_lakes(gl_mask, npoints_radius=10)

    topo_ncfile = data_root / "geophys_452x260_me.nc"


    fig = plt.figure()
    ax = plt.gca()
    plot_domain_and_interest_region(ax, topo_ncfile, hles_region_mask)
    plt.show()



if __name__ == '__main__':
    test()