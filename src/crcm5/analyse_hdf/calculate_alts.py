import os

import pickle
from netCDF4 import Dataset

from mpl_toolkits.basemap import maskoceans

from application_properties import main_decorator
from permafrost.active_layer_thickness import CRCMDataManager
import matplotlib.pyplot as plt
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis
import numpy as np
from crcm5.analyse_hdf import common_plot_params
from permafrost.draw_regions import save_pf_mask_to_netcdf
from util import plot_utils

qc01deg_sim_layer_widths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 3.0, 5.0, 5.0,
                            5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]


def get_cache_file(data_path=None, lay_widths=None, start_year=None, end_year=None):
    fpath = data_path[:-5] + "_{}_{}_{:.5f}_{}-{}.cache".format(sum(lay_widths), len(lay_widths), np.prod(lay_widths), start_year, end_year)
    return fpath


def do_calculations(data_path=None, lay_widths=None, start_year=None, end_year=None):

    cache = get_cache_file(**locals())


    if os.path.isfile(cache):
        return pickle.load(open(cache, "rb"))


    alt_data_manager = CRCMDataManager(layer_widths=lay_widths)

    tsoil3d_field = None


    for lev in range(len(lay_widths)):
        field_dict = analysis.get_annual_maxima(path_to_hdf_file=data_path, var_name="I0", level=lev,
                                                start_year=start_year, end_year=end_year)

        field = np.max([fi for fi in field_dict.values()], axis=0)

        if tsoil3d_field is None:
            tsoil3d_field = np.zeros(field.shape + (len(lay_widths), ))

        print("lev, min, max = {}, {}, {}".format(lev, field.min(), field.max()))

        plt.figure()
        img = plt.contourf(field)
        plt.colorbar(img)
        plt.show()

        tsoil3d_field[:, :, lev] = field

    alt_field = alt_data_manager.get_alt(tsoil3d_field)

    with open(cache, "wb") as f:
        pickle.dump(alt_field, f)

    return alt_field


@main_decorator
def main():

    pf_mask_file = "pf_mask_qc0.1deg.220x220.nc"
    if not os.path.isfile(pf_mask_file):
        path_to_rpn_with_targ_grid = "/RESCUE/skynet3_rech1/huziy/from_guillimin/new_outputs/current_climate_30_yr_sims/quebec_0.1_crcm5-hcd-rl-intfl_ITFS/Samples/quebec_crcm5-hcd-rl-intfl_197901/pm1979010100_00008928p"
        save_pf_mask_to_netcdf(pf_mask_file, path_to_rpn_with_target_grid=path_to_rpn_with_targ_grid)


    with Dataset(pf_mask_file) as ds:
        pf_mask = ds.variables["pf_type"][:]
        print(ds.variables.keys())

    lay_widths = qc01deg_sim_layer_widths

    data_path = "/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5"
    start_year = 1990
    end_year = 1990

    alt_field = do_calculations(data_path=data_path, lay_widths=lay_widths, start_year=start_year, end_year=end_year)

    alt_field = np.ma.masked_where((alt_field < 0) | (alt_field > 8), alt_field)


    # do the plotting
    plot_utils.apply_plot_params()
    fig = plt.figure()

    bmp_info = analysis.get_basemap_info_from_hdf(file_path=data_path)

    # Mask oceans
    alt_field = maskoceans(bmp_info.lons, bmp_info.lats, alt_field)
    alt_field = np.ma.masked_where((pf_mask > 2) | (pf_mask < 1), alt_field)
    alt_field[(pf_mask == 2) & (~alt_field.mask)] = 5.5


    xx, yy = bmp_info.get_proj_xy()

    img = bmp_info.basemap.contourf(xx, yy, alt_field, extend="both")
    bmp_info.basemap.colorbar(img)
    bmp_info.basemap.drawcoastlines()
    plt.contour(xx, yy, pf_mask, levels=[1, 2, 3], colors="k", linewidth=2)
    fig_path = "{}_alt_{}-{}.png".format(data_path[:-5], start_year, end_year)
    fig.savefig(fig_path, transparent=True, dpi=common_plot_params.FIG_SAVE_DPI, bbox_inches="tight")
    print("Saving the plot to {}".format(fig_path))
    plt.show()



if __name__ == '__main__':
    main()
