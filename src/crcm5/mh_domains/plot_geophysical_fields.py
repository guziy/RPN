from pathlib import Path

from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import maskoceans
from rpn.rpn import RPN
from scipy.spatial import KDTree

from application_properties import main_decorator
from glaciers.domain_plot_directions_and_glaciers import plot_acc_area_with_glaciers
from rpn_utils.get_coord_data_from_rpn_file import get_lons_lats_basemap, IndexSubspace

import matplotlib.pyplot as plt

from util import plot_utils
from util.geo import lat_lon
import numpy as np
from matplotlib import colors
from util.geo.mask_from_shp import get_mask

comment = """  Taken from tuiles.cdk:
*     Class       Vegetation type
*     =====       ===============
*       1         (salt) water
*       2         ice
*       3         inland lake
*       4         evergreen needleleaf trees
*       5         evergreen broadleaf trees
*       6         deciduous needleleaf trees
*       7         deciduous broadleaf trees
*       8         tropical broadleaf trees
*       9         drought deciduous trees
*       10        evergreen broadleaf shrub
*       11        deciduous shrubs
*       12        thorn shrubs
*       13        short grass and forbs
*       14        long grass
*       15        crops
*       16        rice
*       17        sugar
*       18        maize
*       19        cotton
*       20        irrigated crops
*       21        urban
*       22        tundra
*       23        swamp
*       24        bare soil
*       25        mixed wood forests
*       26        mixed shrubs
"""

import re

vf_level_to_title = {int(re.findall("\d+", line)[0]): " ".join(re.findall("[a-zA-Z]+", line)).capitalize() for line in comment.split("\n")
                     if len(re.findall("\d+", line)) > 0}


vf_level_to_title[2] = "Glacier fraction"


GLACIERS_LEVEL = 2



img_folder = Path("nei_geofields_plots")

vname_to_level_to_title = {
    "Y2C": {1: "needleleaf fraction", 2: "broadleaf fraction", 3: "crops fraction", 4: "grass fraction", 5: "urban fraction"},
    "VF": vf_level_to_title,
    "DPTH": {0: "depth to bedrock"},
    "SAND": {1: "Sand", 2: "Sand", 3: "Sand"},
    "CLAY": {1: "Clay", 2: "Clay", 3: "Clay"},
    "FACC": {1: r"Drainage area (km**2)"},
    "ME": {0: "Topography (m)", 1: "Topography (m)", 2: "Topography (m)"},
    "NBIN": {8: "Number of elevation bins"},
    "BME": {1: "Minimum bin elevation", 8: "Maximum bin elevation"}

}

vname_to_units = {
    "FACC": "km**2"
}


def get_title_and_units_for_name(vname, level=-1):
    title = vname
    units = None
    if vname in vname_to_level_to_title:
        if level in vname_to_level_to_title[vname]:
            title = vname_to_level_to_title[vname][level]

    if vname in vname_to_units:
        units = vname_to_units[vname]

    return title, units






def plot_field(field_name, data_source, levels, index_subspace=None,
               file_with_target_coords="/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/fields_from_Caio/WC011_VF2.rpn"):


    shape_files = {
        # "k": "/BIG1/aganji/skynet3_rech3/CLASS_snow/output_31year_HiRes_ck_paper_aug26/CORDEX_NA_0.44/PROVINCE.shp",
        "m": "/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/GRDC_basins/GRDC_405_basins_from_mouth.shp"
    }


    lons_t, lats_t, bmap = get_lons_lats_basemap(file_with_target_coords, index_subset=index_subspace)
    xx, yy = bmap(lons_t, lats_t)


    # create the folder for images if it does not exist yet
    if not img_folder.is_dir():
        img_folder.mkdir()


    plot_utils.apply_plot_params()

    vname = field_name

    for ds_label, ds_path in data_source.items():

        with RPN(ds_path) as r:
            assert isinstance(r, RPN)
            print(r.get_list_of_varnames())


            lev_to_field = r.get_2D_field_on_all_levels(name=vname)

            lons_s, lats_s = r.get_longitudes_and_latitudes_for_the_last_read_rec()

            xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons_s.flatten(), lats_s.flatten())
            xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons_t.flatten(), lats_t.flatten())

            ktree = KDTree(list(zip(xs, ys, zs)))


            dists, inds = ktree.query(list(zip(xt, yt, zt)))

            for the_level in levels:
                the_field = lev_to_field[the_level]


                # Do the plotting
                fig = plt.figure()

                ax = plt.gca()
                ax.set_title(vname_to_level_to_title[vname][the_level] + " {}".format(ds_label))

                to_plot = the_field.flatten()[inds].reshape(xx.shape)
                inland = not (vname == "VF" and the_level == 3)
                to_plot = maskoceans(np.where(lons_t < 180, lons_t, lons_t - 360), lats_t, to_plot, inlands=inland)

                if vname in ["FACC"]:
                    cs = bmap.contourf(xx, yy, to_plot, 20, ax=ax, norm=LogNorm())
                else:
                    cs = bmap.contourf(xx, yy, to_plot, 20, ax=ax)

                bmap.colorbar(cs)
                bmap.drawcoastlines(ax=ax)

                # read the relevant shape files
                for i, (clr, shp) in enumerate(shape_files.items()):
                    bmap.readshapefile(shp[:-4], "field_{}".format(i), color=clr, linewidth=2.5)

                img_path = img_folder.joinpath("{}_{}_{}_{}.png".format(vname, the_level, ds_label,
                                                                        vname_to_level_to_title[vname][the_level].replace(" ", "_"))
                                               )
                fig.tight_layout()
                fig.savefig(str(img_path), bbox_inches="tight")
                plt.close(fig)



def plot_glaciers():

    multiplier = 100.0
    selected_vars = {"VF": [GLACIERS_LEVEL, ]}
    data_source = {
        # "(CRCM5-default, from GenPhysX)": "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/geophys_CORDEX_NA_0.11deg_695x680_filled_grDes_barBor_Crop2Gras_peat_with_directions",
        # "(MODIS)": "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/fields_from_Bernardo/VF_WCAN.rpn",
        "(RGI)": "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/fields_from_Caio/WC011_VF2.rpn"
    }


    file_with_target_coords = "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/fields_from_Caio/WC011_VF2.rpn"

    shape_files = {
        # "k": "/BIG1/aganji/skynet3_rech3/CLASS_snow/output_31year_HiRes_ck_paper_aug26/CORDEX_NA_0.44/PROVINCE.shp",
         "m": "/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/GRDC_basins/GRDC_405_basins_from_mouth.shp"
    }


    lons_t, lats_t, bmap = get_lons_lats_basemap(file_with_target_coords)

    nx, ny = lons_t.shape

    indx_subspace = IndexSubspace(i_start=0, i_end=nx // 1.5, j_start=80, j_end=ny - 60)
    lons_t, lats_t, bmap = get_lons_lats_basemap(file_with_target_coords, index_subset=indx_subspace)

    xx, yy = bmap(lons_t, lats_t)


    clevels = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    color_list = ["white", "blue", "royalblue", "dodgerblue", "cyan", "lime", "yellow", "orange", "red", "darkred"]
    assert len(clevels) == len(color_list) + 1

    cmap, norm = colors.from_levels_and_colors(clevels, color_list)

    # create the folder for images if it does not exist yet
    if not img_folder.is_dir():
        img_folder.mkdir()


    plot_utils.apply_plot_params()


    for ds_label, ds_path in data_source.items():

        with RPN(ds_path) as r:
            assert isinstance(r, RPN)
            print(r.get_list_of_varnames())

            for vname, levels in selected_vars.items():
                lev_to_field = r.get_2D_field_on_all_levels(name=vname)

                lons_s, lats_s = r.get_longitudes_and_latitudes_for_the_last_read_rec()

                xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons_s.flatten(), lats_s.flatten())
                xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons_t.flatten(), lats_t.flatten())

                ktree = KDTree(list(zip(xs, ys, zs)))


                dists, inds = ktree.query(list(zip(xt, yt, zt)))

                for the_level in levels:
                    the_field = lev_to_field[the_level]


                    # Do the plotting
                    fig = plt.figure()

                    ax = plt.gca()
                    ax.set_title(vname_to_level_to_title[vname][the_level] + " {}".format(ds_label))

                    to_plot = the_field.flatten()[inds].reshape(xx.shape)

                    to_plot *= multiplier

                    to_plot = maskoceans(np.where(lons_t < 180, lons_t, lons_t - 360), lats_t, to_plot)

                    if vname not in ["FACC"]:
                        # cs = bmap.contourf(xx, yy, to_plot, 20, ax=ax)
                        cs = bmap.pcolormesh(xx, yy, to_plot, cmap=cmap, norm=norm, ax=ax)
                    else:
                        cs = bmap.contourf(xx, yy, to_plot, 20, ax=ax, norm=LogNorm())

                    bmap.colorbar(cs, ticks=clevels)
                    bmap.drawcoastlines(ax=ax)
                    bmap.drawcountries()
                    bmap.drawstates()

                    # read the relevant shape files
                    for i, (clr, shp) in enumerate(shape_files.items()):
                        bmap.readshapefile(shp[:-4], "field_{}".format(i), color=clr)

                    img_path = img_folder.joinpath("{}_{}_{}_{}.png".format(vname, the_level, ds_label,
                                                                            vname_to_level_to_title[vname][the_level].replace(" ", "_")))
                    fig.tight_layout()
                    fig.savefig(str(img_path), bbox_inches="tight")
                    plt.close(fig)




def plot_number_of_bins():

    multiplier = 1

    selected_vars = {"NBIN": [8, ]}
    nbins_max = 8
    data_source = {
        # "(CRCM5-default, from GenPhysX)": "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/geophys_CORDEX_NA_0.11deg_695x680_filled_grDes_barBor_Crop2Gras_peat_with_directions",
        # "(MODIS)": "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/fields_from_Bernardo/VF_WCAN.rpn",
        "": "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/fields_from_Caio/bins_WC011"
    }


    file_with_target_coords = "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/fields_from_Caio/WC011_VF2.rpn"

    shape_files = {
        # "k": "/BIG1/aganji/skynet3_rech3/CLASS_snow/output_31year_HiRes_ck_paper_aug26/CORDEX_NA_0.44/PROVINCE.shp",
         "m": "/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/GRDC_basins/GRDC_405_basins_from_mouth.shp"
    }


    with RPN(file_with_target_coords) as r:
        assert isinstance(r, RPN)
        glacier_fraction = r.get_first_record_for_name_and_level(varname="VF", level=2)


    lons_t, lats_t, bmap = get_lons_lats_basemap(file_with_target_coords)

    nx, ny = lons_t.shape

    indx_subspace = IndexSubspace(i_start=0, i_end=nx // 1.5, j_start=80, j_end=ny - 60)
    lons_t, lats_t, bmap = get_lons_lats_basemap(file_with_target_coords, index_subset=indx_subspace)

    xx, yy = bmap(lons_t, lats_t)


    clevels = np.arange(0.5, 9, 1)

    color_list = ["blue", "royalblue", "dodgerblue", "cyan", "lime", "yellow", "orange", "red", "darkred", "brown"][:len(clevels) - 1]

    print(len(clevels), len(color_list))

    assert len(clevels) == len(color_list) + 1

    cmap, norm = colors.from_levels_and_colors(clevels, color_list)

    # create the folder for images if it does not exist yet
    if not img_folder.is_dir():
        img_folder.mkdir()


    plot_utils.apply_plot_params()


    for ds_label, ds_path in data_source.items():

        with RPN(ds_path) as r:
            assert isinstance(r, RPN)
            print(r.get_list_of_varnames())

            for vname, levels in selected_vars.items():
                lev_to_field = r.get_2D_field_on_all_levels(name=vname)

                lons_s, lats_s = r.get_longitudes_and_latitudes_for_the_last_read_rec()

                xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons_s.flatten(), lats_s.flatten())
                xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons_t.flatten(), lats_t.flatten())

                ktree = KDTree(list(zip(xs, ys, zs)))


                dists, inds = ktree.query(list(zip(xt, yt, zt)))

                for the_level in levels:
                    the_field = lev_to_field[the_level]


                    # Do the plotting
                    fig = plt.figure()

                    ax = plt.gca()
                    ax.set_title(vname_to_level_to_title[vname][the_level] + " {}".format(ds_label))

                    to_plot = the_field.flatten()[inds].reshape(xx.shape)

                    to_plot *= multiplier

                    # mask points without glacier fractions
                    to_plot = np.ma.masked_where(glacier_fraction < 0.01, to_plot)

                    to_plot = maskoceans(np.where(lons_t < 180, lons_t, lons_t - 360), lats_t, to_plot)



                    if vname not in ["FACC"]:
                        # cs = bmap.contourf(xx, yy, to_plot, 20, ax=ax)
                        cs = bmap.pcolormesh(xx, yy, to_plot, cmap=cmap, norm=norm, ax=ax)
                    else:
                        cs = bmap.contourf(xx, yy, to_plot, 20, ax=ax, norm=LogNorm())

                    bmap.colorbar(cs, ticks=clevels)
                    bmap.drawcoastlines(ax=ax)
                    bmap.drawcountries()
                    bmap.drawstates()

                    # read the relevant shape files
                    for i, (clr, shp) in enumerate(shape_files.items()):
                        bmap.readshapefile(shp[:-4], "field_{}".format(i), color=clr)

                    img_path = img_folder.joinpath("{}_{}_{}_{}.png".format(vname, the_level, ds_label,
                                                                            vname_to_level_to_title[vname][the_level].replace(" ", "_")))
                    fig.tight_layout()
                    fig.savefig(str(img_path), bbox_inches="tight")
                    plt.close(fig)




def plot_min_and_max_bin_elevations():

    multiplier = 1

    data_source = {
        # "(CRCM5-default, from GenPhysX)": "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/geophys_CORDEX_NA_0.11deg_695x680_filled_grDes_barBor_Crop2Gras_peat_with_directions",
        # "(MODIS)": "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/fields_from_Bernardo/VF_WCAN.rpn",
        "": "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/fields_from_Caio/bins_WC011"
    }



    file_with_target_coords = "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/fields_from_Caio/WC011_VF2.rpn"

    shape_files = {
        # "k": "/BIG1/aganji/skynet3_rech3/CLASS_snow/output_31year_HiRes_ck_paper_aug26/CORDEX_NA_0.44/PROVINCE.shp",
         "m": "/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/GRDC_basins/GRDC_405_basins_from_mouth.shp"
    }


    lons_t, lats_t, bmap = get_lons_lats_basemap(file_with_target_coords)

    nx, ny = lons_t.shape

    indx_subspace = IndexSubspace(i_start=0, i_end=nx // 1.5, j_start=80, j_end=ny - 60)
    lons_t, lats_t, bmap = get_lons_lats_basemap(file_with_target_coords, index_subset=indx_subspace)

    xx, yy = bmap(lons_t, lats_t)


    # create the folder for images if it does not exist yet
    if not img_folder.is_dir():
        img_folder.mkdir()


    plot_utils.apply_plot_params()


    vname = "BME"

    selected_fields_titles = {
        "BME_min": "Min. bin elevation",
        "BME_max": "Max. bin elevation"
    }

    selected_fields = {k: None for k in selected_fields_titles}

    clevels = np.arange(0, 5600, 200)


    for ds_label, ds_path in data_source.items():

        with RPN(ds_path) as r:
            assert isinstance(r, RPN)
            print(r.get_list_of_varnames())

            lev_to_field = r.get_2D_field_on_all_levels(name=vname)

            lons_s, lats_s = r.get_longitudes_and_latitudes_for_the_last_read_rec()

            xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons_s.flatten(), lats_s.flatten())
            xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons_t.flatten(), lats_t.flatten())

            ktree = KDTree(list(zip(xs, ys, zs)))


            dists, inds = ktree.query(list(zip(xt, yt, zt)))


            field_3d = np.asarray([lev_to_field[lev] for lev in lev_to_field])

            selected_fields["BME_min"] = field_3d.min(axis=0)
            selected_fields["BME_max"] = field_3d.max(axis=0)



            for field_key, the_field in selected_fields.items():

                # Do the plotting
                fig = plt.figure()

                ax = plt.gca()
                ax.set_title(selected_fields_titles[field_key] + " {}".format(ds_label))

                to_plot = the_field.flatten()[inds].reshape(xx.shape)

                to_plot *= multiplier

                to_plot = maskoceans(np.where(lons_t < 180, lons_t, lons_t - 360), lats_t, to_plot)

                cs = bmap.contourf(xx, yy, to_plot, levels=clevels, ax=ax)
                bmap.colorbar(cs)


                bmap.drawcoastlines(ax=ax)
                bmap.drawcountries()
                bmap.drawstates()

                # read the relevant shape files
                for i, (clr, shp) in enumerate(shape_files.items()):
                    bmap.readshapefile(shp[:-4], "field_{}".format(i), color=clr)

                img_path = img_folder.joinpath("{}_{}.png".format(field_key, ds_label))
                fig.tight_layout()
                fig.savefig(str(img_path), bbox_inches="tight", transparent=True)
                plt.close(fig)





def plot_accumulation_area_and_glaciers_for_selected_basin(basin_shp="/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/lat_lon/fraizer/fraizer.shp",
                                                           polygon_name=None,
                                                           hints=None):


    route_data_path = "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/geophys_CORDEX_NA_0.11deg_695x680_filled_grDes_barBor_Crop2Gras_peat_with_directions"

    lons, lats, basemap = get_lons_lats_basemap(route_data_path, varname="FACC")

    basin_mask = get_mask(lons2d=lons, lats2d=lats, shp_path=basin_shp,
                          polygon_name=polygon_name,
                          hints=hints)

    i_arr, j_arr = np.where(basin_mask)

    assert basin_mask.sum() > 0


    plt.figure()
    plt.pcolormesh(basin_mask.T)
    plt.show()


    i_min, i_max = i_arr.min() - 25, i_arr.max() + 5
    j_min, j_max = j_arr.min() - 5, j_arr.max() + 5

    i_min = max(0, i_min)
    i_max = min(i_max, lons.shape[0] - 1)

    j_min = max(0, j_min)
    j_max = min(j_max, lons.shape[1] - 1)

    lons_target, lats_target = lons[i_min: i_max + 1, j_min: j_max + 1], lats[i_min: i_max + 1, j_min: j_max + 1]

    plot_acc_area_with_glaciers(gmask_vname="VF", gmask_level=2,
                                gmask_path="/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/fields_from_Caio/WC011_VF2.rpn",
                                route_data_path="/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/geophys_CORDEX_NA_0.11deg_695x680_filled_grDes_barBor_Crop2Gras_peat_with_directions",
                                lons_target=lons_target, lats_target=lats_target,
                                basin_shape_files=[basin_shp, ])




@main_decorator
def main():


    plot_glaciers()
    # plot_number_of_bins()
    # plot_min_and_max_bin_elevations()



    #depth to bedrock
    field_name = "DPTH"
    data_source = {
        "(ORNLDAAC)": "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/fields_from_Bernardo/DPTH_WCAN.rpn"
    }
    levels = [0, ]
    # plot_field(field_name=field_name, data_source=data_source, levels=levels)


    # Sand and Clay
    data_source = {
        "": "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/fields_from_Bernardo/HWSD_WCAN_011_OM.rpn"
    }
    field_name = "SAND"
    levels = [1, 2, 3]
    # plot_field(field_name=field_name, data_source=data_source, levels=levels)

    field_name = "CLAY"
    # plot_field(field_name=field_name, data_source=data_source, levels=levels)

    field_name = "VF"
    levels = range(1, 25)
    data_source = {
        "(MODIS)": "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/fields_from_Bernardo/VF_WCAN.rpn"
    }
    # plot_field(field_name=field_name, data_source=data_source, levels=levels)


    field_name = "FACC"
    data_source = {
        "(HydroSHEDS)": "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/geophys_CORDEX_NA_0.11deg_695x680_filled_grDes_barBor_Crop2Gras_peat_with_directions",
    }
    levels = [1, ]
    # plot_field(field_name=field_name, data_source=data_source, levels=levels)



    field_name = "ME"
    data_source = {
        "": "/RESCUE/skynet3_rech1/huziy/NEI_geophysics/WC_0.11_deg/geophys_CORDEX_NA_0.11deg_695x680_filled_grDes_barBor_Crop2Gras_peat_with_directions",
    }
    levels = [2, ]
    # plot_field(field_name=field_name, data_source=data_source, levels=levels)





    # directions with glaciers -for Frazer
    plot_accumulation_area_and_glaciers_for_selected_basin(
        # basin_shp="/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/GRDC_basins/GRDC_405_basins_from_mouth.shp",
        basin_shp="/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/lat_lon/fraser/fraser.shp",
        polygon_name=None,
        hints=None
    )


if __name__ == '__main__':
    main()