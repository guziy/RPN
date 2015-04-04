from netCDF4 import Dataset
import os
import Ngl
from . import draw_regions

__author__ = 'huziy'

import numpy as np

def main():
    start_year = 1981
    end_year = 2008

    #mean alt
    path_to_yearly = "alt_era_b1_yearly.nc"
    ds = Dataset(path_to_yearly)

    hm = ds.variables["alt"][:]
    years = ds.variables["year"][:]
    years_sel = np.where(( start_year <= years ) & (years <= end_year))[0]
    print(years_sel)

    hm = hm[np.array(years_sel),:,:]
    print(hm.shape)

    good_points = ~np.any(hm < 0, axis = 0)

    hm2d = np.ma.masked_all(good_points.shape)


    hm2d[good_points] = np.mean( hm[ : , good_points],
                        axis = 0)


    #alt from climatology
    sim_data_folder = "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/era40_driven_b1"
#    dm = CRCMDataManager(data_folder=sim_data_folder)
#    hc = dm.get_alt_using_monthly_mean_climatology(xrange(start_year,end_year+1))



    coord_file = os.path.join(sim_data_folder, "pmNorthAmerica_0.44deg_ERA40-Int_B1_200812_moyenne")
    basemap, lons2d, lats2d = draw_regions.get_basemap_and_coords(resolution="c",
        file_path = coord_file#, llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-10
    )

    #basemap.transform_scalar()

    #basemap = Basemap()
    lons2d[lons2d > 180] -= 360

    x, y = basemap(lons2d, lats2d)
    print(x.min(), x.max())
    permafrost_mask = draw_regions.get_permafrost_mask(lons2d, lats2d)
    mask_cond = (permafrost_mask <= 0) | (permafrost_mask >= 3)

    #plot_utils.apply_plot_params(width_pt=None, width_cm=25,height_cm=35, font_size=12)
    #fig = plt.figure()
    #assert isinstance(fig, Figure)


    h_max = 10
    #cmap = cm.get_cmap("jet",10)
    #cmap.set_over(cmap(1.0))
    clevels = np.arange(0,h_max+1,1)
    #gs = gridspec.GridSpec(1,1)

    all_axes = []
    all_img = []


    #ax = fig.add_subplot(gs[0,0])
    hm2d = np.ma.masked_where(mask_cond | (hm2d > h_max), hm2d)



    wks_res = Ngl.Resources()

    assert isinstance(wks_res, Ngl.Resources)
    wks = Ngl.open_wks("x11", "ALT mean", wks_res)


    wks_res.cnFillOn = True


    wks_res.sfXArray = lons2d
    wks_res.sfYArray = lats2d
    #---Create contours over map.
    map = Ngl.contour_map(wks,hm2d,wks_res)

    #---Draw plot and advance frame. MRB outlines will be included.
    Ngl.draw(map)
    Ngl.frame(wks)



if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print("Hello world")
  