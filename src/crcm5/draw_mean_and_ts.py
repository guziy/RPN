from netCDF4 import Dataset
from crcm5.model_data import Crcm5ModelDataManager
from crcm5.timeseries_plotter import TimeseriesPlotter
from rpn import level_kinds
from rpn.rpn import RPN

__author__ = 'huziy'

import numpy as np

def main():
    import matplotlib.pyplot as plt
    #path = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/test_with_lakeroff"

    #path = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-rl-intfl_spinup"
    path = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-r/all_in_one_folder"
    manager = Crcm5ModelDataManager(
        samples_folder_path=path, file_name_prefix="pm",
        all_files_in_samples_folder=True,
        need_cell_manager=True
    )

    allData = {}
    field2d, allStfl = manager.get_mean_2d_field_and_all_data(var_name="STFL")

    allData["FACC"] = manager.accumulation_area_km2

    traf2d, allTraf = manager.get_mean_2d_field_and_all_data(var_name="TRAF", level=5,
            level_kind= level_kinds.ARBITRARY)
    tdra2d, allTdra = manager.get_mean_2d_field_and_all_data(var_name="TDRA", level=5,
            level_kind= level_kinds.ARBITRARY)


    swe2d, allSwe = manager.get_mean_2d_field_and_all_data(var_name="I5", level=-1,
                level_kind= level_kinds.ARBITRARY)

    pre2d, allPr = manager.get_mean_2d_field_and_all_data(var_name="PR")


    swsr2d,allSwsr = manager.get_mean_2d_field_and_all_data(var_name="SWSR")
    swsl2d,allSwsl = manager.get_mean_2d_field_and_all_data(var_name="SWSL")
    upin2d,allUpin = manager.get_mean_2d_field_and_all_data(var_name="UPIN")
    #gwdi2d,allGwdi = manager.get_mean_2d_field_and_all_data(var_name="GWDI")
    swst2d,allSwst = manager.get_mean_2d_field_and_all_data(var_name="SWST")


    for k,v in allTraf.items():
        allTraf[k] = v * manager.cell_area * 1.0e-3
    for k,v in allTdra.items():
        allTdra[k] = v * manager.cell_area * 1.0e-3

    for k,v in allPr.items():
        allPr[k] = v * manager.cell_area

    #for k,v in allImav.iteritems():
    #    allImav[k] = v * manager.cell_area * 1.0e-3


    for k,v in allSwe.items():
        allSwe[k] *= v * manager.cell_area * 1.0e-3 * 1e-6




    allData["TRAF"] = allTraf
    allData["TDRA"] = allTdra
    allData["PR"] = allPr
    allData["I5"] = allSwe

    allData["STFL"] = allStfl
    allData["SWSR"] = allSwsr
    allData["SWSL"] = allSwsl
    allData["UPIN"] = allUpin
    #allData["GWDI"] = allGwdi
    allData["SWST"] = allSwst




    #read in slope and channel length from the geophysics file
    r = RPN("/home/huziy/skynet3_rech1/geof_lake_infl_exp/geophys_Quebec_0.1deg_260x260_with_dd_v6")


    slp = r.get_first_record_for_name("SLOP")[20:-20,20:-20]
    slp[slp < 1e-4] = 1e-4

    allData["SLOP"] = slp
    allData["LENG"] = r.get_first_record_for_name("LENG")[20:-20,20:-20]
    allData["LKOU"] = r.get_first_record_for_name("LKOU")[20:-20,20:-20]
    r.close()




    basemap = manager.get_omerc_basemap(resolution="c")

    lon, lat = manager.lons2D, manager.lats2D

    fig = plt.figure()
    x, y = basemap(lon, lat)
    field2d = np.ma.masked_where(field2d <= 0.1, field2d)
    #field2d = np.ma.masked_where((field2d < 10000) | (upin2d > 100000), field2d)

    basemap.pcolormesh(x, y, field2d)
    plt.colorbar()
    basemap.drawcoastlines()







    ax = plt.gca()
    TimeseriesPlotter(name_to_date_to_field = allData, basemap = basemap, lons2d=lon, lats2d= lat,
        ax = ax, cell_manager = manager.cell_manager, data_manager=manager)






    fig = plt.figure()
    x, y = basemap(lon, lat)




    fig = plt.figure()
    x, y = basemap(lon, lat)
    basemap.pcolormesh(x, y, traf2d)
    plt.title("Total traf")
    plt.colorbar()
    basemap.drawcoastlines()



    fig = plt.figure()
    x, y = basemap(lon, lat)
    basemap.pcolormesh(x, y, swst2d)
    plt.title("SWST")
    plt.colorbar()
    basemap.drawcoastlines()



    traf2dsoil, dummy  = manager.get_mean_2d_field_and_all_data(var_name="TRAF", level=1,
            level_kind= level_kinds.ARBITRARY)

    traf2dlake, dummy = manager.get_mean_2d_field_and_all_data(var_name="TRAF", level=6,
            level_kind= level_kinds.ARBITRARY)





    fig = plt.figure()
    x, y = basemap(lon, lat)
    basemap.pcolormesh(x, y, upin2d)
    plt.title("UPIN")
    plt.colorbar()
    basemap.drawcoastlines()



    plt.show()
    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print("Hello world")
  
