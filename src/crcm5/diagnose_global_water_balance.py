from datetime import datetime
from crcm5.model_data import Crcm5ModelDataManager
import matplotlib.pyplot as plt

__author__ = 'huziy'

import numpy as np

def main():
    #base_data_path =  "/home/huziy/skynet3_exec1/from_guillimin/new_outputs/quebec_86x86_0.5deg_wo_lakes_and_wo_lakeroff"
    base_data_path = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_lowres_005"
    base_data_manager = Crcm5ModelDataManager(samples_folder_path=base_data_path,
        all_files_in_samples_folder=True, need_cell_manager=True)


    basemap = base_data_manager.get_omerc_basemap()
    x, y = basemap(base_data_manager.lons2D, base_data_manager.lats2D)

    cellManager = base_data_manager.cell_manager

    rout_domain_mask = (base_data_manager.bankfull_storage_m3 > 0)
    rout_domain_mask = rout_domain_mask.astype(np.int)

    plt.figure()
    basemap.pcolormesh(x,y, rout_domain_mask)
    plt.title("rout domain mask (banfull storage > 0)")

    plt.figure()
    cbf = base_data_manager.cbf
    cbf = np.ma.masked_where(cbf < 0, cbf)
    basemap.pcolormesh(x,y, cbf)
    plt.colorbar()
    plt.title("rout domain mask (coef_bf > 0)")




    plt.figure()
    bfstore = np.ma.masked_where(base_data_manager.bankfull_storage_m3 < 0, base_data_manager.bankfull_storage_m3)
    basemap.pcolormesh(x,y, bfstore)
    plt.colorbar()
    basemap.drawcoastlines()
    plt.title("bankful storage")


    plt.figure()
    acc_area = base_data_manager.accumulation_area_km2
    basemap.pcolormesh(x, y, np.ma.masked_where(acc_area <= 0, acc_area))




    outlet_mask = cellManager.get_outlet_mask(rout_domain_mask=rout_domain_mask)
    rout_domain_mask = cellManager.get_coherent_rout_domain_mask(outlet_mask)

    rout_domain_mask = np.ma.masked_where(rout_domain_mask == 0, rout_domain_mask)
    plt.figure()
    basemap.pcolormesh(x,y, rout_domain_mask)
    plt.title("rout domain mask (coh)")



    ls_mask = base_data_manager.land_sea_mask



    plt.figure()
    basemap.pcolormesh(x, y, np.ma.masked_where(outlet_mask == 0, outlet_mask))
    basemap.drawcoastlines()
    #plt.show()

    start_year = 1986
    end_year = 1987

    dateo = datetime(1985,1,1)





    for lev in range(1,7):
        lksroff = base_data_manager.get_annual_mean_fields(start_year = start_year, end_year=end_year, varname="TRAF", level = lev)
        plt.figure()
        plt.title("lev = {0}".format(lev))
        basemap.pcolormesh(x, y, lksroff[1987])

    plt.show()





    ds = 0

    for year in range(start_year, end_year+1):
        sr_start = base_data_manager.get_2d_field_for_date(date = datetime(year,1,1), dateo=dateo,varname = "SWSR")
        sr_start = np.sum(sr_start * rout_domain_mask)

        sl_start = base_data_manager.get_2d_field_for_date(date = datetime(year,1,1),dateo=dateo,varname = "SWSL")
        sl_start = np.sum(sl_start * rout_domain_mask)

        sr_end = base_data_manager.get_2d_field_for_date(date = datetime(year+1,1,1),dateo=dateo,varname = "SWSR")
        sr_end = np.sum(sr_end * rout_domain_mask)

        sl_end = base_data_manager.get_2d_field_for_date(date = datetime(year+1,1,1),dateo=dateo,varname = "SWSL")
        sl_end = np.sum(sl_end * rout_domain_mask)


        sgw_start = base_data_manager.get_2d_field_for_date(date = datetime(year,1,1),dateo=dateo, varname="GWST")
        sgw_start = np.sum(sgw_start * rout_domain_mask)

        sgw_end = base_data_manager.get_2d_field_for_date(date = datetime(year+1, 1,1),dateo=dateo, varname="GWST")
        sgw_end = np.sum(sgw_end * rout_domain_mask)

        ds += (sl_end + sr_end + sgw_end) - (sl_start + sr_start + sgw_start)

        print "year = {0}".format(year)
        print "Sgw(start, end), dSgw = {0}, {1}, {2}".format(sgw_start, sgw_end, sgw_end - sgw_start)
        print "Slake(start, end), dSlake = {0}, {1}, {2}".format(sl_start, sl_end, sl_end - sl_start)
        print "Sriv(start, end) = {0}, {1}, {2}".format(sr_start, sr_end, sr_end - sr_start)


    ds /= float(end_year - start_year + 1)


    sroff = base_data_manager.get_annual_mean_fields(start_year = start_year, end_year=end_year, varname="TRAF", level = 5)
    subsroff = base_data_manager.get_annual_mean_fields(start_year = start_year, end_year=end_year, varname="TDRA", level = 5)

    stfl = base_data_manager.get_annual_mean_fields(start_year=start_year, end_year=end_year, varname="STFL")


    roff_sum = 0
    stfl_sum = 0
    for year in sroff.keys():
        stfl_sum += np.sum( outlet_mask * stfl[year] )
        roff_sum += np.sum( rout_domain_mask * (sroff[year] + subsroff[year]) * base_data_manager.cell_area * 1e-3)


    print "number of selected years = {0}".format(len(stfl))
    stfl_mean = stfl_sum / float(len(stfl))
    roff_mean = roff_sum / float(len(sroff))

    print "Total runoff (annual mean, m**3/s): {0:g}".format( roff_mean )
    print "Streamflow (annual mean, m**3/s): {0:g}".format( stfl_mean )
    print "stfl - runoff (annual mean, m**3/s): {0:g}".format( stfl_mean - roff_mean )

    #print "dS_riv = {0} m**3".format(sr_end - sr_start)
    #print "dS_lake = {0} m**3".format(sl_end - sl_start)

    print "dS actual = {0:g} m**3".format(ds)

    print "dS from balance = {0:g} m**3".format((roff_mean - stfl_mean) * 365 * 24 * 60 * 60)



    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print "Hello world"
  