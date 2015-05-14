__author__ = 'huziy'

if __name__ == '__main__':
    import crcm5.analyse_hdf.climate_change.plot_performance_err_with_cru as pe_cru
    pe_cru.main_wrapper()

    import crcm5.analyse_hdf.climate_change.plot_performance_err_with_udel as pe_udel
    pe_udel.main_wrapper()
