from util import plot_utils

__author__ = 'huziy'


def validate_forcing_temp_and_precip_with_cru():
    """
    Compare DFS temperature (K) and precip, with the ones from the CRU data set
    """
    from forcing import compare_forcing_clim_with_CRU
    plot_utils.apply_plot_params(font_size=18, width_pt=None, width_cm=13, height_cm=25)
    compare_forcing_clim_with_CRU.main()
    compare_forcing_clim_with_CRU.main(dfs_var_name="precip",
                                       cru_var_name="pre",
                                       cru_file="data/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.pre.dat.nc")



def validate_mean_lake_sst_with_glsea():
    import compare_lake_averaged_sst_with_glsea
    configure()
    compare_lake_averaged_sst_with_glsea.main()


def plot_max_av_icecover():
    import plot_maximum_ice_cover
    plot_utils.apply_plot_params(font_size=20, width_pt=None, width_cm=20, height_cm=10)
    plot_maximum_ice_cover.main()
    configure()

def configure():
    plot_utils.apply_plot_params(font_size=20, width_pt=None, width_cm=20, height_cm=10)

def animate_flow():
    configure()
    import nemo_animate_flow_vectors
    nemo_animate_flow_vectors.main()


def main():
    #validate_forcing_temp_and_precip_with_cru()
    validate_mean_lake_sst_with_glsea()
    plot_max_av_icecover()
    #animate_flow()


if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()