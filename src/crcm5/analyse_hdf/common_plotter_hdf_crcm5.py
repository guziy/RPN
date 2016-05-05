from collections import OrderedDict
from datetime import datetime
from util import plot_utils

__author__ = 'huziy'


# this is intended as a common launcher for plotting graphs, in hope that
# all graphs will have the same parameters


from multiprocessing import Process


def configure():
    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=17, height_cm=20)
    import application_properties

    application_properties.set_current_directory()


def explore_seasonal_interflow():
    from . import explore_interflow_field

    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=17, height_cm=17)
    # hdf_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_do_not_discard_small.hdf"
    # hdf_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_sani-10000.hdf"
    hdf_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_sani-10000_not_care_about_thfc.hdf"
    explore_interflow_field.calculate_and_plot_seasonal_means(
        hdf_path=hdf_path,
        start_year=1979,
        end_year=1980)


def compare_simulations():
    from crcm5.analyse_hdf import compare_modelled_2d_fields

    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=34, height_cm=30)
    p = Process(target=compare_modelled_2d_fields.main)
    p.start()


def compare_simulations_differences():
    """
    Plot changes for different variables for different seasons for 2 simulations

    """
    from crcm5.analyse_hdf import compare_modelled_2d_fields_increments_only as comp_incrs


    # varnames = ["STFA", "TT", "PR", "AV", "AH", "TRAF", "TDRA", "I5", "IMAV", "I0", "I1", "AS"]

    # varnames = ["TT", "TRAF", "I1", "PR", "TDRA", "AV"]

    # varnames = ["AV", "AH", "TT", "I1", "I0", "PR", "TRAF", "STFA", "STFL", "AS", "IMAV"]
    # varnames = ["TT", "TRAF"]

    varnames = ["STFA", "TRAF", "I1", "PR", "TDRA", "AV"]
    levels = len(varnames) * [0, ]

    # varnames = ["AS", "STFA", "AV", "AH", "I0", "I1", "TT", "PR"]
    # levels = [None, None, None, None, 1, 1, None, None]

    # Used to plot control and differences

    season_to_months = OrderedDict([
        ("Winter", [12, 1, 2]),
        ("Spring", list(range(3, 6))),
        ("Summer", list(range(6, 9))),
        ("Fall", list(range(9, 12)))
    ])

    # season_to_months = OrderedDict([
    # ("May", [5, ]),
    # ("June", [6, ]),
    #     ("July", [7, ]),
    #     ("August", [8, ])
    # ])

    start_year = 1991
    end_year = 2010

    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=20, height_cm=30)
    comp_incrs.plot_control_and_differences_in_one_panel_for_all_seasons_for_all_vars(varnames=varnames,
                                                                                      levels=levels,
                                                                                      season_to_months=season_to_months,
                                                                                      start_year=start_year,
                                                                                      end_year=end_year)


def compare_obs_and_model_at_points():
    plot_utils.apply_plot_params(font_size=14, width_pt=None, width_cm=20)
    from crcm5.analyse_hdf import compare_streamflow_with_obs

    start_date = datetime(1980, 1, 1)
    end_date = datetime(2010, 12, 31)

    # compare_streamflow_with_obs.main(start_date=start_date, end_date=end_date)
    compare_streamflow_with_obs.main_for_cc_paper(start_date=start_date, end_date=end_date)
    # reset back plot parameters to the default ones
    configure()


def compare_obs_and_model_at_river_outlet_points():
    from crcm5.analyse_hdf import compare_streamflow_with_obs

    compare_streamflow_with_obs.point_comparisons_at_outlets()


def do_plot_static_fields():
    from crcm5.analyse_hdf import plot_static_fields

    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=20, height_cm=17)
    p = Process(target=plot_static_fields.main)
    p.start()


def plot_vertical_soil_moisture_cross_section():
    from crcm5.analyse_hdf import compare_soil_moisture_profiles_upstream_of_stations as profiles

    profiles.main(start_year=1980, end_year=1989)


def plot_static_fields_histograms():
    from crcm5.analyse_hdf import plot_static_fields

    plot_static_fields.plot_histograms(
        path="/home/huziy/skynet3_rech1/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_do_not_discard_small.hdf")


def compare_2d_seasonal_means_from_simulations():
    from crcm5.analyse_hdf import compare_modelled_2d_fields as comp

    # varnames = ["STFA", "TT", "PR", "AV", "AH", "TRAF", "TDRA", "I5", "IMAV", "I0", "I1", "AS"]

    # varnames = ["AB", "AU", "AR", "AI", "AD"]

    # varnames = ["AD-AI", "AD", "AR"]
    # varnames = ["STFA"]


    # varnames = ["RH", "AV", "AH", "TT", "I1", "I0", "PR", "TRAF", "STFA", "STFL", "AS", "IMAV"]
    # varnames = ["TT", "TRAF"]
    # levels = len(varnames) * [0, ]

    # varnames = ["AD", "STFA", "AV", "AH", "I0", "I1", "TT", "PR", "TRAF"]
    # These are actually 0-based level indices
    # levels = [None, None, None, None, 1, 1, None, None, 0]

    varnames = ["STFA", ]
    levels = [0, ]

    # Used to plot control and differences

    season_to_months = OrderedDict([
        ("Winter", [12, 1, 2]),
        ("Spring", list(range(3, 6))),
        ("Summer", list(range(6, 9))),
        ("Fall", list(range(9, 12)))
    ])

    # season_to_months = OrderedDict([
    # ("May", [5, ]),
    # ("June", [6, ]),
    #     ("July", [7, ]),
    #     ("August", [8, ])
    # ])

    start_year = 1991
    end_year = 2010

    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=20, height_cm=10)
    comp.plot_control_and_differences_in_one_panel_for_all_seasons(varnames=varnames,
                                                                   levels=levels,
                                                                   season_to_months=season_to_months,
                                                                   start_year=start_year,
                                                                   end_year=end_year)


def validate_seasonal_mean_atm_fields():
    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=20, height_cm=20)
    from crcm5.analyse_hdf import validate_model_fields

    p = Process(target=validate_model_fields.do_4_seasons, kwargs=dict(
        start_year=1980, end_year=2010))
    p.start()


def plot_soil_profiles():
    from . import plot_soil_profile_in_time_mean_for_region

    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=20, height_cm=10)
    plot_soil_profile_in_time_mean_for_region.exp_plot_one_simulation()


def compare_quantiles():
    from crcm5.analyse_hdf import lake_effect_on_streamflow_quantiles as lkeff

    plot_utils.apply_plot_params(font_size=14, width_pt=None, width_cm=20, height_cm=7)
    # plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=24, height_cm=12)
    lkeff.main()


def compare_obs_and_model_lake_levels_at_points():
    from crcm5.analyse_hdf import compare_lake_levels_with_obs

    plot_utils.apply_plot_params(font_size=14, width_pt=None, width_cm=25, height_cm=18)

    start_date = datetime(1980, 1, 1)
    end_date = datetime(2010, 12, 31)

    compare_lake_levels_with_obs.main(start_date=start_date, end_date=end_date)
    configure()


def plot_veg_fractions_for_a_random_run():
    """
    Used to only once plot data for a random (even not mine) run

    """
    from rpn_utils import plot_veg_fractions

    b1, lons1, lats1, data1, label1 = plot_veg_fractions.main()
    b2, lons2, lats2, data2, label2 = plot_veg_fractions.main(
        base_folder="/home/huziy/skynet3_rech1/veg_fractions/CCCma",
        fname="an_geop_gemgcm_zer40.rpn",
        label="CCCma", canopy_name="FCAN", depth_to_bedrock_name="DPTH")

    plot_veg_fractions.plot_depth_to_bedrock(b1,
                                             lons1, lats1, data1["BDRCK_DEPTH"], "USGS",
                                             lons2, lats2, data2["BDRCK_DEPTH"], "CCCma")
    del data1["BDRCK_DEPTH"], data2["BDRCK_DEPTH"]

    plot_veg_fractions.plot_difference(b1, lons1, lats1, data1, "USGS", lons2, lats2, data2, "CCCma")


def plot_diff_in_soil_profiles():
    from crcm5.analyse_hdf import plot_soil_profile_in_time_mean_for_region

    plot_utils.apply_plot_params(font_size=7, width_pt=None, width_cm=20, height_cm=10)
    plot_soil_profile_in_time_mean_for_region.main_compare_two_simulations()


if __name__ == "__main__":
    import time

    t0 = time.clock()

    import matplotlib 
    matplotlib.use("Agg")   
 
    configure()
    #
    # do_plot_static_fields()
    compare_2d_seasonal_means_from_simulations()

    # Compare observed and modelled streamflow and upstream caracteristics for streamflow gauging stations
    #compare_obs_and_model_at_points()

    # compare_simulations()
    # validate_seasonal_mean_atm_fields()

    # plot_static_fields_histograms()

    # Compares Q10 and Q90 calculated from observed and modelled climatologic hydrographs
    # compare_quantiles()

    # plot_vertical_soil_moisture_cross_section()

    # explore_seasonal_interflow()

    # Compare observed and modelled streamflow and upstream caracteristics for river outlets
    # compare_obs_and_model_at_river_outlet_points()


    # Compare modelled and observed lake level anomalies at points
    # compare_obs_and_model_lake_levels_at_points()

    # plot_veg_fractions_for_a_random_run()

    # plot soil profiles
    # plot_soil_profiles()

    # plot differences in soil profiles between 2 simulations
    # plot_diff_in_soil_profiles()

    # Seasonal mean differences for all variablesbetween 2 simulations
    # compare_simulations_differences()

    print("Execution time: {0} seconds".format(time.clock() - t0))
