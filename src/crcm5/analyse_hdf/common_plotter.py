from util import plot_utils

__author__ = 'huziy'


#this is intended as a common launcher for plotting graphs, in hope that
#all graphs will have the same parameters


from multiprocessing import Process

def configure():
    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=17, height_cm=20)
    import application_properties
    application_properties.set_current_directory()


def compare_simulations():
    import compare_modelled_2d_fields
    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=17, height_cm=15)
    p = Process(target = compare_modelled_2d_fields.main)
    p.start()

def compare_obs_and_model_at_points():
    import compare_streamflow_with_obs
    p = Process(target=compare_streamflow_with_obs.main)
    p.start()

def plot_static_fields():
    import plot_static_fields
    p = Process(target=plot_static_fields.main)
    p.start()


def validate_seasonal_mean_atm_fields():
    plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=17, height_cm=15)
    import validate_model_fields
    p = Process(target=validate_model_fields.do_4_seasons, kwargs=dict(
        start_year = 1980, end_year = 1988))
    p.start()

if __name__ == "__main__":
    configure()
    #compare_obs_and_model_at_points()
    #compare_simulations()
    plot_static_fields()
    #validate_seasonal_mean_atm_fields()

