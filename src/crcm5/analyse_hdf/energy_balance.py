from collections import OrderedDict

__author__ = 'huziy'

season_to_months = OrderedDict([
    ("Winter", [12, 1, 2]),
    ("Spring", list(range(3, 6))),
    ("Summer", list(range(6, 9))),
    ("Fall", list(range(9, 12)))
])


def compare_sensible_and_latent_heat_fluxes():
    pass


def main():
    import application_properties
    application_properties.set_current_directory()




if __name__ == '__main__':
    main()