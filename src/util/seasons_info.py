from collections import OrderedDict

__author__ = 'huziy'

DEFAULT_SEASON_TO_MONTHS = OrderedDict([
    ("Winter", (1, 2, 12)),
    ("Spring", range(3, 6)),
    ("Summer", range(6, 9)),
    ("Fall", range(9, 12)),
])