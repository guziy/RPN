from collections import OrderedDict

__author__ = 'huziy'


class ExtremeProperties(object):
    seed = 10

    # Make it small for testing
    nbootstrap = 1000

    low = "low"
    high = "high"
    extreme_types = [high, low]

    extreme_type_to_return_periods = OrderedDict([
        ("high", [10, 50]),
        ("low", [2, 5]),
    ])

    extreme_type_to_month_of_interest = OrderedDict([
        ("high", range(3, 7)),
        ("low", range(1, 6)),
    ])

    extreme_type_to_n_agv_days = OrderedDict([
        ("high", 1),
        ("low", 15),
    ])

    def __init__(self, ret_lev_dict=None, std_dict=None):
        self.return_lev_dict = ret_lev_dict if ret_lev_dict is not None else {}
        self.std_dict = std_dict if std_dict is not None else {}

    def get_low_rl_for_period(self, return_period=2):
        return self.return_lev_dict[self.low][return_period]

    def get_high_rl_for_period(self, return_period=10):
        return self.return_lev_dict[self.high][return_period]

    def get_rl_and_std(self, ex_type=high, return_period=10):
        """
        Return level along with the standard deviation calculated
        using bootstrap
        :param ex_type:
        :param return_period:
        :return:
        """
        return [z[ex_type][return_period] for z in (self.return_lev_dict, self.std_dict)]

    def __str__(self):
        s = ""
        for et in self.extreme_types:
            s += et + ", periods:\n\t{}\n".format(",".join([str(t) for t in self.return_lev_dict[et].keys()]))
        return s
