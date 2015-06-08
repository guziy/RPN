__author__ = 'huziy'


class RunConfig(object):
    def __init__(self, data_path="", start_year=None, end_year=None, label=""):
        self.data_path = data_path
        self.start_year = start_year
        self.end_year = end_year
        self.label = label


    def get_shifted_config(self, shift_years):

        sy = self.start_year + shift_years
        ey = self.end_year + shift_years
        path = self.data_path.replace(str(self.start_year), str(sy)).replace(str(self.end_year), str(ey))
        return RunConfig(data_path=path, start_year=sy, end_year=ey, label=self.label)


    def __str__(self):
        return "{}, {}-{}: {}".format(self.label, self.start_year, self.end_year, self.data_path)


    def __hash__(self):
        # return super(RunConfig, self).__hash__()
        return self.__str__().__hash__()


    def __eq__(self, other):
        if other is None:
            return False

        if not isinstance(other, RunConfig):
            return False

        if self.__hash__() == other.__hash__():
            b = all((self.data_path == other.data_path,
                     self.start_year == other.start_year,
                     self.end_year == other.end_year,
                     self.label == other.label))
            return b
        else:
            return False

    def get_sim_id(self):
        """

        Get short, but still hopefully unique, simulation identificator
        :return:
        """
        return "{}_{}-{}".format(self.label, self.start_year, self.end_year)