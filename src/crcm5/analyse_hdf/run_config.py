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
        path = self.data_path.replace(str(self.start_year), str(sy)).replace(str(self.start_year), str(ey))
        return RunConfig(data_path=path, start_year=sy, end_year=ey, label=self.label)


    def __str__(self):
        return "{}, {}-{}: {}".format(self.label, self.start_year, self.end_year, self.data_path)
