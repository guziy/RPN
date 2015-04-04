__author__ = 'huziy'


class ScoreTable(object):
    #A ScoreTable per station
    def __init__(self):
        self.sim_to_var_to_ns_score = {}
        self.station = None
        pass

    def set_ns_score(self, sim_label, var_name, score_value):
        if not len(self.sim_to_var_to_ns_score):
            self.sim_to_var_to_ns_score[sim_label] = {}

        self.sim_to_var_to_ns_score[sim_label][var_name] = score_value

    def print_score_table(self):
        for sim_label, var_to_ns_score in self.sim_to_var_to_ns_score.items():
            for var_name, ns_score in var_to_ns_score.items():
                print("NS: {0}\t{1}\t{2}".format(sim_label, var_name, ns_score))