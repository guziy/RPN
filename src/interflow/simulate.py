__author__ = 'huziy'


#Offline interflow simulation of a selected gridpoint


class Params(object):
    """
    An object for storing element parameters
    """

    slope = None
    depth_to_bedrock = None
    time_step = None
    drainage_density = None

    #Volumetric fractions
    theta_porosity = None
    theta_liq_initial = None
    theta_liq_field_capacity = None


    #hydraulic conductivity
    k_hyd_cond_top = None
    k_hyd_exp_coef = 0.5

    def __init__(self):
        pass


    def initialize(self):
        self.depth_to_bedrock = 1



def simulate():
    pass


if __name__ == "__main__":
    simulate()