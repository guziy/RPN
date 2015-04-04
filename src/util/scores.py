__author__ = 'huziy'

import numpy as np


def nash_sutcliffe(mod, obs):
    """
    E = 1 - sum((mod-obs)**2)/sum((obs-obs.mean())**2)
    """
    mod = np.array(mod)
    obs = np.array(obs)
    return 1 - sum((mod - obs) ** 2) / sum((obs - obs.mean()) ** 2)

def corr_coef(mod, obs):
    return np.corrcoef([mod, obs])


def main():
    #TODO: implement
    pass


if __name__ == "__main__":
    main()
    print("Hello world")
  