from nose.tools import ok_

__author__ = 'huziy'

from util import scores

import numpy as np
from numpy import random


def test_nash_sutcliff():
    """
    test nash sutcliff

    """
    x = random.randn(10)
    ok_(scores.nash_sutcliffe(x, x) == 1)
    ok_(scores.nash_sutcliffe(x, 5 * x) > 0)
    ok_(scores.nash_sutcliffe(x, -x) < 0)


def test_corr_coef():
    x = random.randn(10)
    print(scores.corr_coef(x, x))
    print(scores.corr_coef(x, -x))

    ok_(scores.corr_coef(x, x) == 1)

    ok_(scores.corr_coef(x, 5 * x) > 0)
    ok_(scores.corr_coef(x, -x) < 0)



if __name__ == "__main__":
    test_nash_sutcliff()
    test_corr_coef()