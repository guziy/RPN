__author__ = "huziy"
__date__ = "$22 oct. 2010 12:00:55$"

import os
import pickle
from math import *

import numpy as np
import matplotlib.pyplot as plt

# from osgeo import gdal, ogr
import scipy.optimize as opt

from scipy.special import gamma
from datetime import datetime

from datetime import timedelta

# replaces lmoments package (used for initial guess and for validadion of the fitting with
# maximum likelihood)
import lmoments3

inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio
fig_width = 2000 * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]

font_size = 25


def zoom_to_qc():
    ymin, ymax = plt.ylim()
    plt.ylim(ymin + 0.05 * (ymax - ymin), ymax * 0.25)

    xmin, xmax = plt.xlim()
    plt.xlim(xmin + (xmax - xmin) * 0.55, 0.72 * xmax)


BIG_NUM = 1.0e6



def get_return_level_for_type_and_period(pars, return_period, extreme_type="high"):
    # sigma, mu, ksi, zero_fraction = pars
    # (i.e. as should be returned by the optimize_stationary_for_period function)

    assert len(pars) == 4

    if extreme_type.lower() == "high":
        get_high_ret_level_stationary(pars, return_period)
    else:
        get_low_ret_level_stationary(pars, return_period)


def get_high_ret_level_stationary(pars, return_period):
    # sigma, mu, ksi, zero_fraction = pars

    if pars[0] is None:
        return -1
    sigma, mu, ksi, zero_fraction = pars

    y = np.log(float(return_period) / (float(return_period) - 1.0))
    if np.abs(ksi) < 1.0e-5:
        lev = -sigma * np.log(y) + mu
    else:
        lev = sigma / ksi * (np.power(y, -ksi) - 1.0) + mu
    return lev


# sigma, mu, ksi, zero_fraction = pars
def get_low_ret_level_stationary(pars, return_period):
    return get_low_ret_level(params=pars[0:3], return_period=return_period,
                             zero_fraction=pars[3])


# rlevel = sigma/ksi * (ln(T/(1-Tz))^(-ksi) - 1) + mu
def get_low_ret_level(params=[], return_period=2, zero_fraction=0.0):
    if 1.0 / return_period <= zero_fraction:
        return 0

    if params[0] is None:
        return -1
    sigma, mu, ksi = params

    if np.abs(ksi) < 1.0e-2:
        lev = mu - np.log(np.log(return_period)) * sigma
    else:
        y = np.log(return_period * (1.0 - zero_fraction) / (1.0 - return_period * zero_fraction))
        lev = sigma / ksi * ( np.power(y, -ksi) - 1.0) + mu
    return lev


# Martins E.S. (2000)
def ksi_pdf(ksi):
    if abs(ksi) >= 0.5:
        return 0
    else:
        return 1  # temporary disable prior distribution function

    p = 6.0
    q = 9.0
    b = gamma(p) * gamma(q) / gamma(p + q)
    return (-ksi + 0.5) ** (p - 1) * (0.5 + ksi) ** (q - 1) / b


# Coles 1999
def ksi_pdf_coles(ksi):
    if ksi <= 0:
        return 1.0
    if ksi >= 1:
        return 0.0
    alpha = 1.0
    lam = 1.0
    return np.exp(-lam * (1.0 / (1.0 - ksi) - 1) ** alpha)


def qfunc(x, sigma, mu, ksi):
    """
    Helper function (1 + ksi*(x - mu) / sigma)^(-1/ksi)
    """
    if sigma <= 1.0e-10:  #sigma > 0
        return None

    if 1.0 + ksi * (x - mu) / sigma <= 0:
        return None

    if abs(ksi) <= 1.0e-5:  #ksi != 0
        the_power = -(x - mu) / sigma
        result = np.exp(the_power)
        assert result > 0, 'the_power = {0}, mu = {1}, sigma = {2}'.format(the_power, mu, sigma)
        return result

    the_base = 1.0 + ksi * (x - mu) / sigma
    result = the_base ** (-1.0 / ksi)

    if isinf(result) or result == 0:
        return None

    if not result:
        print(x, mu, sigma)
        print(the_base)
        print(-1.0 / ksi)

    message = 'in qfunc: result = {0}, x = {1}, sigma = {2}, mu = {3}, ksi = {4}, the_base = {5}'
    assert result > 0.0, message.format(result, x, sigma, mu, ksi, the_base)

    if isinf(result) or isnan(result):
        print('Too big numbers: ', the_base, result)
        assert False, 'qfunc = {0}'.format(result)
        return None
    return result


# -ln(gevpdf * ksi_pdf)
def objective_function_stationary_high(pars, data):
    result = 0.0
    sigma, mu, ksi = pars

    ksi_probability = ksi_pdf(ksi)

    if not ksi_probability:
        return BIG_NUM

    for the_data in data:
        qi = qfunc(the_data, sigma, mu, ksi)
        if qi is None:
            return BIG_NUM
        assert qi > 0, 'qi = {0}'.format(qi)

        minus_ln_pdfi = log(sigma) - (ksi + 1.0) * log(qi) + qi - log(ksi_probability)
        if minus_ln_pdfi < 0:
            return BIG_NUM
        result += minus_ln_pdfi

    assert np.isfinite(result), 'result is nan, result = {0}'.format(result)
    return result


# -ln(gevpdf* ksi_pdf)
def objective_function_stationary_low(pars, data):
    """
    objective function to minimize for stationary case
    """
    result = 0.0
    sigma, mu, ksi = pars

    ksi_probability = ksi_pdf(ksi)

    if not ksi_probability:
        return BIG_NUM

    for the_data in data:
        qi = qfunc(the_data, sigma, mu, ksi)
        if qi is None:
            return BIG_NUM
        assert qi > 0, 'qi = {0}'.format(qi)
        minus_ln_pdfi = np.log(sigma) - (ksi + 1.0) * log(qi) + qi - log(ksi_probability)
        if minus_ln_pdfi < 0:
            return BIG_NUM

        result += minus_ln_pdfi

    assert np.isfinite(result), 'result is nan, result = {0}'.format(result)
    return result


# vals timeseries for a point
def get_initial_params(vals):
    assert len(vals) > 0, 'len(vals) = {0}'.format(len(vals))
    ksi0 = 0.1

    if len(vals) == 1:
        bias = 1
    else:
        bias = 0

    sigma0 = np.sqrt(6.0 * np.cov(vals, bias=bias)) / pi

    if not sigma0:
        sigma0 = 0.2 * np.mean(vals)

    mu0 = np.mean(vals) - 0.57722 * sigma0

    assert np.isfinite(mu0), 'mu0 = {0}'.format(mu0)
    return [sigma0, mu0, ksi0]


# returns initial parameters using L-moments
def get_initial_params_using_lm(vals):
    from lmoments3 import distr
    sorted_vals = list(sorted(vals))

    the_moments = lmoments3.lmom_ratios(sorted_vals, nmom=3)
    ksi, mu, sigma = distr.gev.lmom_fit(sorted_vals, the_moments[0:3]).values()
    return [sigma, mu, -ksi]  # -ksi because they are using -ksi convention


# optimize using maxima over certain period
# returns [sigma, mu, ksi, zero_fraction]
def optimize_stationary_for_period(extremes, high_flow=True, use_lmoments=False):
    """

    """
    indices = np.where(extremes > 0)
    zero_fraction = 1.0 - extremes[indices].shape[0] / float(len(extremes))

    # if all values are 0, do not optimize, return None for the parametes values
    if zero_fraction >= 0.5:
        return [None, None, None, 1.0]



    # L-moments
    if use_lmoments:
        pars = get_initial_params_using_lm(extremes[indices])
        pars.append(zero_fraction)
        lev = get_high_ret_level_stationary(pars, 10.0)
        if isnan(lev):
            print(pars)
            print(extremes[indices].tolist())
            assert False, 'lev = {0}'.format(lev)
        return pars

    # multiply by a factor in order to eliminate 0 and negative return levels
    the_min = np.min(extremes[indices])
    if the_min < 100:
        factor = 100.0 / the_min
    else:
        factor = 1.0

    extremes *= factor

    if high_flow:
        objective_function = objective_function_stationary_high
    else:
        objective_function = objective_function_stationary_low

    pars0 = get_initial_params(extremes[indices])



    #    pars0 = get_initial_params_using_lm(extremes[indices])
    #    if objective_function(pars0, extremes[indices]) == BIG_NUM:
    #        pars0 = get_initial_params(extremes[indices])

    # default simplex
    pars, z, niter, funcalls, warnflag, all_vecs = opt.fmin(objective_function, pars0,
                                                            args=(extremes[indices],),
                                                            maxfun=10000,
                                                            full_output=True,
                                                            disp=False,
                                                            maxiter=10000,
                                                            retall=True)

    # powell method
    #    pars, z, direc, niter, funcalls, warnflag, all_vecs = opt.fmin_powell(objective_function,
    #                                                        pars0,
    #                                                        args = (extremes[indices],),
    #                                                        maxfun = 10000,
    #                                                        full_output = True,
    #                                                        disp = False,
    #                                                        maxiter = 10000,
    #                                                        retall = True
    #                                                        )


    if warnflag:
        print(list(extremes))
        print(warnflag)
        print(pars)
        assert False, 'warnflag != 0'


    # assert warnflag == 0, 'warnflag = {0}, z = {1}, \n extremes = {2}'.format(warnflag, z, str(extremes))
    assert z > 0, 'z <= 0'

    if z < 0:
        print('converged to negative objective function')
        return [None, None, None, zero_fraction]

    if z == BIG_NUM:
        print('high_flow = ', high_flow)
        print(extremes)
        print(extremes[indices].tolist())
        print(pars)
        print(all_vecs)
        #       assert False
        return [None, None, None, zero_fraction]

    assert z != BIG_NUM, 'z == BIG_NUM'
    assert z >= 0, 'z < 0'

    pars[0] /= factor
    pars[1] /= factor
    pars = np.append(pars, zero_fraction)
    extremes /= factor  # change back the extremes
    return pars


def optimize_stationary_for_period_and_all_cells_using_data(
        data=None,
        high_flow=True):
    """
    optimization for data and whole domain
    """
    pars_set = []
    # for all grid cells
    for pos in range(data.shape[1]):
        #        print pos
        #        print '-'*10
        #        print 'data'
        #        print list(data[:, pos])
        pars = optimize_stationary_for_period(data[:, pos], high_flow=high_flow)
        #        print 'pars = ', pars
        pars_set.append(pars)
    return pars_set


def optimize_stationary_for_period_and_all_cells(
        data_file='data/streamflows/hydrosheds_euler9/aex_discharge_1970_01_01_00_00.nc',
        paramfile='gev_params_stationary',
        high_flow=True,
        start_month=1, end_month=12,
        start_date=datetime(1970, 1, 1, 0, 0),
        end_date=datetime(1999, 12, 31, 0, 0),
        event_duration=timedelta(days=1)):
    print(paramfile)

    # check whether optimization is required
    if os.path.isfile(paramfile):
        print('already optimized, if you want to reoptimize delete %s' % paramfile)
        pars_set = pickle.load(open(paramfile))
        return pars_set

    # TODO: get streamflow data yearly min/max data
    data = None

    data = np.array(data).transpose()
    pars_set = optimize_stationary_for_period_and_all_cells_using_data(data=data,
                                                                       high_flow=high_flow)
    f = open(paramfile, 'w')
    pickle.dump(pars_set, f)
    f.close()
    return pars_set


def main():
    pass


def get_gevd_params_for_id_and_type(sim_id='', high=True):
    prefix = 'gev_params_stationary'
    postfix = '_high' if high else '_low'
    file = prefix + '_' + sim_id + postfix
    return pickle.load(open(file))


def get_levels_for_type_and_id(sim_id, return_period=None, extreme_type='high'):
    file_name_prefix = 'gev_params_stationary'
    if extreme_type == 'high':
        return get_high_levels_for_id(sim_id, file_name_prefix, postfix='_' + extreme_type, return_period=return_period)
    else:
        return get_low_levels_for_id(sim_id, file_name_prefix, postfix='_' + extreme_type, return_period=return_period)


def get_high_levels_for_id(sim_id, prefix='gev_params_stationary', postfix='_high', return_period=10):
    """

    Note: The start and the end years as well as months ofthe period of interest should be included
    either in prefix or postfix for consistency, or sim_id

    :param sim_id:
    :param prefix:
    :param postfix:
    :param return_period:
    :return:
    """
    file = prefix + '_' + sim_id + postfix
    print(file)
    pars_set = pickle.load(open(file))

    field = np.zeros((len(pars_set),))
    for pos, pars in enumerate(pars_set):
        field[pos] = get_high_ret_level_stationary(pars, return_period)
    return field


def get_low_levels_for_id(sim_id, prefix='gev_params_stationary', postfix='_low', return_period=10):
    file = prefix + '_' + sim_id + postfix
    pars_set = pickle.load(open(file))

    field = np.zeros((len(pars_set),))
    for pos, pars in enumerate(pars_set):
        field[pos] = get_low_ret_level_stationary(pars, return_period)
    return field


def test():
    pars = [8.4116509126033642e-05, 0.00084966170834377408, 0.10000000000000001]
    vals = [0.0010247972095385194, 0.0010247972095385194, 0.0012944934424012899,
            0.0042147189378738403, 0.00098561809863895178, 0.00095898169092833996,
            0.002480002585798502, 0.00084966170834377408, 0.0034388666972517967,
            0.0016178090590983629, 0.0013241175329312682, 0.0020841944497078657,
            0.001562419580295682, 0.0022000106982886791, 0.005726152565330267,
            0.0010590874589979649, 0.0014877116773277521, 0.0010104207322001457,
            0.0019218671368435025, 0.0030378694646060467, 0.0014164787717163563,
            0.00090275343973189592, 0.001988076139241457, 0.0026944593992084265,
            0.0033022623974829912, 0.0021143041085451841, 0.001547978725284338,
            0.0013833490666002035, 0.0042443717829883099, 0.0024236994795501232]

    print(BIG_NUM)


def test_lm():
    x = [360.228515625, 513.506103515625, 273.85031127929688, 340.94839477539062,
         244.13925170898438, 283.414306640625, 394.42819213867188, 284.3604736328125,
         281.26956176757812, 241.46173095703125, 489.75482177734375, 236.31536865234375,
         407.55133056640625, 244.6295166015625, 432.40670776367188, 260.501953125,
         517.23052978515625, 317.6553955078125, 407.61935424804688, 275.0709228515625,
         330.369140625, 285.92086791992188, 247.9954833984375, 344.34811401367188,
         379.55596923828125, 330.80569458007812, 312.35330200195312, 251.79550170898438,
         372.66928100585938, 239.72474670410156]

    #    print(get_initial_params_using_lm(x))
    print(np.mean(x))
    pars = [128.28104749, 578.4927539, 0.62410911]
    data = [588.4747314453125, 693.6640625, 519.03155517578125, 716.58013916015625,
            686.29168701171875, 432.65786743164062, 682.72113037109375, 730.12603759765625,
            698.971923828125, 491.75332641601562, 597.258544921875, 487.13619995117188, 482.33123779296875,
            573.57861328125, 801.67169189453125, 616.41668701171875, 690.954833984375, 671.31646728515625,
            680.87554931640625, 534.18414306640625, 427.86019897460938, 236.22953796386719, 691.40972900390625,
            599.84637451171875,
            545.3563232421875, 553.059814453125, 549.1295166015625, 658.3983154296875, 719.122802734375,
            636.84906005859375]

    import lmoments3
    from lmoments3 import distr

    the_moments = lmoments3.lmom_ratios(sorted(data), 5)
    pars = distr.gev.lmom_fit(sorted(data), lmom_ratios=the_moments)

    print("Fitted params using lmoments: ", pars)
    xi, mu, sigma = pars.values()
    print(objective_function_stationary_high([sigma, mu, -xi], data))

    print("Fitted using MLE: ", distr.gev.fit(sorted(data)))

    print("Fitted using custom method (Huziy et al 2013), not using l-moments: ", optimize_stationary_for_period(
        np.array(sorted(data))))
    print("Fitted using custom method (Huziy et al 2013), using l-moments: ",
          optimize_stationary_for_period(np.array(sorted(data)), use_lmoments=True))


    from scipy.stats import genextreme
    print("Fitted using scipy.stats.genextreme: ", genextreme.fit(np.array(sorted(data))))
    print("10 year high flow return level: ", get_high_ret_level_stationary([sigma, mu, -xi, 0], 10))
    print("10 year high flow return level: ", get_high_ret_level_stationary([sigma, mu, -0.5, 0], 10))


if __name__ == "__main__":
    #    fit_merged_for_current_and_future()
    # stationary()
    #    test_lm()

    test_lm()

    print("Hello World")
