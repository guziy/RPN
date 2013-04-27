from datetime import datetime, timedelta
import pickle
from matplotlib import cm

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt

def analytical(a,b, s, s0):
    """
    get analytical value for t(s)
    where t(s) is the solution to the equation ds/dt=-a*s**(4/3)+b
    """

    assert  (a > 0) and (b > 0)
    k = (a/b)**0.25
    alpha = 3.0 / (k**3 * b)
    v = k * s ** (1.0/3.0)
    v0 = k * s0 ** (1.0/3.0)
    return -alpha/2.0 * (np.arctan2(v, 1.0) - np.arctan2(v0, 1.0)) +  \
            0.25 * alpha * np.log(np.abs((1+v)*(1-v0) / ((1-v)*(1+v0))))


    pass


def get_eq_store_lake(ifw, lk_area):
    h0 = 5.0
    s0 = h0 * lk_area
    kr = 0.01 / (60.0*24*60.0)
    return (ifw * s0 ** 1.5 / kr) ** 0.4



def get_eq_store(ifw, sbf, cbf, cob):
    p = 4.0/3.0
    tmp = cbf * sbf ** p
    #print tmp
    return (ifw/cbf)**(1.0/p) if ifw <= tmp else sbf + ((ifw - tmp)/cob)**(1.0/p)


def Q(s, k_bf, k_ob, s_bf):
    if s <= s_bf:
        return k_bf * s ** (4.0/3.0)
    else:
        #print "here"
        return k_ob * (s - s_bf) ** (4.0/3.0) + \
               k_bf * s_bf ** (4.0/3.0)

def Q_lake(s, lk_area):
    h0 = 5.0
    s0 = h0 * lk_area
    kr = 0.01 / (60.0*24*60.0)
    return kr * s * (s/s0) ** 1.5



def rk4_l(s1, inflow, nsteps, dt, lk_area):
    si = s1
    s_eq = get_eq_store_lake(inflow, lk_area)
    #print(inflow, Q_lake(si, lk_area))
    for i in xrange(nsteps):

        k1 = dt * (inflow - Q_lake(si, lk_area))
        k2 = dt * (inflow - Q_lake(si+0.5*k1, lk_area))
        k3 = dt * (inflow - Q_lake(si+0.5*k2, lk_area))
        k4 = dt * (inflow - Q_lake(si+k3, lk_area))
        print k1,k2,k3,k4
        si += (1.0/6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


    if si < s1:
        si = max(si, s_eq)
    else:
        si = min(si, s_eq)

    return si

def rk4_r(s1, inflow, nsteps, dt, k_bf, k_ob, s_bf):
    si = s1
    s_next = -1

    s_eq = get_eq_store(inflow, s_bf, k_bf, k_ob)
    #print(inflow, Q(si, k_bf, k_ob, s_bf))
    k1 = k2 = k3 = k4 = -1
    for i in xrange(nsteps):
        k1 = k2 = k3 = k4 = -1
        s_next = -1
        k1 = dt * (inflow - Q(si, k_bf, k_ob, s_bf))

        tmp = si+0.5*k1
        if tmp <= 0: break
        k2 = dt * (inflow - Q(tmp, k_bf, k_ob, s_bf))

        tmp = si+0.5*k2
        if tmp <= 0: break
        k3 = dt * (inflow - Q(tmp, k_bf, k_ob, s_bf))

        tmp = si+k3
        if tmp <= 0: break
        k4 = dt * (inflow - Q(tmp, k_bf, k_ob, s_bf))

        si += (1.0/6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if si <= 0: break
        s_next = si


    if s_next < 0:
        if k1 * k2 * k3 * k4 <= 0:
            print k1,k2,k3,k4,si
        si = s_eq


    if si > s1:
        si = min(s_eq, si)
    else:
        si = max(s_eq, si)

    return si




def main():

    i = 64
    j = 102

    fName = "route_params_{0}_{1}.bin".format(i, j)

    info = pickle.load(open(fName))

    upin = info["UPIN"]
    traf = info["TRAF"]
    gwdi = info["GWDI"]

    stfl = info["STFL"]
    swsr = info["SWSR"]
    swsl = info["SWSL"]
    swst = info["SWST"]




    cbf = info["CBF"]
    sbfm = info["SBFM"]
    is_gl_outlet = info["LKOU"] == 1
    lake_fraction = info["LKFR"]
    lake_area = info["LKAR"]


    print is_gl_outlet, lake_fraction, lake_area

    start_date = datetime(1979,1,1)
    end_date = datetime(1979, 2, 28,23)
    dt = timedelta(seconds = 300)

    d = start_date
    stfl_list = [stfl[start_date]]
    swsr_list = [swsr[start_date]]
    swsl_list = [swsl[start_date]]


    the_upin = -1
    the_traf = -1
    the_gwdi = -1
    dates = [start_date]
    while d <= end_date:
        if d in upin.keys():
            the_upin = upin[d]
            the_traf = traf[d]
            the_gwdi = gwdi[d]


        assert the_upin >= 0


        res = step(inflow_loc= the_gwdi + the_traf, inflow_upstream= the_upin, dt0 = dt.seconds + dt.days * 24 * 60 * 60,
                        s_bf=sbfm, k_bf=cbf, s0_r=swsr_list[-1], s0_l=swsl_list[-1], is_gl_outlet= is_gl_outlet,
                        lk_fraction= lake_fraction, lk_area= lake_area
        )


        stfl_list.append(res[0])
        swsr_list.append(res[1])
        swsl_list.append(res[2])
        d += dt
        dates.append(d)


    plt.figure()
    plt.plot(dates, stfl_list, label = "STFL")


    dates1 = list(sorted( gwdi.keys() ))
    plt.plot(dates1, [gwdi[d] for d in dates1], label = "GWDI")
    plt.plot(dates1, [upin[d] for d in dates1], label = "UPIN")
    plt.plot(dates1, [traf[d] for d in dates1], label = "TRAF")
    plt.legend()

    plt.figure()
    plt.plot(dates1, [stfl[d] for d in dates1], label = "STFL(FOR)")
    plt.legend()

    plt.figure()
    plt.plot(dates, swsr_list, label = "SWSR")
    plt.plot(dates, swsl_list, label = "SWSL")
    plt.legend()

    plt.figure()
    plt.plot(dates1, [swst[d] for d in dates1], label = "SWST(FOR)")
    plt.plot(dates1, [swsr[d] for d in dates1], label = "SWSR(FOR)")
    plt.plot(dates1, [swsl[d] for d in dates1], label = "SWSL(FOR)")
    plt.legend()
    plt.show()




def step(inflow_loc =  8.995517, inflow_upstream = 0,
         dt0 = 300.0,
         s_bf = 2368261.,
         k_bf = 2.5176147E-03,
         s0_r =  467.9909,
         s0_l = 0,
         is_gl_outlet = False,
         acc =5e-3,
         lk_fraction = 0,
         lk_area = 0
         ):

    #streamflow was negative  -1.3249715E-05  so putting it to 0
    #storage_prev_rl(j), inflow_rl(j)*dt, storage2(j),types(j), all_params(j,:), outflow
    #Other:      99.99550       4.5013428461970761E-003    100.0040     river
    #Other:    3.2324451E-08   1.6162225E-08    1738320.

    #return [store_lake, store_river, streamflow]



    store_lake = -1
    store_river = -1
    streamflow = -1


    k_ob = k_bf / 2.0


    inflow_lake = 0
    inflow_river = 0





    if is_gl_outlet:
        inflow_lake = inflow_loc + inflow_upstream
        inflow_river = 0
    else:
        if 1e-4 <= lk_fraction <= 0.6:
            inflow_lake = inflow_loc
            inflow_river = inflow_upstream
        elif lk_fraction > 0.6:
            store_lake = s0_l
            store_river = s0_r
            return [inflow_loc + inflow_upstream,store_river,store_lake]



    dt = dt0
    nsteps = 1
    s2 = None
    while True:
        if lk_fraction <= 0.0001:break
        s1 = rk4_l(s0_l, inflow_lake, nsteps, dt, lk_area)
        nsteps *= 2
        dt /= 2.0
        s2 = rk4_l(s0_l, inflow_lake, nsteps, dt, lk_area)

        curr_acc = np.abs(s1 - s2) / (0.5 * (s1 + s2)) if s1 ** 2 + s2 ** 2 > 0 else 0

        if curr_acc <= acc:
           break

    if lk_fraction > 1e-4:
        inflow_river = (s0_l - s2) / dt0 + inflow_lake
        sws_lake = s2
    else:
        sws_lake = 0
        inflow_river = inflow_loc + inflow_upstream


    dt = dt0
    nsteps = 1
    while True:

        s1 = rk4_r(s0_r, inflow_river, nsteps, dt, k_bf, k_ob, s_bf)
        nsteps *= 2
        dt /= 2.0
        s2 = rk4_r(s0_r, inflow_river, nsteps, dt, k_bf, k_ob, s_bf)

        curr_acc = np.abs(s1 - s2) / (0.5 * (s1 + s2)) if s1 ** 2 + s2 ** 2 > 0 else 0

        if curr_acc <= acc:
           break


    streamflow = (s0_r - s2) / dt0 + inflow_river
    sws_river = s2


    return streamflow, sws_river, sws_lake

    pass

#route params (stfl-too-high):     300.0000000000000         20186.78
#Other:    5.1975232E+08   8.8461121E-07   4.4230561E-07    1270794.
#Other:    5.0000001E-02    0.000000        0.000000      F   4.7354861E+08
#Other:    5.0000001E-02    174199.2


if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    print step(
        inflow_loc=0,
        inflow_upstream=72829.49,
        k_bf=4.3803720E-07,
        s0_r=5.8258285E+08,   acc= 0.05
    )
    #main()

  
