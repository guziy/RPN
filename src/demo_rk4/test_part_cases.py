__author__ = 'huziy'

import numpy as np



def Q(s, k_bf, k_ob, s_bf):
    if s <= s_bf:
        return k_bf * s ** (4.0/3.0)
    else:
        return k_ob * (s - s_bf) ** (4.0/3.0) + \
               k_bf * s_bf ** (4.0/3.0)



def rk4(s1, inflow, nsteps, dt, k_bf, k_ob, s_bf):
    si = s1
    print(inflow, Q(si, k_bf, k_ob, s_bf))
    for i in xrange(nsteps):
        k1 = dt * (inflow - Q(si, k_bf, k_ob, s_bf))
        k2 = dt * (inflow - Q(si+0.5*k1, k_bf, k_ob, s_bf))
        k3 = dt * (inflow - Q(si+0.5*k2, k_bf, k_ob, s_bf))
        k4 = dt * (inflow - Q(si+k3, k_bf, k_ob, s_bf))
        si += (1.0/6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return si



def main():

    s_bf = 186161.0
    k_bf = 9.5980613e-5
    k_ob = k_bf / 2.0
    s0 = 7.109328

    acc = 5e-3

    inflow = 2.0503999e-6
    dt = 300.0
    nsteps = 1
    s = -1
    curr_acc = -1
    while dt >= 5:
        s1 = s
        s = rk4(s0, inflow, nsteps, dt, k_bf, k_ob, s_bf)
        nsteps *= 2
        curr_acc = np.abs(s1-s) / (0.5 * (s1 + s))
        if acc >= curr_acc and s1 >= 0:
            break

        dt /= 2.0

    print dt, s, curr_acc
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  