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


def get_eq_store(ifw, sbf, cbf, cob):
    p = 4.0/3.0
    tmp = cbf * sbf ** p
    #print tmp
    return (ifw/cbf)**(1.0/p) if ifw <= tmp else sbf + ((ifw - tmp)/cob)**(1.0/p)


def Q(s, k_bf, k_ob, s_bf):
    if s <= s_bf:
        return k_bf * s ** (4.0/3.0)
    else:
        print "here"
        return k_ob * (s - s_bf) ** (4.0/3.0) + \
               k_bf * s_bf ** (4.0/3.0)

def Q_lake(s, lk_area):
    h0 = 5.0
    s0 = h0 * lk_area
    kr = 0.01 / (60.0*24*60.0)
    return kr * s * (s/s0) ** 1.5



def rk4_l(s1, inflow, nsteps, dt, lk_area):
    si = s1
    #print(inflow, Q_lake(si, lk_area))
    for i in xrange(nsteps):
        k1 = dt * (inflow - Q_lake(si, lk_area))
        k2 = dt * (inflow - Q_lake(si+0.5*k1, lk_area))
        k3 = dt * (inflow - Q_lake(si+0.5*k2, lk_area))
        k4 = dt * (inflow - Q_lake(si+k3, lk_area))
        si += (1.0/6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return si

def rk4_r(s1, inflow, nsteps, dt, k_bf, k_ob, s_bf):
    si = s1
    s_next = -1
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
        si = get_eq_store(inflow, s_bf, k_bf, k_ob)

    return si



def main(inflow =  36.35912, nits = 20):

    dt0 = 1200.0

    s_bf = 831202.2
    k_bf = 8.1845646E-05
    k_ob = k_bf / 2.0
    s0 =  19918.93
    lk_area = 1.4348829E+07
    is_lake = False


    acc = 1e-5



    err_list = []

    dt = dt0
    nsteps = 1
    curr_acc = -1
    for i in range(nits):
        if is_lake:
            s1 = rk4_l(s0, inflow, nsteps, dt, lk_area)
        else:
            s1 = rk4_r(s0, inflow, nsteps, dt, k_bf, k_ob, s_bf)

        nsteps *= 2
        dt /= 2.0

        if is_lake:
            s2 = rk4_l(s0, inflow, nsteps, dt, lk_area)
        else:
            s2 = rk4_r(s0, inflow, nsteps, dt, k_bf, k_ob, s_bf)




        print( "s1 = {0}, s2 = {1}, dt = {2}, nsteps = {3}".format(s1, s2, dt, nsteps) )
        curr_acc = np.abs(s1 - s2) / (0.5 * (s1 + s2)) if s1 ** 2 + s2 ** 2 > 0 else 0

        err_list.append(curr_acc)


        print("accuracy = {0}".format(curr_acc))
        #if acc >= curr_acc and s1 >= 0:
        #    break

   # dt_a = analytical(k_bf, inflow, s2, s0)


   # ss = np.linspace(0, s0+inflow*dt0, 100)
   # ts = map(lambda x: analytical(k_bf,inflow,x,s0),ss)

  #  plt.figure()
  #  plt.plot(ts, ss, "k", lw = 3)
  #  plt.show()

  #  print "dt_a = ", dt_a
    print dt, s1,s2, curr_acc
    print "store(next) = ", s2
    print "max possible store = ", s0 + inflow * dt0

    q = (inflow * dt0 - s2 + s0) / dt0
    print "streamflow = ", q


    print "ballance: dS = {0}".format(inflow * dt0 - q * dt0)
    print "actual: dS = {0}".format(s2 - s0)

    print len(err_list)
    return err_list
    #plt.plot(err_list)
    #plt.show()

    pass

if __name__ == "__main__":
    errs = []
    infls = list(range(0,100,1))
    its = list(range(1,7))
    for infl in infls:
        errs.append( main(inflow=infl, nits=len(its)) )

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    infls, its = np.meshgrid(infls, its)
    print errs[0]
    errs = np.array(errs).transpose()

    print infls.shape
    print its.shape
    print errs.shape

    errs = errs[2:,:]
    infls = infls[2:,:]
    its = its[2:,:]

    surf = ax.plot_surface( infls, its, errs, cmap=cm.coolwarm,
            linewidth=2, antialiased=False)

    ax.set_zlim3d(0, np.max(errs))
    print np.max(errs)
    plt.colorbar(surf)
    plt.show()

    print "Hello world"
  