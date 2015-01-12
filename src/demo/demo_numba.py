__author__ = 'huziy'

from numba import jit
import time

@jit("int64(int32)")
def fib(n):
    return 1 if n < 2 else fib(n - 1) + fib(n - 2)


if __name__ == '__main__':
    t0 = time.clock()
    fib(30)
    print "Ellapsed time: {} s".format(time.clock() - t0)
