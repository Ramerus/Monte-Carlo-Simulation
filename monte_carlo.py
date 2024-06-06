from scipy.stats import levy_stable
import matplotlib.pyplot as plt
import numpy as np
import random
import statistics

def r1_observations(n_observations = 750):
    '''
    Makes np.array of 1-day	returns
    n_observations -- number of prices
    '''
    r1 = levy_stable.rvs(1.7, 0.0, loc = 1.0, scale = 1.0, size = n_observations)
    return r1


def prices_maker(first_price, r1, n_observations):
    '''
    Makes np.array of prices 
    n_observations -- number of prices
    r1 -- np.array with prices
    first_price - random price of first day
    '''
    p_first = random.uniform(0.8*first_price, 1.2*first_price)
    p = np.zeros(n_observations)
    p[0] = p_first
    for i in range(1, n_observations):
        p[i] = p[i-1]*(r1[i-1]+1)
    return p


def r10_observations(n_observations, first_price, n_tests, r1):
    '''
    Makes np.array of 10-days overlapping proportional returns
    n_observations -- number of prices
    r1 -- np.array with prices
    first_price - random price of first day
    '''
    for i in range(n_tests):
        r10 = np.zeros(1)
        r10_temp = np.zeros(n_observations-10)
        p = prices_maker(first_price, r1, n_observations)
        for i in range(1, n_observations - 10):
            r10_temp[i] = (p[i+10]-p[i])/p[i]
        r10 = np.concatenate([r10, r10_temp])
    return r10[1:]

def quantile_test(n_quntile_tests, delta_n, n_observations, first_price, r1):
    '''
    Makes np.array of 0.01 quantile of r_10 distribution for different number of tests 
    n_observations -- number of prices
    r1 -- np.array with prices
    first_price - random price of first day
    delta_n -- step of number of tests
    n_quntile_tests -- number of tests
    '''
    result = []
    for i in range(1, n_quntile_tests+1):
        n_tests = i*delta_n
        r10 = r10_observations(n_observations, first_price, n_tests, r1)
        q = statistics.quantiles(r10, n=100, method='exclusive')
        result.append(q[0])
    return result


if __name__ == "__main__":
    n_observations = 750
    n_tests = 1000
    first_price = 1000
    n_quntile_tests = 10
    delta_n = 10
    result = []
    r1 = r1_observations(n_observations)
    r10 = r10_observations(n_observations, first_price, n_tests, r1)
    q = statistics.quantiles(r10, n=100, method='exclusive')
    result = r10[np.where(r10 <= q[0])]
    plt.figure()
    plt.hist(result, density = True, bins = 50)
    plt.show()

    print(quantile_test(n_quntile_tests, delta_n, n_observations, first_price, r1))

