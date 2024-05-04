# Narunat Pantapalin_5406173_CSE
# Neeraj Garud_5269400_CSE
# Kshitij Patle_5420023_CSE

import numpy as np
import scipy.integrate

def expocos(x) :
    return np.e**x * np.cos(x)
def summed_trapezoid_rule(f, a, b, J) :
    node = np.linspace(a, b, J + 1)
    total_area = 0
    for idx in range(J) :
        total_area += (f(node[idx]) + f(node[idx + 1])) * (node[idx + 1] - node[idx]) / 2
    return total_area
def romberg_extrapolation_helper(f, a, b, n, J) :
    if  n == 1 :
        return summed_trapezoid_rule(f, a, b, J)
    else :
        return (2**n * romberg_extrapolation_helper(f, a, b, n - 1, 2 * J) - romberg_extrapolation_helper(f, a, b, n - 1, J))/(2**n - 1)
    
def romberg_extrapolation(f, a, b, n) :
    J = 1
    return romberg_extrapolation_helper(f, a, b , n, J)

def optimized_romberg_extrapolation(f, a, b, tolerance) :
    error = np.inf
    n = 1
    while error > tolerance :
        exact_result, _ = scipy.integrate.quad(expocos, 0, 1)
        app_result = romberg_extrapolation(f, a, b, n)
        error = np.abs(app_result - exact_result)
        n += 1
    return app_result, n

if __name__ == "__main__" :
    exact_result, _ = scipy.integrate.quad(expocos, 0, 1)
    result, n = optimized_romberg_extrapolation(expocos, 0, 1, 1e-6)
    print("exact result:", exact_result, "sq unit")
    print("approximate result:", result, "sq unit with Romberg step =", n)