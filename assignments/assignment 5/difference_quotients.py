# Narunat Pantapalin_5406173_CSE
# Neeraj Garud_5269400_CSE
# Kshitij Patle_5420023_CSE

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def sinln(x) :
    return np.sin(x) * np.log(x)
def sinln_derivative(x) :
    return np.cos(x) * np.log(x) + np.sin(x) / x
def forward_diff_quot(f, x0, h) :
    return (f(x0 + h) - f(x0)) / h
def backward_diff_quot(f, x0, h) :
    return (f(x0) - f(x0 - h)) / h
def central_diff_quot(f, x0, h) :
    return (f(x0 + h) - f(x0 - h)) / (2 * h)


def plot_tangent(f, x0, h) :
    plt.figure(0)
    resolution = 100
    x = np.linspace(0, 1, resolution)
    plt.plot(x, f(x) , color='blue', label = "f(x)")
    plt.scatter(x0, f(x0), color = "red", marker = "x")
    plt.plot(x, central_diff_quot(f, x0, h) * (x - x0) + f(x0), color = "orange", linestyle = "--", label = "tangent")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
def plot_compare_method(f, x0) :
    plt.figure(1)
    resolution = 100
    helper = np.linspace(-5, -1, resolution)

    h = [10**(i) for i in helper]
    analytical_df = sinln_derivative(x0)
    error = dict()
    error["forward"] = [np.abs(analytical_df - forward_diff_quot(sinln, x0, h_i)) for h_i in h]
    error["backward"] = [np.abs(analytical_df - backward_diff_quot(sinln, x0, h_i)) for h_i in h]
    error["central"] = [np.abs(analytical_df - central_diff_quot(sinln, x0, h_i)) for h_i in h]

    for method, _ in error.items()  :
        plt.semilogy(h, error[method], label = method)

    plt.xlabel("h")
    plt.ylabel("|error(x)|")
    plt.legend()
    plt.show()
if __name__ == "__main__" :
    f = sinln
    x0 = 1/2
    h = 0.001
    df_forward = forward_diff_quot(f, x0, h) 
    df_backward = backward_diff_quot(f, x0, h) 
    df_central = central_diff_quot(f, x0, h) 
    print("result for forward differentiation quotient:", df_forward)
    print("result for backward differentiation quotient:", df_backward)
    print("result for central differentiation quotient:", df_central)
    plot_tangent(f, x0, h)
    plot_compare_method(f, x0)