# Narunat Pantapalin_5406173_CSE
# Neeraj Garud_5269400_CSE
# Kshitij Patle_5420023_CSE

import numpy as np
import scipy
import scipy.integrate
import matplotlib.pyplot as plt

# (a) print integration of f(x) = e**x from x = 0 to x = 2 * pi
def exponential(x) :
    return np.e**x
I, err = scipy.integrate.quad(exponential, 0, 2 * np.pi)


def left_rectangle(f, a, b) :
    return f(a) * (b - a)
def right_rectangle(f, a, b) :
    return f(b) * (b - a)
def midpoint(f, a, b) :
    return f((a + b)/2) * (b - a)
def trapezoid(f, a, b) :
    return 0.5 * (f(a) + f(b)) * (b - a)
def kepler(f, a, b) :
    return (b - a)/6 *  (f(a) + 4 * f((a+b)/2) + f(b))
def own(f, a, b):
    return (b - a)/8 *  (f(a) + 3 * f((2*a+b)/3 ) + 3 * f((a+2*b)/3 ) + f(b)) 

class LagrangeInterpolation :
    def __init__(self, x: np.array, y: np.array) :
        self.x: np.array = x
        self.y: np.array = y
    def fit(self, resolution) :
        x_interpolated = np.linspace(min(self.x), max(self.x), resolution)
        y_interpolated = np.zeros(resolution)
        for k in range(resolution) :
            total = 0
            for i in range(len(self.x)) :
                x_i = self.x[i]
                y_i = self.y[i]
                basis = y_i
                for x_j in self.x :
                    if x_i != x_j :
                        basis *= (x_interpolated[k] - x_j)/(x_i - x_j)
                total += basis
            y_interpolated[k] = total

        return x_interpolated, y_interpolated

def left_rectangle_function(f, a, x) :
    return np.ones(x.shape[0])*f(a)
def right_rectangle_function(f, b, x) :
    return np.ones(x.shape[0])*f(b)
def midpoint_function(f, a, b, x) :
    return np.ones(x.shape[0])*f((a + b)/2)
def trapezoid_function(f, a, b, x) :
    return np.linspace(f(a), f(b), len(x))
def kepler_function(f, a, b, x) :
    poly = LagrangeInterpolation(x = [a, (a + b)/2, b], y = [f(a), f((a + b)/2), f(b)])
    _, y_interpolated = poly.fit(resolution = len(x))
    return y_interpolated
def own_quadrature_function(f, a, b, x) :
    # x0 = a, x1 = (2*a + b)/3 , x2 = (a + 2*b)/3 , x3 = b
    poly = LagrangeInterpolation(x = [a, (2*a + b)/3, (a + 2*b)/3, b], y = [f(a), f((2*a + b)/3), f((a + 2*b)/3), f(b)])
    _, y_interpolated = poly.fit(resolution = len(x))
    return y_interpolated


print("Exact area := ", "{:.02f}".format(I), "unit**2")
print("Calculated area from Left rectangle rule :=", "{:.02f}".format(left_rectangle(exponential, a = 0, b = 2 * np.pi )), "unit**2")
print("Calculated area from Right rectangle rule :=", "{:.02f}".format(right_rectangle(exponential, a = 0, b = 2 * np.pi )), "unit**2")
print("Calculated area from Midpoint rule :=", "{:.02f}".format(midpoint(exponential, a = 0, b = 2 * np.pi )), "unit**2")
print("Calculated area from Trapezoid rule :=", "{:.02f}".format(trapezoid(exponential, a = 0, b = 2 * np.pi )), "unit**2")
print("Calculated area from Simpson’s/ Kepler’s barrel rule :=", "{:.02f}".format(kepler(exponential, a = 0, b = 2 * np.pi )), "unit**2")
print("Calculated area from own (3/8) rule :=", "{:.02f}".format(own(exponential, a = 0, b = 2 * np.pi )), "unit**2")

fig, ax = plt.subplots(nrows=2, ncols=3)
fig.set_figheight(6)
fig.set_figwidth(8)
x = np.linspace(0, 2 * np.pi, 100)
i = 0
titles = ["Left Rectangle Rule","Right Rectangle Rule", "Midpoint Rule", "Trapezoid Rule", "Simpson's/Kepler's Barrel Rule", "Own Quadrature Rule"]
for ax_row in ax :
    for ax_col in ax_row :
        ax_col.plot(x, exponential(x) , color='blue', label = "Actual Function")
        ax_col.fill_between(x, exponential(x), color = "blue", alpha = 0.3)
        match i:
            case 0:
                ax_col.plot(x, left_rectangle_function(f = exponential, a = x[0], x = x), color = "green")
                ax_col.fill_between(x, left_rectangle_function(f = exponential, a = x[0], x = x), color = "green", alpha = 0.3)
            case 1:
                ax_col.plot(x, right_rectangle_function(f = exponential, b = x[-1], x = x), color = "green")
                ax_col.fill_between(x, right_rectangle_function(f = exponential, b = x[0], x = x), color = "green", alpha = 0.3)
            case 2:
                ax_col.plot(x, midpoint_function(f = exponential, a = 0, b = x[-1], x = x), color = "green")
                ax_col.fill_between(x, midpoint_function(f = exponential, a = 0, b = x[-1], x = x), color = "green", alpha = 0.3)
            case 3:
                ax_col.plot(x, trapezoid_function(f = exponential, a = 0, b = x[-1], x = x), color = "green")
                ax_col.fill_between(x, trapezoid_function(f = exponential, a = 0, b = x[-1], x = x), color = "green", alpha = 0.3)
            case 4:
                ax_col.plot(x, kepler_function(f = exponential, a = 0, b = x[-1], x = x), color = "green")
                ax_col.fill_between(x, kepler_function(f = exponential, a = 0, b = x[-1], x = x), color = "green", alpha = 0.3)
            case 5:
                ax_col.plot(x, own_quadrature_function(f = exponential, a = 0, b = x[-1], x = x), color = "green")
                ax_col.fill_between(x, own_quadrature_function(f = exponential, a = 0, b = x[-1], x = x), color = "green", alpha = 0.3)
        ax_col.set_xlabel('x')
        ax_col.set_ylabel('f(x)')
        ax_col.set_title(titles[i])
        i += 1
fig.tight_layout()
plt.show()
