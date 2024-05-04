# Narunat Pantapalin_5406173_CSE
# Neeraj Garud_5269400_CSE
# Kshitij Patle_5420023_CSE

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def trapezoid(f, a, b) :
    return 0.5 * (f(a) + f(b)) * (b - a)

def f(x) :
    return np.sin(np.pi * x)

def area_calculate(f, a, b, J) :
    intervals = np.linspace(a, b, J + 1)
    total_area = 0
    for idx in range(J - 1) :
        total_area += trapezoid(f, intervals[idx], intervals[idx + 1])

    return total_area
def summed_trapezoidal(f, a, b, J) :
    intervals = np.linspace(a, b, J + 1)
    total_area = area_calculate(f, a, b, J)

    x = np.linspace(a, b, 100)
    y_trapezoid = f(intervals)

    global fig, ax, line, fill

    fig = plt.figure(figsize = (5,4)) 
    ax = fig.add_subplot(111)
    ax.plot(x, f(x) , color='blue', label = "Actual")
    line, = ax.plot(intervals, y_trapezoid , color='green', label = "Approximate (area = {:.2f} sq unit)".format(total_area))
    fill = ax.fill_between(intervals, y_trapezoid, color = "green", alpha = 0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend(bbox_to_anchor = (1.1, 0))
    ax.set_title(f"Summed trapezoidal rule for f(x) = sin(pi * x) with J = {str(J)}")


def update_plus(click):
    global J, fill, line
    if click :
        J += 1
        line.remove()
        fill.remove()
        intervals = np.linspace(a, b, J + 1)
        y_trapezoid = f(intervals)
        total_area = area_calculate(f, a, b, J)
        line, = ax.plot(intervals, y_trapezoid , color='green', label = "Approximate (area = {:.2f} sq unit)".format(total_area))
        fill = ax.fill_between(intervals, y_trapezoid, color = "green", alpha = 0.3)
        ax.set_title(f"Summed trapezoidal rule for f(x) = sin(pi * x) with J = {str(J)}")
        ax.legend(bbox_to_anchor = (1.1, 0))
        plt.draw()
            
def update_minus(click):
    global J, fill, line
    if click :
        J -= 1
        line.remove()
        fill.remove()
        intervals = np.linspace(a, b, J + 1)
        y_trapezoid = f(intervals)
        total_area = area_calculate(f, a, b, J)
        line, = ax.plot(intervals, y_trapezoid , color='green', label = "Approximate (area = {:.2f} sq unit)".format(total_area))
        fill = ax.fill_between(intervals, y_trapezoid, color = "green", alpha = 0.3)
        ax.set_title(f"Summed trapezoidal rule for f(x) = sin(pi * x) with J = {str(J)}")
        ax.legend(bbox_to_anchor = (1.1, 0))
        plt.draw()

if __name__ == "__main__" :
    global a, b, J
    a = 0
    b = 2
    J = 10
    
    summed_trapezoidal(f, a, b, J)

    button_plus_pos = fig.add_axes([0.8, 0.8, 0.1, 0.075])
    button_minus_pos = fig.add_axes([0.8, 0.72, 0.1, 0.075])

    button_plus = Button(button_plus_pos, "J++")
    button_plus.on_clicked(update_plus)
    button_minus = Button(button_minus_pos, "J--")
    button_minus.on_clicked(update_minus)

    plt.show()
    
