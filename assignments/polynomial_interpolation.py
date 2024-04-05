import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Slider

def runge_function(x) :
    return 1 / (1 + x**2)
def vander(x_points: np.array) :
    degree = len(x_points) - 1
    dummy_list = []
    for x_point in x_points :
        dummy_list.append([x_point**i for i in range(degree + 1)])
    return np.array(dummy_list)
def poly_coeffs(x_points, y_points) :
    return np.matmul(np.linalg.inv(vander(x_points)),y_points)

def polynomial(x, coeff_list) :
    return sum([coeff * x**i for i, coeff in enumerate(coeff_list)])


if __name__ == "__main__" :
    # for runge function plot
    x = np.linspace(-3, 3, 100)
    y = runge_function(x)
    # for scatter plot and interpolated function plot
    number_of_data_points = 10
    x_points = np.linspace(-3, 3, number_of_data_points)
    y_points = runge_function(x_points)
    y_interpolated = polynomial(x, poly_coeffs(np.linspace(-3, 3, number_of_data_points), y_points))


    fig, ax = plt.subplots()
    line, = ax.plot(x, y)
    ax.plot(x, y, label='Runge Function Plot', color='red')
    scatter = ax.scatter(x_points, y_points, label='Data Points', color='blue')
    poly, = ax.plot(x, y_interpolated, label='Interpolated Polynomial', color='green')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Polynomial Interpolation')
    plt.legend()
    ax_data_points = plt.axes([0.25, 0, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider= Slider(ax_data_points, '#Data Points', valmin=3, valmax=100, valinit= number_of_data_points)
    def update(val):
        number_of_data_points = int(slider.val)  # Get the current value of the slider
        scatter.set_offsets(np.column_stack((np.linspace(-3, 3, number_of_data_points), runge_function(np.linspace(-3, 3, number_of_data_points))))) 
        poly.set_ydata(polynomial(x, poly_coeffs(np.linspace(-3, 3, number_of_data_points), runge_function(np.linspace(-3, 3, number_of_data_points)))))  # Update the y-data of the plot
        fig.canvas.draw_idle() 
    slider.on_changed(update)
    plt.show()

# Note for Assignment 1.2 f)
# When increasing the number of data points, in the region of x between (-2,2) the result is closer to the exact function until 34 data points, 
# after this the result in this region is diverging from the exact function (Using slider on the plot to visualize how the plot change). 
# Furthermore, in the region of x between (-3, -2) and (2, 3), the result is diverging from the exact function since the beginning. 
# My intepretation: More polynomial degree doesn't mean better result