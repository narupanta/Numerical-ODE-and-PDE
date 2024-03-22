import matplotlib.pyplot as plt
import numpy as np

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
    def plot(self) :
        plt.scatter(self.x, self.y, label='Scatter Plot', color='blue')

        # Create line plot
        x_interpolated, y_interpolated = self.fit(100)
        plt.plot(x_interpolated, y_interpolated, label='Line Plot', color='red')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Scatter and Line Plot')

        # Add legend
        plt.legend()

        # Display the plot
        plt.show()

if __name__ == "__main__" :
    l = LagrangeInterpolation(x = np.array([0, 1, 1.1, 5/4, 2.4, 3]), y = np.array([0.5, 10, 2, 0, 2, 0.5]))
    l.plot()


