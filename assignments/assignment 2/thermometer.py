import random
import numpy as np
import matplotlib.pyplot as plt
random.seed(42)
noise = 0.75
def get_measurements(T) :
    return 2 + 0.2 * T + random.uniform(-noise, noise)
def regression(x, y) :
    #linear regression problem
    A = np.array([[1, x_i] for x_i in x])
    a_min = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), y)
    alpha = a_min[0]
    beta = a_min[1]
    return alpha, beta
def MSE(y_regression, y_data) :
    return 1/len(y_data) * np.sum((y_regression - y_data)**2)
# Example data
T = np.linspace(0, 80, 15)
H = get_measurements(T)
alpha, beta = regression(T, H)
H_regression = alpha + beta * T
# Create scatter plot
plt.scatter(T, H,  label='Data Points', marker='x')
plt.plot(T, H, linestyle='--', color='red', label = "Regression")
plt.text(30, 5, 'H_regression = {:.2f} + {:.2f} * T'.format(alpha, beta), bbox=dict(facecolor='white', alpha=0.5))
plt.text(30, 3, 'MSE = {:.2e}'.format(MSE(H_regression, H)), bbox=dict(facecolor='white', alpha=0.5))
# Add labels and title
plt.xlabel('Temperature')
plt.ylabel('Height')
plt.title('')
plt.legend()
# Show plot
plt.show()
