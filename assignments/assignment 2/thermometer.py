# Narunat Pantapalin_5406173_CSE
# Neeraj Garud_5269400_CSE
# Kshitij Patle_5420023_CSE
import random
import numpy as np
import matplotlib.pyplot as plt
random.seed(42)
noise = 0.75
size = 50
def get_measurement(T) :
    # (a) get the measurement with random noise between -0.75 and 0.75
    return 2 + 0.2 * T + np.random.uniform(-noise, noise, size = size)
def regression(x, y) :
    # (c) calculate the parameters of the linear regression by using
    # pseudo inverse method
    # alpha -> the coefficient of degree 0
    # beta -> the coefficient of degree 1 
    A = np.array([[1, x_i] for x_i in x])
    a_min = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), y)
    alpha = a_min[0]
    beta = a_min[1]
    return alpha, beta

# (e) calculate the MSE (Mean Squared Error) between the linear regression and the data points
def MSE(y_regression, y_data) :
    return 1/len(y_data) * np.sum((y_regression - y_data)**2)
def MAE(y_regression, y_data) :
    # (e.1) MAE (Mean Absolute Error) is another metric for determining the accuracy of the linear regression.
    # MSE identifies the larger errors to a great degree compared to MAE - MSE squares the errors 
    # before averaging them, which means larger errors have a disproportionately larger effect on 
    # the MSE compared to smaller errors. This can make MSE more sensitive to outliers than MAE.
    # In summary, the MAE is suitable for higher noise/outlier data than MSE.
    return 1/len(y_data) * np.sum(abs(y_regression - y_data))

T = np.linspace(0, 80, size)
H = get_measurement(T)

alpha, beta = regression(T, H)
H_regression = alpha + beta * T
# (b) create a visualization of the relationship between temperature and height of the liquid pillar
plt.scatter(T, H,  label='Data Points', marker='x')
# plt.plot(T, H)
# (d) plot the linear regression along with the data points  
plt.plot(T, H_regression, linestyle='--', color='red', label = "Regression")
plt.text(40, 8, 'H_regression = {:.2f} + {:.2f} * T'.format(alpha, beta), bbox=dict(facecolor='white', alpha=0.5))
plt.text(40, 5, 'MSE = {:.2e}'.format(MSE(H_regression, H)), bbox=dict(facecolor='white', alpha=0.5))
plt.text(40, 3, 'MAE = {:.2e}'.format(MAE(H_regression, H)), bbox=dict(facecolor='white', alpha=0.5))
# Add labels and title
plt.xlabel('T: Temperature')
plt.ylabel('H: Height of the liquid pillar')
plt.title('Linear Regression')
# plt.xlim((0,8))
# plt.ylim((2,4))
plt.legend()
# Show plot
plt.show()
