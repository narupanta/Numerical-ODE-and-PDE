#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
plt.xlabel('T')
plt.ylabel('H')
plt.title('')
plt.legend()
# Show plot
plt.show()


# In[19]:


import random
import numpy as np
import matplotlib.pyplot as plt

# Function to simulate measurements
def get_measurement(T):
    random.seed(42)  # Seed with the temperature for reproducibility but different results for each T
    noise = random.uniform(-0.75, 0.75)
    height = 2 + (1/5) * T + noise
    return height

# Get data for all temperatures from 0 to 80
temperatures = np.linspace(0, 80, 15)
measurements = [get_measurement(T) for T in temperatures]
noises = [alpha + beta * T + random.uniform(-0.75, 0.75) for T in temperatures]

# Function for linear regression to find alpha and beta
def regression(X):
    # X is an array of (T, h) pairs
    T = np.array([x[0] for x in X]) #(indexed by 0, which is the temperature Ti)
    h = np.array([x[1] for x in X]) #(indexed by 1, which is the height hi)
    # Building the design matrix for linear regression
    A = np.vstack([T, np.ones(len(T))]).T
    # Solving for linear least squares
    beta, alpha = np.linalg.lstsq(A, h, rcond=None)[0]
    return alpha, beta

# Creating (T, h) pairs for regression
data_pairs = list(zip(temperatures, measurements))
alpha, beta = regression(data_pairs)

# Calculate Mean Squared Error
predicted_heights = [alpha + beta * T for T in temperatures]
mse = np.mean([(h - ph) ** 2 for h, ph in zip(measurements, predicted_heights)])

# Compute Mean Absolute Error (MAE)
mae = np.mean([abs(h - ph) for h, ph in zip(measurements, predicted_heights)])

# Plotting the regression line
plt.scatter(temperatures, measurements, color='red', s=20, label='Data Points')
plt.scatter(temperatures, noises, color='blue', s=20, label='Noise')
plt.plot(temperatures, [alpha + beta * T for T in temperatures], 'green', label=f'Regression Line: h = {alpha:.2f} + {beta:.2f}T')
plt.title('Regression Analysis')
plt.text(40, 3, f'Mean Squared Error = {mse:.2e}', bbox=dict(facecolor='white', alpha=0.5))
plt.text(40, 5, f'Mean Absolute Error = {mae:.2e}', bbox=dict(facecolor='white', alpha=0.5))
plt.xlabel('Temperature (°C)')
plt.ylabel('Height of Liquid (cm)')
plt.legend()
plt.grid(True)
plt.show()

