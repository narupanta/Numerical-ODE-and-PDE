import numpy as np
import random
import matplotlib.pyplot as plt

#function get_measurement(T)
random.seed(42)
temp_points = 45
def get_measurement(t):
    
    error = np.random.uniform(-0.75,0.75, size = temp_points)   #error generation between -0.75 to 0.75
    height = 2 + 0.5*t + error

    return height

t = np.linspace(0, 80 ,temp_points)   #generation of temparture values between 0 to 80, as t is a natural number
h = get_measurement(t)        #height values for corresponding temperature

plt.plot(t, h, 'rx')         # b is for blue, o for circle dot
plt.xlabel("Temparture")
plt.ylabel("Height")
# plt.show()

#regression function
def regression(X):

    x_data_points = X[:,0]
    y_data_points = X[:,1]

    x_mean = np.mean(x_data_points)
    y_mean = np.mean(y_data_points)

    x_minus_mean = x_data_points - x_mean
    y_minus_mean = y_data_points - y_mean

    num = 0
    dem = 0
    for i in range(len(X)):
        
        num += x_minus_mean[i] * y_minus_mean[i]
        dem += x_minus_mean[i]**2

    beta = num / dem
    alpha = y_mean - (beta * x_mean)

    return alpha, beta

X = np.column_stack((t,h))
alpha, beta = regression(X)

y = alpha + beta*t
plt.plot(t,y, linestyle='dashed')
plt.show()

#calculating the Mean squared error
mse=0
for i in range(len(h)):
    mse += ( (y[i]-h[i])**2 )
    
mse = mse / len(h)

print("MSE:",mse)
