import numpy as np
import matplotlib.pyplot as plt

def forward_diff_quot(f,x0,h):
    return (f(x0 + h) - f(x0)) / h

def backward_diff_quot(f,x0,h):
    return (f(x0) - f(x0 - h)) / h

def central_diff_quot(f,x0,h):
    return  (f(x0 + h) - f(x0 - h)) / (2*h)

#function f(x) definition
def func_x(x):                        
    return ( np.sin(x) * np.log(x) )

def derivative_func_x(x):
    return  np.cos(x) * np.log(x) + np.sin(x) / x

x0 = 0.5
h = 0.1

x_axis = np.linspace(0.0000001, 1, 100)
f_curve = func_x(x_axis)
#Equation of a straight line: y = mx + c
f_tangent_curve = derivative_func_x(x0) * (x_axis - x0) + func_x(x0)

# plt.figure()
# plt.plot( x_axis,f_curve)
# plt.plot( x_axis,f_tangent_curve,'r')
# plt.show()

# print("Forward Approximation is = ",forward_diff_quot(func_x,x0,h))
# print("Backward Approximation is = ",backward_diff_quot(func_x,x0,h))
# print("Central Approximation is = ",central_diff_quot(func_x,x0,h))
# print("Calulated value is = ",derivative_func_x(x0))

step_h = []
forward_error = []
backward_error = []
central_error = []
for i in range(-1,-6,-1):
    step_h.append(10**i)
    forward_error.append(derivative_func_x(x0) - forward_diff_quot(func_x,x0,10**i))
    backward_error.append(derivative_func_x(x0) - backward_diff_quot(func_x,x0,10**i))
    central_error.append(derivative_func_x(x0) - central_diff_quot(func_x,x0,10**i))

plt.figure()
plt.semilogy(forward_error, step_h)
plt.semilogy(backward_error,step_h,'r')
plt.semilogy(central_error,step_h,'g')
plt.show()