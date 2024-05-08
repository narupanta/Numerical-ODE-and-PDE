# Narunat Pantapalin_5406173_CSE
# Neeraj Garud_5269400_CSE
# Kshitij Patle_5420023_CSE

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

#constants
x0 = 0.5
h = 0.1

x_axis = np.linspace(0.0000001, 1, 100)
f_curve = func_x(x_axis)
#Equation of a straight line: y = mx + c
f_tangent_curve = derivative_func_x(x0) * (x_axis - x0) + func_x(x0)

# creating graph space for two graphs
graph, (plot1, plot2) = plt.subplots(1, 2, figsize=(10,5))
plot1.plot( x_axis,f_curve)
plot1.plot( x_axis,f_tangent_curve,'r')

step_h = np.linspace(0.1,0.00001,100)
forward_error = (derivative_func_x(x0) - forward_diff_quot(func_x,x0,step_h))
backward_error = (derivative_func_x(x0) - backward_diff_quot(func_x,x0,step_h))
central_error = (derivative_func_x(x0) - central_diff_quot(func_x,x0,step_h))

plot2.semilogy(step_h, forward_error, label="right")
plot2.semilogy(step_h, backward_error,label="left", color= 'r')
plot2.semilogy(step_h, central_error,label="central", color = 'g')
plot2.invert_xaxis()
plt.xlabel("h")
plt.ylabel("|f'(x0) - f' l,r,c(x0)|")
plt.show()