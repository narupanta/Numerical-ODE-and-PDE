# Narunat Pantapalin_5406173_CSE
# Neeraj Garud_5269400_CSE
# Kshitij Patle_5420023_CSE

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate

a = 0
b = 2
h = (b-a)/2
J = [3,5,10,20,30]

def sine(x):
    return np.sin(np.pi * x)

computation, abs_error = scipy.integrate.quad(sine, a,b)

def summed_trapezoidal(f, a, b, J):
    steps = np.linspace(a, b, J)
    #print("Original steps",steps)
    area = h/2 * ( f(a) + f(b) + summed_area(steps, sine) ) 
    return area

def summed_area(steps, f ):
    new_area=0
    for i in range(1, len(steps)-1):
        #print("New steps",steps[i])
        new_area += (f(steps[i]) + f(steps[i+1]))  
    return new_area

print("The actual computation =",computation)
for j in J:
    print("The summed_trapezoidal rule for J=",j, " is = ", summed_trapezoidal(sine,a, b, j))




