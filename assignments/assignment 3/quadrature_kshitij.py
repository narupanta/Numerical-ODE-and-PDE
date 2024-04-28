import numpy as np
import scipy
import scipy.integrate
import matplotlib.pyplot as plt

def exponential(x):
    return np.exp(x)
a=0
b= 2*np.pi

computation, abs_error = scipy.integrate.quad(exponential, a,b)

x_axis = np.linspace(a,b,100)

def left_rectangle(a,b, f):
    plt.plot(x_axis,)         

    return f(a)*(b-a)
def right_rectangle(a,b,f):
    return f(b)*(b-a)
def midpoint(a,b, f):
    return f( (a+b)/2 )*(b-a)
def trapezoid(a,b, f):
    return (b-a)/2 * (f(a)+f(b))
def keplar(a,b, f):
    return (b-a)/6 * (f(a) + 4*f((a+b)/2) + f(b))
def own(a,b, f):
    return 0

left_rec = left_rectangle(a,b, exponential)
right_rec = right_rectangle(a,b, exponential)
mid = midpoint(a,b, exponential)
trap = trapezoid(a,b, exponential)
kep = keplar(a,b, exponential)

print("The actual computation",computation)
print("Left rectangle apporoximation: ", left_rec, 
      "\nRight rectangle apporoximation: ", right_rec,
      "\nMidpoint apporoximation: ", mid,
      "\nTrapezoid apporoximation: ", trap,
      "\nKeplar apporoximation: ", kep)
