import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.integrate 

#constants
g = 9.81
r = 0.01
m = 0.01 
c = 5000
d = 0.1

#initialization
t_span = [0,3]
t_eval = np.linspace(0,3,300)

def motion(t, y):
    h, v = y
    if(h>=r):
        dvdt = -1*g
        dydt = v
    else:
        dydt = v
        dvdt = (m*g - c*h - d*v)/m

    return dydt, dvdt

y0 = [1,0] #h0=1, v0=0

sol = scipy.integrate.solve_ivp(motion, t_span, y0, t_eval=t_eval)

ball =  plt.plot(sol.t, sol.y[0])
# plt.show()

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.2)
ball, = ax.plot([], [], 'o', markersize=10)

# Initialization function
def init():
    ball.set_data([], [])
    return ball,

# Animation function
def animate(i):
    x = 0.5  # horizontal position (fixed)
    y = sol.y[0][i]  # vertical position
    ball.set_data(x, y)
    return ball,

# Create the animation
ani = FuncAnimation(fig, animate, frames=len(t_eval), init_func=init, blit=True)

# Display the animation
plt.show()