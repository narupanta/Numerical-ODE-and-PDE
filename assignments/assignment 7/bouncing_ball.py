from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
class Ball :
    def __init__(self, m, c, d, r, g) -> None:
        self.m = m
        self.c = c
        self.d = d
        self.r = r
        self.g = g
    def freefall(self, t, y) :
        return np.array([y[1], -self.g])
    def bouncing(self, t, y) :
        return np.array([y[1], -(y[0] * self.c + y[1] * self.d)/self.m + self.g])
    def fty(self, t, y) :
        if y[0] < self.r :
            ret = self.bouncing(t, y)
        else :
            ret = self.freefall(t, y)
        return ret
if __name__ == "__main__" :
    # physical settings
    g = 9.81
    r = 0.01
    m = 0.01
    c = 5000
    d = 0.1
    h0 = 1
    v0 = 0
    # numerical settings
    time_step = 300
    time_start = 0
    time_end = 3
    t_i = np.linspace(time_start, time_end, time_step)
    y0 = [h0, v0]

    ball = Ball(m, c, d, r, g)
    dydt = ball.fty
    solution = solve_ivp(dydt, (time_start, time_end), y0, method = 'RK23', t_eval = t_i)
    t = solution.t
    y = solution.y[0]

    # Plot the results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(t, y, label = "h(t)")
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('h(t)')
    ax[0].set_ylim(0, 1)
    ax[0].set_title('Solution of ODE')
    ax[0].legend()

    # Animation
    ax[1].set_xlim(-1, 1)
    ax[1].set_ylim(0, 1.2)
    ball, = ax[1].plot([], [], 'o', markersize=5)
    ax[1].set_xticks([]) 
    # Define the initialization function
    def init():
        ball.set_data([], [])
        return ball,

    # Define the animation function
    def animate(i):
        ball.set_xdata(0)
        ball.set_ydata(y[i])
        return ball,

    # Create the animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(y), interval=20, blit=True)

    # Show the plot
    plt.show()


    #compare ode solver
    methods = [["RK23", "RK45", "DOP853"], ["Radau", "BDF", "LSODA"]]
    fig, ax = plt.subplots(2, 3)

    for i in range(2) :
        for j in range(3) :
            solution_met = solve_ivp(dydt, (time_start, time_end), y0, method = methods[i][j], t_eval = t_i)
            t_met = solution_met.t
            y_met = solution_met.y[0]
            ax[i][j].plot(t_met, y_met, label = methods[i][j])
            if i == 1 :
                ax[i][j].set_xlabel('Time')
            if j == 0 :
                ax[i][j].set_ylabel('h(t)')
            ax[i][j].set_ylim(0, 1)
            ax[i][j].legend()
    plt.show()


    