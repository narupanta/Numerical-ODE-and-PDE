# Narunat Pantapalin_5406173_CSE
# Neeraj Garud_5269400_CSE
# Kshitij Patle_5420023_CSE
# imports and bigger plot
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.integrate as integrate

from typing import Protocol

plt.rcParams['figure.figsize'] = [10, 10]


# start plot
plt.figure(0)
plt.title("Erde-Mond orbit")

ax = plt.gca()
ax.set_xlim([-1.5, 1.3])
ax.set_ylim([-0.9, 0.9])

# moon and earth
plt.scatter([0, 1], [0, 0], color="green", zorder=0, s=60)
plt.text(-0.075, -0.1, "Erde", fontsize=12)
plt.text(1.0 -0.075, -0.1, "Mond", fontsize=12)

# reference points
plt.scatter(
    [
        1.2,
        0.54515565,
        -0.56370094,
        -1.25574148,
        -0.74762901,
        0.3609336
    ],
    [
        0,
        -0.5536609,
        -0.65126906,
        -0.0999865,
        0.66389666,
        0.43941454
    ],
    color="red",
    zorder=0
)

# uncomment later
#plt.legend()
# plt.show()

class Solver(Protocol) :
    def solve(self) -> None :
        pass

def jacobian(f, x, h=1e-8):
    n = len(x)
    fx = f(x)
    m = len(fx)
    
    # Initialize the Jacobian matrix with zeros
    J = [[0 for _ in range(n)] for _ in range(m)]
    
    # Compute the partial derivatives
    for i in range(n):
        x_step = list(x)
        x_step[i] -= h
        fx_minus = f(x_step)
        x_step[i] += 2 * h
        fx_plus = f(x_step)
        
        for j in range(m):
            J[j][i] = (fx_plus[j] - fx_minus[j]) / (2*h)
    
    return np.array(J, dtype="float64")

def newtonMulti(func, x0, tol = 1e-8, max_iter = 100) :
    x = x0.copy()
    x_history = np.array(x0)
    for i in range(max_iter) :
        delta_x = np.linalg.solve(jacobian(func, x), func(x))
        x -= delta_x
        err = np.linalg.norm(x - x_history)/np.linalg.norm(x_history)
        if abs(err) <= tol :
            break
        x_history = x
    return x

def trajectory(X, mu) :
    x1, x2, y1, y2 = X[0], X[1], X[2], X[3]
    return np.array([x2,
                     x1 + 2 * y2 - (1 - mu) * (x1 + mu)/((x1 + mu)**2 + y1**2)**1.5 - mu * (x1 - (1 - mu))/((x1 - 1 + mu)**2 + y1**2)**1.5,
                     y2,
                     y1 - 2 * x2 - (1 - mu) * (y1)/((x1 + mu)**2 + y1**2)**1.5 - mu * y1/((x1 - 1 + mu)**2 + y1**2)**1.5])
class ExplicitEuler(Solver) :
    def __init__(self, f, y0, h, t_end) -> None:
        super().__init__()
        self.f = f
        self.t_end = t_end
        self.y0 = np.array(y0)
        self.steps = int(t_end / h) + 1
        self.h = h

    def solve(self) :
        y = np.zeros((self.steps, self.y0.shape[0]))
        t = np.linspace(0, self.t_end, self.steps)
        y[0] = self.y0
        for i in range(self.steps - 1) :

            y[i + 1] = y[i] + self.h * self.f(t[i], y[i]) 
        return t, y[:, 0], y[:, 1], y[:, 2], y[:, 3]

class ImplicitEuler(Solver) :
    def __init__(self, f, y0, h, t_end) -> None:
        super().__init__()
        self.f = f
        self.y0 = np.array(y0)
        self.h = h
        self.t_end = t_end
        self.steps = int(t_end / h) + 1
    def solve(self) -> None :
        y = np.zeros((self.steps, self.y0.shape[0]))
        t = np.linspace(0, self.steps * self.h, self.steps)
        y[0] = self.y0
        for i in range(self.steps - 1) :
            r = lambda x : x - y[i] - self.h * self.f(t[i + 1], x)
            y[i + 1] = newtonMulti(r, y[i])
        return t, y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    
class CrankNicolson(Solver) :
    def __init__(self, f, y0, h, t_end) -> None:
        super().__init__()
        self.f = f
        self.y0 = np.array(y0)
        self.h = h
        self.t_end = t_end
        self.steps = int(t_end / h) + 1
    def solve(self) -> None :
        y = np.zeros((self.steps, self.y0.shape[0]))
        t = np.linspace(0, self.steps * self.h, self.steps)
        y[0] = self.y0
        for i in range(self.steps - 1) :
            r = lambda x : x - y[i] - self.h/2 * (self.f(t[i + 1], x) + self.f(t[i], y[i]))
            y[i + 1] = newtonMulti(r, y[i])
        return t, y[:, 0], y[:, 1], y[:, 2], y[:, 3]
class ImprovedEuler(Solver) :
    def __init__(self, f, y0, h, t_end) -> None:
        super().__init__()
        self.f = f
        self.y0 = np.array(y0)
        self.h = h
        self.t_end = t_end
        self.steps = int(t_end / h) + 1
    def solve(self) -> None :
        y = np.zeros((self.steps, self.y0.shape[0]))
        t = np.linspace(0, self.steps * self.h, self.steps)
        y[0] = self.y0
        for i in range(self.steps - 1) :
            r = lambda x : x - y[i] - self.h * self.f(t[i] + self.h/2, y[i] + self.h/2 * self.f(t[i], y[i]))
            y[i + 1] = newtonMulti(r, y[i])
        return t, y[:, 0], y[:, 1], y[:, 2], y[:, 3]    
if __name__ == "__main__" :
    f = lambda t, x : trajectory(x, mu = 1/82.45)
    ic = [1.2, 0, 0, -1.049357509830350]
    T = 6.19
    h = 0.001
    sol = integrate.solve_ivp(f, (0, T), ic, t_eval = np.linspace(0, T, int(1e5)))

    methods = {"ExplicitEuler": ExplicitEuler(f, ic, h, T),
               "ImplicitEuler": ImplicitEuler(f, ic, h, T),
               "CrankNicolson": CrankNicolson(f, ic, h, T),
               "ImprovedEuler": ImprovedEuler(f, ic, h, T)}
    plt.figure(0)
    for method_name, solver in methods.items() :
        t, x1, x2, y1, y2 = solver.solve()
        plt.plot(x1, y1, label = f"{method_name}", linewidth=0.8)
    plt.plot(sol.y[0], sol.y[2], label = f"built-in Python solver" )
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.title("Erde-Mond system")    
    plt.show()

