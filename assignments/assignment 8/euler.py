# Narunat Pantapalin_5406173_CSE
# Neeraj Garud_5269400_CSE
# Kshitij Patle_5420023_CSE
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import Protocol
from scipy.optimize import root

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

def central_diff_quot(f, x0, h):
    return  (f(x0 + h) - f(x0 - h)) / (2*h)

def newton1D(func, x0, tol = 1e-6, max_iter = 100) :
    x = x0
    for i in range(max_iter) :
        delta_x = func(x)/central_diff_quot(func, x, 1e-8)
        x -= delta_x
        if abs(delta_x) <= tol :
            break
    return x

def newton2D(func, x0, tol = 1e-8, max_iter = 100) :
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
class Solver(Protocol) :
    def solve(self) -> None :
        pass

class ExplicitEuler1D(Solver) :
    def __init__(self, f, t_end, y0, h) -> None:
        super().__init__()
        self.f = f
        self.t_end = t_end
        self.y0 = y0
        self.steps = int(t_end / h) + 1
        self.h = h

    def solve(self) :
        y = np.zeros(self.steps)
        t = np.linspace(0, self.t_end, self.steps)
        y[0] = self.y0
        for i in range(self.steps - 1) :
            y[i + 1] = y[i] + self.h * self.f(t[i], y[i]) 
        return t, y
    def plot(self) -> None:
        t, y = self.solve()
        plt.figure(0)
        plt.plot(t, y, label = "step size = {}".format(self.h))
        plt.xlabel("$t$")
        plt.ylabel("$y$")
        plt.legend()
        plt.title("")

class ImplicitEuler1D(Solver) :
    def __init__(self, f, t_end, y0, h) -> None:
        super().__init__()
        self.f = f
        self.t_end = t_end
        self.y0 = y0
        self.steps = int(t_end / h) + 1
        self.h = h
    def solve(self) -> None :
        y = np.zeros(self.steps)
        t = np.linspace(0, self.t_end, self.steps)
        y[0] = self.y0
        for i in range(self.steps - 1) :
            r = lambda x : x - y[i] - self.h * self.f(t[i + 1], x)
            y[i + 1] = newton1D(r, y[i])
        return t, y
    def plot(self) -> None:
        t, y = self.solve()
        plt.figure(1)
        plt.plot(t, y, label = "step size = {}".format(self.h))
        plt.xlabel("$t$")
        plt.ylabel("$y$")
        plt.legend()
        plt.title("")
class CrankNicolson1D(Solver) :
    def __init__(self, f, t_end, y0, h) -> None:
        super().__init__()
        self.f = f
        self.t_end = t_end
        self.y0 = y0
        self.steps = int(t_end / h) + 1 
        self.h = h
    def solve(self) -> None :
        y = np.zeros(self.steps)
        t = np.linspace(0, self.t_end, self.steps)
        y[0] = self.y0
        for i in range(self.steps - 1) :
            r = lambda x : x - y[i] - self.h/2 * (self.f(t[i + 1], x) + self.f(t[i], y[i]))
            y[i + 1] = newton1D(r, y[i])
        return t, y
    def plot(self) -> None:
        t, y = self.solve()
        plt.figure(2)
        plt.plot(t, y, label = "step size = {}".format(self.h))
        plt.xlabel("$t$")
        plt.ylabel("$y$")
        plt.legend()
        plt.title("")

class ExplicitEuler2D(Solver) :
    def __init__(self, f, t_end, y0, h) -> None:
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
        return t, y[:, 0], y[:, 1]
    def plot(self) -> None:
        t, y = self.solve()
        plt.figure(0)
        plt.plot(t, y, label = "step size = {}".format(self.h))
        plt.xlabel("$t$")
        plt.ylabel("$y$")
        plt.legend()
        plt.title("")

class ImplicitEuler2D(Solver) :
    def __init__(self, f, t_end, y0, h) -> None:
        super().__init__()
        self.f = f
        self.t_end = t_end
        self.y0 = np.array(y0)
        self.steps = int(t_end / h) + 1
        self.h = h
    def solve(self) -> None :
        y = np.zeros((self.steps, self.y0.shape[0]))
        t = np.linspace(0, self.t_end, self.steps)
        y[0] = self.y0
        for i in range(self.steps - 1) :
            r = lambda x : x - y[i] - self.h * self.f(t[i + 1], x)
            y[i + 1] = newton2D(r, y[i])
        return t, y[:, 0], y[:, 1]
    def plot(self) -> None:
        t, y = self.solve()
        plt.figure(1)
        plt.plot(t, y, label = "step size = {}".format(self.h))
        plt.xlabel("$t$")
        plt.ylabel("$y$")
        plt.legend()
        plt.title("")
class CrankNicolson2D(Solver) :
    def __init__(self, f, t_end, y0, h) -> None:
        super().__init__()
        self.f = f
        self.t_end = t_end
        self.y0 = np.array(y0)
        self.steps = int(t_end / h) + 1 
        self.h = h
    def solve(self) -> None :
        y = np.zeros((self.steps, self.y0.shape[0]))
        t = np.linspace(0, self.t_end, self.steps)
        y[0] = self.y0
        for i in range(self.steps - 1) :
            r = lambda x : x - y[i] - self.h/2 * (self.f(t[i + 1], x) + self.f(t[i], y[i]))
            y[i + 1] = newton2D(r, y[i])
        return t, y[:, 0], y[:, 1]
    def plot(self) -> None:
        t, y = self.solve()
        plt.figure(2)
        plt.plot(t, y, label = "step size = {}".format(self.h))
        plt.xlabel("$t$")
        plt.ylabel("$y$")
        plt.legend()
        plt.title("")


if __name__ == "__main__" :
    f = lambda t, y : 2 * t * (1 + y)
    t_end = 2
    y0 = 0
    h = [1, 0.5, 0.1, 0.01, 0.0001]

    for h_i in h :
        solver = ExplicitEuler1D(f, t_end, y0, h = h_i)
        solution = solver.solve()
        solver.plot()

    for h_i in h :
        solver = ImplicitEuler1D(f, t_end, y0, h = h_i)
        solution = solver.solve()
        solver.plot()

    for h_i in h :
        solver = CrankNicolson1D(f, t_end, y0, h = h_i)
        solution = solver.solve()
        solver.plot()
    
    plt.show()