# Narunat Pantapalin_5406173_CSE
# Neeraj Garud_5269400_CSE
# Kshitij Patle_5420023_CSE
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import Protocol
from scipy.optimize import root

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

def lorenz_system(x, sigma, rho, beta) :
    return np.array([sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]])


class CrankNicolsonMulti(Solver) :
    def __init__(self, f, y0, h, steps) -> None:
        super().__init__()
        self.f = f
        self.y0 = np.array(y0)
        self.h = h
        self.steps = int(steps)
    def solve(self) -> None :
        y = np.zeros((self.steps, self.y0.shape[0]))
        t = np.linspace(0, self.steps * self.h, self.steps)
        y[0] = self.y0
        for i in range(self.steps - 1) :
            r = lambda x : x - y[i] - self.h/2 * (self.f(t[i + 1], x) + self.f(t[i], y[i]))
            y[i + 1] = newtonMulti(r, y[i])
        return t, y[:, 0], y[:, 1], y[:, 2]
    
class EulerHeun(Solver) :
    def __init__(self, f, y0, h, steps) -> None:
        super().__init__()
        self.f = f
        self.y0 = np.array(y0)
        self.h = h
        self.steps = int(steps)
    def solve(self) -> None :
        y = np.zeros((self.steps, self.y0.shape[0]))
        t = np.linspace(0, self.steps * self.h, self.steps)
        y[0] = self.y0
        for i in range(self.steps - 1) :
            r = lambda x : x - y[i] - self.h/2 * (self.f(t[i + 1], y[i] + self.h * self.f(t[i], y[i])) + self.f(t[i], y[i]))
            y[i + 1] = newtonMulti(r, y[i])
        return t, y[:, 0], y[:, 1], y[:, 2]
    
class ImprovedEuler(Solver) :
    def __init__(self, f, y0, h, steps) -> None:
        super().__init__()
        self.f = f
        self.y0 = np.array(y0)
        self.h = h
        self.steps = int(steps)
    def solve(self) -> None :
        y = np.zeros((self.steps, self.y0.shape[0]))
        t = np.linspace(0, self.steps * self.h, self.steps)
        y[0] = self.y0
        for i in range(self.steps - 1) :
            r = lambda x : x - y[i] - self.h * self.f(t[i] + self.h/2, y[i] + self.h/2 * self.f(t[i], y[i]))
            y[i + 1] = newtonMulti(r, y[i])
        return t, y[:, 0], y[:, 1], y[:, 2]
    
if __name__ == "__main__" :
    f = lambda t, x : lorenz_system(x, 10, 28, 8/3)
    ics = [[1, 1, 1], [1, 1, 1.01], [1, 1, 1.02]]
    plt.figure(0)
    ax = plt.axes(projection ='3d')
    for ic in ics :
        crankNicolsonSolver = CrankNicolsonMulti(f, ic, 0.01, 4000)
        t, x1, x2, x3 = crankNicolsonSolver.solve()
        ax.plot3D(x1, x2, x3, label = f"with initial condition: {ic}", linewidth=0.5)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.legend()
    ax.set_title("Chaotic system with CrankNicolsonSolver")    


    plt.figure(1)
    ax = plt.axes(projection ='3d')
    for ic in ics :
        eulerHeunSolver = EulerHeun(f, ic, 0.01, 4000)
        t, x1, x2, x3 = eulerHeunSolver.solve()
        ax.plot3D(x1, x2, x3, label = f"with initial condition: {ic}", linewidth=0.5)
    
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.legend()
    ax.set_title("Chaotic system with Euler-Heun")    

    plt.figure(2)
    ax = plt.axes(projection ='3d')
    for ic in ics :
        improvedEulerSolver = ImprovedEuler(f, ic, 0.01, 4000)
        t, x1, x2, x3 = improvedEulerSolver.solve()
        ax.plot3D(x1, x2, x3, label = f"with initial condition: {ic}", linewidth=0.5)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.legend()
    ax.set_title("Chaotic system with improved Euler")   

    plt.show()