# Narunat Pantapalin_5406173_CSE
# Neeraj Garud_5269400_CSE
# Kshitij Patle_5420023_CSE

import numpy as np
import matplotlib.pyplot as plt
class Poisson1D :
    def __init__(self, a, b, N, g, f, mu) :
        self.a = a
        self.b = b
        self.N = N
        self.g = g
        self.f = f
        self.mu = mu
    def domain_discretization(self) :
        return np.linspace(self.a, self.b, self.N + 1)
    def generate_system_of_equation(self) :
        A = np.zeros((self.N - 1, self.N + 1))
        for i in range(A.shape[0]) :
            A[i][i] = 1
            A[i][i + 1] = -2
            A[i][i + 2] = 1
        return A
    def insert_dirichlet_bc(self, A, f) :
        bc1, bc2 = np.zeros(self.N + 1), np.zeros(self.N + 1)
        bc1[0] = 1
        bc2[-1] = 1
        A = np.insert(A, 0, bc1, axis = 0)
        A = np.insert(A, A.shape[0], bc2, axis = 0)
        f[0] = g(self.a)
        f[-1] = g(self.b) 
        return A, f
    def solve(self) :
        x = self.domain_discretization()
        A = self.generate_system_of_equation()
        f = -((self.b - self.a)/self.N)**2 * self.f(mu = self.mu, x = x)
        modified_A, f_modified = self.insert_dirichlet_bc(A, f)
        solution = np.linalg.solve(modified_A, f_modified) 
        return solution
    def plot(self) :
        x = self.domain_discretization()
        u = self.solve()
        plt.figure(0)
        plt.plot(x, u, label = f"$\mu = {self.mu}$")
def g(x) :
    return 1 - x
def f(mu, x) :
    return -mu * np.sin(np.pi * x) 

if __name__ == "__main__" :
    mu_list = [1, 5, 10, 15, 20, 30]
    for mu in mu_list :
        p = Poisson1D(0, 1, 100, g, f, mu) 
        p.plot()
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.title("1D Poisson's Equation at different $\mu$")
    plt.show()


