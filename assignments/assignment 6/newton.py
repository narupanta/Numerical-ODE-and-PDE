import numpy as np
import matplotlib.pyplot as plt

def F(X):
    x1, x2 = X
    return np.array([np.e**(-np.e**(-(x1 + x2))) - x2 * (1 + x1**2), x1 * np.cos(x2) + x2 * np.sin(x1) - 1/2], dtype = "float64")
def G(X) :
    x1, x2 = X
    return np.e**(-1/3*x1**3 + x1 - x2**2)
def gradient(f, x, h=1e-8):
    n = len(x)
    grad = [0] * n
    
    # Compute the partial derivatives
    for i in range(n):
        x_step = list(x)
        x_step[i] += h
        grad[i] = (f(x_step) - f(x)) / h
    
    return grad
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
def newton(func, x0) :
    x = x0
    x_history = np.array([x0] + [[0,0]]*50)
    for i in range(1, 51) :
        x -= np.linalg.solve(jacobian(func, x), func(x))
        x_history[i] = x
        err = x - x_history[i - 1]
        err = np.linalg.norm(x_history[i] - x_history[i - 1])/np.linalg.norm(x_history[i - 1])
        if err <= 1e-6 :
            x_history = x_history[:i+1]
            break
    return np.array(x_history)
def plot_newton_path(x_history) :
    plt.figure(0)
    plt.plot(x_history[:, 0], x_history[:, 1])
    plt.scatter(x_history[:-1, 0], x_history[:-1, 1], color = "green")
    plt.scatter(x_history[-1, 0], x_history[-1, 1], color = "red", label = "final iteration: ({:.2f}, {:.2f})".format(x_history[-1, 0], x_history[-1, 1]))
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.title("Path which Newton’s method takes in Assignment 6.1 (c)")
    plt.show()
if __name__ == "__main__" :
    x_history = newton(F, np.array([0, 0], dtype="float64"))
    plot_newton_path(x_history)

    x2_history = newton(lambda x : gradient(G(x), x) , np.array([0, 0], dtype = "float64"))
    plot_newton_path(x2_history)
