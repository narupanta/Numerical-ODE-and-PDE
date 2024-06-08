# Narunat Pantapalin_5406173_CSE
# Neeraj Garud_5269400_CSE
# Kshitij Patle_5420023_CSE
from euler import ExplicitEuler1D, ImplicitEuler1D, CrankNicolson1D
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__" :
    f = lambda t, y : 2 * t * (1 + y)
    t_end = 2
    y0 = 0
    h = [1, 0.5, 0.1, 0.01]
    fig, ax = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    for i, h_i in enumerate(h) :

        explicitEuler1DSolver = ExplicitEuler1D(f, t_end, y0, h = h_i)
        t, explicitEuler1DSolution = explicitEuler1DSolver.solve()

        implicitEuler1DSolver = ImplicitEuler1D(f, t_end, y0, h = h_i)
        t, implicitEuler1DSolution = implicitEuler1DSolver.solve()

        crankNicolson1DSolver = CrankNicolson1D(f, t_end, y0, h = h_i)
        t, crankNicolsonSolution = crankNicolson1DSolver.solve()

        t_analytic = np.linspace(0, t_end, 1000)
        y_analytic = np.exp(t_analytic ** 2) - 1

        ax[i//2][i%2].plot(t_analytic, y_analytic, label="Analytic Solution")
        ax[i//2][i%2].plot(t, explicitEuler1DSolution, label="ExplicitEuler1D")
        ax[i//2][i%2].plot(t, implicitEuler1DSolution, label="ImplicitEuler1D")
        ax[i//2][i%2].plot(t, crankNicolsonSolution, label="CrankNicolson1D")
        ax[i//2][i%2].set_xlabel("$t$")
        ax[i//2][i%2].set_ylabel("$y$")
        ax[i//2][i%2].legend()
        ax[i//2][i%2].set_title(f"Solution for h ={h_i}")


    plt.show()
