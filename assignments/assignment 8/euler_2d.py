# Narunat Pantapalin_5406173_CSE
# Neeraj Garud_5269400_CSE
# Kshitij Patle_5420023_CSE
from euler import ExplicitEuler2D, ImplicitEuler2D, CrankNicolson2D
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
if __name__ == "__main__" :
    f = lambda t, y : np.array([-4 * y[0] + 6 * y[1], 31 * y[0] - 189 * y[1]])
    t_end = 2
    y0 = [1, 2]
    h = [0.01, 0.02]
    fig, ax = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

    time = sp.symbols('t')
    y1 = sp.Function('y1')(time)
    y2 = sp.Function('y2')(time)
    eq1 = sp.Eq(y1.diff(time), -4 * y1 + 6 * y2)
    eq2 = sp.Eq(y2.diff(time), 31 * y1 - 189 * y2)
    ics = {y1.subs(time, 0): y0[0], y2.subs(time, 0): y0[1]}
    sol_analytic = sp.dsolve((eq1, eq2), ics=ics)
    y1_sol = sp.lambdify(time, sol_analytic[0].rhs, 'numpy')
    y2_sol = sp.lambdify(time, sol_analytic[1].rhs, 'numpy')
    for i, h_i in enumerate(h) :

        explicitEuler2DSolver = ExplicitEuler2D(f, t_end, y0, h = h_i)
        t, y_exp1, y_exp2 = explicitEuler2DSolver.solve()

        implicitEuler2DSolver = ImplicitEuler2D(f, t_end, y0, h = h_i)
        t, y_imp1, y_imp2 = implicitEuler2DSolver.solve()

        crankNicolson2DSolver = CrankNicolson2D(f, t_end, y0, h = h_i)
        t, y_crank1, y_crank2 = crankNicolson2DSolver.solve()
        t_analytic = np.linspace(0, t_end, 1000)


        ax[i][0].plot(t_analytic, y1_sol(t_analytic), label="Analytic Solution")
        ax[i][1].plot(t_analytic, y2_sol(t_analytic), label="Analytic Solution")
        ax[i][0].plot(t, y_exp1, label="ExplicitEuler2D")
        ax[i][1].plot(t, y_exp2, label="ExplicitEuler2D")
        ax[i][0].plot(t, y_imp1, label="ImplicitEuler2D")
        ax[i][1].plot(t, y_imp2, label="ImplicitEuler2D")
        ax[i][0].plot(t, y_crank1, label="CrankNicolson2D")
        ax[i][1].plot(t, y_crank2, label="CrankNicolson2D")


        ax[i][0].set_xlabel("$t$")
        ax[i][0].set_ylabel("$y1$")
        ax[i][0].set_ylim((0, 1.2))
        ax[i][0].legend()
        ax[i][0].set_title(f"Solution y1 for h ={h_i}")
        ax[i][1].set_xlabel("$t$")
        ax[i][1].set_ylabel("$y2$")
        ax[i][1].set_ylim((-1, 2.5))
        ax[i][1].legend()
        ax[i][1].set_title(f"Solution y2 (y1') for h ={h_i}")


    plt.show()
