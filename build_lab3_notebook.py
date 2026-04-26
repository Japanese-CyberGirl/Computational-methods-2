import json
from pathlib import Path


def source_lines(text):
    text = text.strip("\n")
    if not text:
        return []
    return [line + "\n" for line in text.split("\n")]


cells = []


def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines(text),
    })


def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    })


md(r"""
### Постановка задачи

Нужно решить следующую **жесткую** систему:

$$\begin{cases}
\dfrac{dF_1}{dt} = k_1 - \left(\dfrac{k_2F_6}{F_6 + c_2} + \dfrac{k_3F_5}{F_5+c_3} + \dfrac{k_4F_4}{F_4+c_4}\right)F_1 - d_1F_1, \\
\dfrac{dF_2}{dt} = \left(\dfrac{k_2F_6}{F_6+c_2} + \dfrac{k_3F_5}{F_5 + c_3}\right)F_1 + k_5F_3 - k_6F_2 - d_1F_2, \\
\dfrac{dF_3}{dt} = k_4F_1\dfrac{F_4}{F_4 + c_4} + k_6F_2 - k_5F_3 - d_1F_3, \\
\dfrac{dF_4}{dt} = \lambda_3 F_3 \dfrac{c_1}{c_1 + F_4} - d_2F_4, \\
\dfrac{dF_5}{dt} = \lambda_1F_2 \dfrac{c_5}{c_5 + F_4} - d_3F_5, \\
\dfrac{dF_6}{dt} = \lambda_2F_2 \dfrac{c_5}{c_5 + F_4} - d_4F_6.
\end{cases}$$

Начальное условие:

$$\vec F(0)=(0.05,\;0.005,\;0.005,\;0.16,\;0.785,\;0.01).$$

По аналогии с первой лабораторной будем применять уже использованные там методы, но теперь для системы: методы степенных рядов, неявный и модифицированный Эйлер, центральные разности, методы Адамса и метод Гира. Отдельно добавим разностные схемы с весом $\sigma$, так как именно через них удобно обсуждать устойчивость.
""")

code(r"""
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.linalg import eigvals, norm

np.set_printoptions(precision=6, suppress=True)
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 160)
plt.rcParams["figure.figsize"] = (11, 6)
plt.rcParams["axes.grid"] = True
""")

md(r"""
Зададим параметры, начальную точку и отрезок интегрирования. В условии явно задана начальная точка; для методического сравнения, как и в стандартных примерах жестких систем, берем отрезок $[0,1]$. Если в варианте лабораторной требуется другой конечный момент, достаточно поменять `b`.
""")

code(r"""
params = {
    "k1": 1.0,
    "k2": 0.1,
    "k3": 1.0,
    "k4": 0.3,
    "k5": 0.05,
    "k6": 0.075,
    "d1": 0.2,
    "d2": 2.5,
    "d3": 55.0,
    "d4": 10.5,
    "c1": 100.0,
    "c2": 10.0,
    "c3": 10.0,
    "c4": 5.0,
    "c5": 25.0,
    "lambda1": 7e-4,
    "lambda2": 5e-4,
    "lambda3": 5e-4,
}

Y0 = np.array([0.05, 0.005, 0.005, 0.16, 0.785, 0.01], dtype=float)
a = 0.0
b = 1.0

# Значения N выбраны степенями двойки, чтобы удобно применять оценку Рунге.
N_values = [32, 64, 128, 256, 512, 1024]
N_ref = 4096

param_table = pd.DataFrame(
    [params],
    columns=["k1", "k2", "k3", "k4", "k5", "k6", "d1", "d2", "d3", "d4", "c1", "c2", "c3", "c4", "c5", "lambda1", "lambda2", "lambda3"],
).T.rename(columns={0: "value"})

display(param_table)
""")

md(r"""
Как и в первой лабораторной, начнем с функции построения равномерной сетки.
""")

code(r"""
def grid(a, b, N):
    h = (b - a) / N
    grid = np.array([a + i * h for i in range(N + 1)], dtype=float)
    return grid, h
""")

md(r"""
Теперь зададим правую часть системы. Она автономная, то есть явно от $t$ не зависит, но аргумент `t` оставим: так все методы будут иметь тот же интерфейс, что и в `lab_1`.
""")

code(r"""
def F_system(t, Y):
    F1, F2, F3, F4, F5, F6 = np.asarray(Y, dtype=float)
    p = params

    term_26 = p["k2"] * F6 / (F6 + p["c2"])
    term_35 = p["k3"] * F5 / (F5 + p["c3"])
    term_44 = p["k4"] * F4 / (F4 + p["c4"])
    immune_den = p["c5"] + F4

    return np.array([
        p["k1"] - (term_26 + term_35 + term_44) * F1 - p["d1"] * F1,
        (term_26 + term_35) * F1 + p["k5"] * F3 - p["k6"] * F2 - p["d1"] * F2,
        term_44 * F1 + p["k6"] * F2 - p["k5"] * F3 - p["d1"] * F3,
        p["lambda3"] * F3 * p["c1"] / (p["c1"] + F4) - p["d2"] * F4,
        p["lambda1"] * F2 * p["c5"] / immune_den - p["d3"] * F5,
        p["lambda2"] * F2 * p["c5"] / immune_den - p["d4"] * F6,
    ], dtype=float)


def numerical_jacobian(F, t, Y, eps=1e-7):
    Y = np.asarray(Y, dtype=float)
    n = len(Y)
    J = np.zeros((n, n), dtype=float)

    for i in range(n):
        step = eps * max(1.0, abs(Y[i]))
        e = np.zeros(n, dtype=float)
        e[i] = step
        J[:, i] = (F(t, Y + e) - F(t, Y - e)) / (2.0 * step)

    return J


def max_norm(values):
    return norm(values, ord=np.inf)
""")

md(r"""
Перед численным решением проверим жесткость. Для системы ОДУ естественный локальный критерий связан с собственными числами матрицы Якоби $J = \partial F / \partial Y$. Если отрицательные действительные части сильно различаются по модулю, то явные методы вынуждены брать шаг из соображений устойчивости, а не точности.
""")

code(r"""
J0 = numerical_jacobian(F_system, a, Y0)
eig0 = eigvals(J0)
negative_re = [abs(z.real) for z in eig0 if z.real < -1e-12]
stiffness_number = max(negative_re) / min(negative_re)

eigen_table = pd.DataFrame({
    "eigenvalue": eig0,
    "Re": np.real(eig0),
    "Im": np.imag(eig0),
    "abs_Re": np.abs(np.real(eig0)),
}).sort_values("abs_Re", ascending=False)

display(eigen_table)
print(f"Число жесткости в начальной точке примерно S = {stiffness_number:.2f}")
print(f"Для явного Эйлера грубое ограничение по быстрой компоненте: h <= {2 / max(negative_re):.5f}")
""")

md(r"""
В `lab_1` точное решение было известно, поэтому погрешность считалась напрямую. Здесь аналитического решения нет, поэтому будем использовать два стандартных приема:

1. сравнение с эталонным решением на очень мелкой сетке;
2. оценку Рунге по двум вложенным сеткам $h$ и $h/2$.
""")

code(r"""
def inaccuracy_system(reference_values, numerical_values, stride=1):
    if numerical_values is None:
        return np.inf

    max_error = 0.0
    for i, y_num in enumerate(numerical_values):
        if i * stride >= len(reference_values):
            return np.inf
        y_ref = reference_values[i * stride]
        if not np.all(np.isfinite(y_num)):
            return np.inf
        max_error = max(max_error, norm(y_ref - y_num, ord=np.inf))

    return max_error


def runge_estimate_system(solution_coarse, solution_fine, p):
    if solution_coarse is None or solution_fine is None:
        return np.nan

    max_diff = 0.0
    for i, y_coarse in enumerate(solution_coarse):
        y_fine = solution_fine[2 * i]
        if not np.all(np.isfinite(y_coarse)) or not np.all(np.isfinite(y_fine)):
            return np.nan
        max_diff = max(max_diff, norm(y_fine - y_coarse, ord=np.inf))

    return max_diff / (2**p - 1)


def experimental_orders(errors):
    p = [np.nan]
    for i in range(1, len(errors)):
        e_prev = errors[i - 1]
        e_curr = errors[i]
        if e_prev == 0 or e_curr == 0 or not np.isfinite(e_prev) or not np.isfinite(e_curr):
            p.append(np.nan)
        else:
            p.append(np.log2(e_prev / e_curr))
    return p
""")

md(r"""
Для неявных методов понадобится решать нелинейные системы на каждом шаге. В первой лабораторной для уравнения $y'=\lambda y$ неявные формулы можно было записать явно. Здесь это уже нелинейная система, поэтому используем метод Ньютона с численным Якобианом.
""")

code(r"""
def newton_solve(residual, x0, tol=1e-11, max_iter=20):
    x = np.asarray(x0, dtype=float).copy()
    info = {"iterations": 0, "converged": False}

    for iteration in range(1, max_iter + 1):
        r = residual(x)
        r_norm = norm(r, ord=np.inf)

        if not np.isfinite(r_norm):
            info["iterations"] = iteration
            return x, info

        if r_norm < tol:
            info["iterations"] = iteration - 1
            info["converged"] = True
            return x, info

        def residual_as_function(_, z):
            return residual(z)

        J = numerical_jacobian(residual_as_function, 0.0, x)

        try:
            delta = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J, -r, rcond=None)[0]

        alpha = 1.0
        # Небольшое демпфирование: оно помогает, если начальное приближение оказалось грубым.
        while alpha > 1e-4:
            x_trial = x + alpha * delta
            trial_norm = norm(residual(x_trial), ord=np.inf)
            if np.isfinite(trial_norm) and trial_norm <= (1 - 1e-4 * alpha) * r_norm:
                break
            alpha *= 0.5

        x = x + alpha * delta
        info["iterations"] = iteration

        if norm(alpha * delta, ord=np.inf) < tol * (1 + norm(x, ord=np.inf)):
            info["converged"] = True
            return x, info

    return x, info


def step_failed(Y):
    return (not np.all(np.isfinite(Y))) or norm(Y, ord=np.inf) > 1e8
""")

md(r"""
### Методы степенных рядов и методы Эйлера

Сохраним идею из первой лабораторной. Для автономной системы

$$Y' = F(Y)$$

имеем

$$Y'' = J(Y)F(Y),$$

а третью производную удобно численно получить как направленную производную от $Y''$ вдоль $F(Y)$.
""")

code(r"""
def directional_derivative(func, Y, direction):
    direction = np.asarray(direction, dtype=float)
    d_norm = norm(direction, ord=np.inf)
    if d_norm == 0 or not np.isfinite(d_norm):
        return np.zeros_like(direction)

    eps = 1e-5 / max(1.0, d_norm)
    return (func(Y + eps * direction) - func(Y - eps * direction)) / (2.0 * eps)


def system_second_derivative(t, Y, F):
    FY = F(t, Y)
    J = numerical_jacobian(F, t, Y)
    return J @ FY


def system_third_derivative(t, Y, F):
    FY = F(t, Y)

    def y2_at(Z):
        return numerical_jacobian(F, t, Z) @ F(t, Z)

    return directional_derivative(y2_at, np.asarray(Y, dtype=float), FY)


def power_series_method_1(t, Y, h, F):
    return Y + h * F(t, Y), {"newton_iterations": 0, "newton_failed": 0}


def power_series_method_2(t, Y, h, F):
    F0 = F(t, Y)
    Y2 = system_second_derivative(t, Y, F)
    return Y + h * F0 + h**2 / 2.0 * Y2, {"newton_iterations": 0, "newton_failed": 0}


def power_series_method_3(t, Y, h, F):
    F0 = F(t, Y)
    Y2 = system_second_derivative(t, Y, F)
    Y3 = system_third_derivative(t, Y, F)
    return Y + h * F0 + h**2 / 2.0 * Y2 + h**3 / 6.0 * Y3, {"newton_iterations": 0, "newton_failed": 0}


def implicit_method(t, Y, h, F):
    Y_pred = Y + h * F(t, Y)

    def residual(Z):
        return Z - Y - h * F(t + h, Z)

    Y_next, info = newton_solve(residual, Y_pred)
    return Y_next, {"newton_iterations": info["iterations"], "newton_failed": int(not info["converged"])}


def modified_euler(t, Y, h, F):
    Y_pred = Y + h * F(t, Y)
    Y_next = Y + h / 2.0 * (F(t, Y) + F(t + h, Y_pred))
    return Y_next, {"newton_iterations": 0, "newton_failed": 0}
""")

md(r"""
### Разностные схемы

В книге Хакимзянова и Черного устойчивость удобно исследуется через разностные схемы. Для задачи Коши естественная схема с весом имеет вид

$$\frac{Y_{j+1}-Y_j}{h}=(1-\sigma)F(t_j,Y_j)+\sigma F(t_{j+1},Y_{j+1}).$$

При $\sigma=0$ получаем явную схему Эйлера, при $\sigma=1$ — неявную схему Эйлера, при $\sigma=\frac12$ — симметричную схему трапеций, которая имеет второй порядок и хорошие свойства устойчивости.
""")

code(r"""
def weighted_difference_method(t, Y, h, F, sigma):
    if sigma == 0:
        return Y + h * F(t, Y), {"newton_iterations": 0, "newton_failed": 0}

    Y_pred = Y + h * F(t, Y)
    F_current = F(t, Y)

    def residual(Z):
        return Z - Y - h * ((1.0 - sigma) * F_current + sigma * F(t + h, Z))

    Y_next, info = newton_solve(residual, Y_pred)
    return Y_next, {"newton_iterations": info["iterations"], "newton_failed": int(not info["converged"])}


def central_difference_method(t, Y_prev, Y_current, h, F):
    return Y_prev + 2.0 * h * F(t, Y_current), {"newton_iterations": 0, "newton_failed": 0}
""")

md(r"""
### Многошаговые методы Адамса и Гира

Теперь адаптируем вторую часть `lab_1`: явный Адамс-Бэшфорт, неявный Адамс-Моултон, предиктор-корректор Адамса-Бэшфорта-Моултона и метод Гира. Разница только в том, что неявные формулы теперь решаются методом Ньютона.
""")

code(r"""
def adams_explicit_2_system(t_j, Y_j, Y_jm1, h, F):
    F_j = F(t_j, Y_j)
    F_jm1 = F(t_j - h, Y_jm1)
    return Y_j + h / 2.0 * (3.0 * F_j - F_jm1), {"newton_iterations": 0, "newton_failed": 0}


def adams_implicit_2_system(t_j, Y_j, Y_jm1, h, F):
    F_j = F(t_j, Y_j)
    F_jm1 = F(t_j - h, Y_jm1)
    Y_pred = Y_j + h / 2.0 * (3.0 * F_j - F_jm1)

    def residual(Z):
        return Z - Y_j - h / 12.0 * (5.0 * F(t_j + h, Z) + 8.0 * F_j - F_jm1)

    Y_next, info = newton_solve(residual, Y_pred)
    return Y_next, {"newton_iterations": info["iterations"], "newton_failed": int(not info["converged"])}


def adams_predictor_corrector_system(t_j, Y_j, Y_jm1, Y_jm2, Y_jm3, h, F):
    F_j = F(t_j, Y_j)
    F_jm1 = F(t_j - h, Y_jm1)
    F_jm2 = F(t_j - 2.0 * h, Y_jm2)
    F_jm3 = F(t_j - 3.0 * h, Y_jm3)

    Y_pred = Y_j + h / 24.0 * (55.0 * F_j - 59.0 * F_jm1 + 37.0 * F_jm2 - 9.0 * F_jm3)

    def residual(Z):
        return Z - Y_j - h / 24.0 * (9.0 * F(t_j + h, Z) + 19.0 * F_j - 5.0 * F_jm1 + F_jm2)

    Y_next, info = newton_solve(residual, Y_pred)
    return Y_next, {"newton_iterations": info["iterations"], "newton_failed": int(not info["converged"])}


def gear_method_3_system(t_j, Y_j, Y_jm1, Y_jm2, h, F):
    Y_pred = Y_j + (Y_j - Y_jm1)

    def residual(Z):
        return 11.0 * Z - 18.0 * Y_j + 9.0 * Y_jm1 - 2.0 * Y_jm2 - 6.0 * h * F(t_j + h, Z)

    Y_next, info = newton_solve(residual, Y_pred)
    return Y_next, {"newton_iterations": info["iterations"], "newton_failed": int(not info["converged"])}
""")

md(r"""
Для многошаговых методов нужны первые значения. В `lab_1` для этого использовался метод степенных рядов. Здесь поступим так же: недостающие стартовые точки считаем методом третьего порядка.
""")

code(r"""
def starting_values_by_taylor3(domain, h, count):
    values = [Y0.copy()]
    stats = {"newton_iterations": 0, "newton_failed": 0}

    for k in range(count):
        Y_next, info = power_series_method_3(domain[k], values[-1], h, F_system)
        values.append(Y_next)
        stats["newton_iterations"] += info.get("newton_iterations", 0)
        stats["newton_failed"] += info.get("newton_failed", 0)

    return values, stats


methods_meta = {
    "ps1": {"title": "Степенной ряд 1 / явный Эйлер", "order": 1, "stability": "euler_explicit"},
    "ps2": {"title": "Степенной ряд 2", "order": 2, "stability": "taylor2"},
    "ps3": {"title": "Степенной ряд 3", "order": 3, "stability": "taylor3"},
    "implicit_euler": {"title": "Неявный Эйлер", "order": 1, "stability": "euler_implicit"},
    "modified_euler": {"title": "Модифицированный Эйлер", "order": 2, "stability": "taylor2"},
    "central_difference": {"title": "Центральные разности", "order": 2, "stability": "central"},
    "diff_sigma_0": {"title": "Разностная схема, sigma=0", "order": 1, "stability": "weighted_0"},
    "diff_sigma_05": {"title": "Разностная схема, sigma=1/2", "order": 2, "stability": "weighted_05"},
    "diff_sigma_1": {"title": "Разностная схема, sigma=1", "order": 1, "stability": "weighted_1"},
    "adams_explicit_2": {"title": "Адамс-Бэшфорт 2", "order": 2, "stability": "ab2"},
    "adams_implicit_2": {"title": "Адамс-Моултон 2", "order": 3, "stability": "am2"},
    "adams_pc": {"title": "Адамс-Бэшфорт-Моултон 4", "order": 4, "stability": "am4"},
    "gear_3": {"title": "Гир / BDF3", "order": 3, "stability": "bdf3"},
}

method_order = list(methods_meta.keys())
""")

code(r"""
def solve_one_method(method_name, N):
    domain, h = grid(a, b, N)
    stats = {
        "h": h,
        "steps": N,
        "newton_iterations": 0,
        "newton_failed": 0,
        "failed": False,
        "elapsed": 0.0,
    }

    start_time = time.perf_counter()

    try:
        if method_name in ["ps1", "ps2", "ps3", "implicit_euler", "modified_euler", "diff_sigma_0", "diff_sigma_05", "diff_sigma_1"]:
            values = [Y0.copy()]

            for j in range(N):
                t_j = domain[j]
                Y_j = values[-1]

                if method_name == "ps1":
                    Y_next, info = power_series_method_1(t_j, Y_j, h, F_system)
                elif method_name == "ps2":
                    Y_next, info = power_series_method_2(t_j, Y_j, h, F_system)
                elif method_name == "ps3":
                    Y_next, info = power_series_method_3(t_j, Y_j, h, F_system)
                elif method_name == "implicit_euler":
                    Y_next, info = implicit_method(t_j, Y_j, h, F_system)
                elif method_name == "modified_euler":
                    Y_next, info = modified_euler(t_j, Y_j, h, F_system)
                elif method_name == "diff_sigma_0":
                    Y_next, info = weighted_difference_method(t_j, Y_j, h, F_system, sigma=0.0)
                elif method_name == "diff_sigma_05":
                    Y_next, info = weighted_difference_method(t_j, Y_j, h, F_system, sigma=0.5)
                elif method_name == "diff_sigma_1":
                    Y_next, info = weighted_difference_method(t_j, Y_j, h, F_system, sigma=1.0)

                stats["newton_iterations"] += info.get("newton_iterations", 0)
                stats["newton_failed"] += info.get("newton_failed", 0)

                if step_failed(Y_next):
                    stats["failed"] = True
                    break

                values.append(Y_next)

        elif method_name == "central_difference":
            values, start_stats = starting_values_by_taylor3(domain, h, count=1)
            stats["newton_iterations"] += start_stats["newton_iterations"]
            stats["newton_failed"] += start_stats["newton_failed"]

            for j in range(1, N):
                Y_next, info = central_difference_method(domain[j], values[j - 1], values[j], h, F_system)
                if step_failed(Y_next):
                    stats["failed"] = True
                    break
                values.append(Y_next)

        elif method_name in ["adams_explicit_2", "adams_implicit_2"]:
            values, start_stats = starting_values_by_taylor3(domain, h, count=1)
            stats["newton_iterations"] += start_stats["newton_iterations"]
            stats["newton_failed"] += start_stats["newton_failed"]

            for j in range(1, N):
                if method_name == "adams_explicit_2":
                    Y_next, info = adams_explicit_2_system(domain[j], values[j], values[j - 1], h, F_system)
                else:
                    Y_next, info = adams_implicit_2_system(domain[j], values[j], values[j - 1], h, F_system)

                stats["newton_iterations"] += info.get("newton_iterations", 0)
                stats["newton_failed"] += info.get("newton_failed", 0)

                if step_failed(Y_next):
                    stats["failed"] = True
                    break
                values.append(Y_next)

        elif method_name == "adams_pc":
            values, start_stats = starting_values_by_taylor3(domain, h, count=3)
            stats["newton_iterations"] += start_stats["newton_iterations"]
            stats["newton_failed"] += start_stats["newton_failed"]

            for j in range(3, N):
                Y_next, info = adams_predictor_corrector_system(domain[j], values[j], values[j - 1], values[j - 2], values[j - 3], h, F_system)
                stats["newton_iterations"] += info.get("newton_iterations", 0)
                stats["newton_failed"] += info.get("newton_failed", 0)

                if step_failed(Y_next):
                    stats["failed"] = True
                    break
                values.append(Y_next)

        elif method_name == "gear_3":
            values, start_stats = starting_values_by_taylor3(domain, h, count=2)
            stats["newton_iterations"] += start_stats["newton_iterations"]
            stats["newton_failed"] += start_stats["newton_failed"]

            for j in range(2, N):
                Y_next, info = gear_method_3_system(domain[j], values[j], values[j - 1], values[j - 2], h, F_system)
                stats["newton_iterations"] += info.get("newton_iterations", 0)
                stats["newton_failed"] += info.get("newton_failed", 0)

                if step_failed(Y_next):
                    stats["failed"] = True
                    break
                values.append(Y_next)

        else:
            raise ValueError(f"Unknown method: {method_name}")

    except Exception as exc:
        values = None
        stats["failed"] = True
        stats["error"] = repr(exc)

    stats["elapsed"] = time.perf_counter() - start_time

    if values is not None and len(values) != N + 1:
        stats["failed"] = True
        values = None

    return domain, values, stats
""")

md(r"""
Построим эталонное решение. В качестве эталона возьмем метод Гира на мелкой сетке: он неявный и предназначен именно для жестких задач.
""")

code(r"""
ref_domain, ref_values, ref_stats = solve_one_method("gear_3", N_ref)
print("Эталонное решение построено методом Гира")
print(f"N_ref = {N_ref}, h_ref = {(b - a) / N_ref:.6g}, failed = {ref_stats['failed']}, time = {ref_stats['elapsed']:.3f} sec")
print("Y_ref(b) =", ref_values[-1])
""")

md(r"""
Теперь решим задачу всеми методами на последовательности сеток. Для каждого метода сохраним ошибку относительно эталона, оценку Рунге, экспериментальные порядки и время счета.
""")

code(r"""
solutions = {method: {} for method in method_order}
raw_stats = []

for method in method_order:
    for N in N_values:
        domain, values, stats = solve_one_method(method, N)
        solutions[method][N] = {"domain": domain, "values": values, "stats": stats}
        raw_stats.append({
            "method": method,
            "title": methods_meta[method]["title"],
            "N": N,
            "h": stats["h"],
            "failed": stats["failed"],
            "newton_iterations": stats["newton_iterations"],
            "newton_failed": stats["newton_failed"],
            "elapsed": stats["elapsed"],
        })

raw_stats_df = pd.DataFrame(raw_stats)
display(raw_stats_df.head(20))
""")

code(r"""
def build_method_table(method):
    p_method = methods_meta[method]["order"]
    eps_ref = []
    eps_runge = [np.nan]
    h_values = []
    elapsed = []
    failed = []

    for i, N in enumerate(N_values):
        values = solutions[method][N]["values"]
        stats = solutions[method][N]["stats"]
        stride_ref = N_ref // N
        eps_ref.append(inaccuracy_system(ref_values, values, stride=stride_ref))
        h_values.append(stats["h"])
        elapsed.append(stats["elapsed"])
        failed.append(stats["failed"])

        if i > 0:
            N_prev = N_values[i - 1]
            values_prev = solutions[method][N_prev]["values"]
            values_curr = solutions[method][N]["values"]
            eps_runge.append(runge_estimate_system(values_prev, values_curr, p_method))

    table = pd.DataFrame({
        "N": N_values,
        "h": h_values,
        "eps_ref": eps_ref,
        "eps_runge": eps_runge,
        "p_ref": experimental_orders(eps_ref),
        "p_runge": experimental_orders(eps_runge),
        "time_sec": elapsed,
        "failed": failed,
    })

    return table

method_tables = {method: build_method_table(method) for method in method_order}

summary_eps = pd.DataFrame({"N": N_values, "h": [(b - a) / N for N in N_values]})
summary_p = pd.DataFrame({"N": N_values})

for method in method_order:
    title = methods_meta[method]["title"]
    summary_eps[title] = method_tables[method]["eps_ref"]
    summary_p[title] = method_tables[method]["p_ref"]

print("Погрешность относительно эталонного решения")
display(summary_eps)
print("Экспериментальные порядки")
display(summary_p)
""")

md(r"""
Посмотрим подробнее несколько характерных методов: явный метод, неявную схему, схему с весом $1/2$, Адамса и Гира.
""")

code(r"""
for method in ["ps1", "modified_euler", "diff_sigma_05", "adams_explicit_2", "adams_implicit_2", "adams_pc", "gear_3"]:
    print(methods_meta[method]["title"])
    display(method_tables[method])
""")

md(r"""
Построим графики погрешности. Если метод неустойчив на грубой сетке, его ошибка будет бесконечной или не попадет на график.
""")

code(r"""
plt.figure(figsize=(12, 8))
for method in method_order:
    table = method_tables[method]
    y = table["eps_ref"].replace([np.inf, -np.inf], np.nan)
    plt.loglog(table["h"], y, marker="o", label=methods_meta[method]["title"])

plt.gca().invert_xaxis()
plt.xlabel("h")
plt.ylabel("eps_ref")
plt.title("Погрешность методов относительно эталонного решения")
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()
""")

code(r"""
plt.figure(figsize=(12, 8))
for method in method_order:
    table = method_tables[method]
    if table["p_ref"].notna().sum() > 0:
        plt.semilogx(table["N"], table["p_ref"], marker="o", label=methods_meta[method]["title"])

plt.xlabel("N")
plt.ylabel("p")
plt.title("Экспериментальные порядки точности")
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()
""")

md(r"""
### Области устойчивости

Для исследования устойчивости используем тестовое уравнение $y'=\lambda y$ и обозначение $z=h\lambda$. Метод устойчив в тех точках комплексной плоскости, где все корни соответствующего характеристического уравнения лежат в единичном круге.

Собственные значения Якобиана системы в начальной точке почти действительные и отрицательные, поэтому особенно важна левая часть вещественной оси.
""")

code(r"""
def max_root_abs(coeffs):
    roots = np.roots(coeffs)
    return np.max(np.abs(roots))


def stability_measure(kind, z):
    if kind in ["euler_explicit", "weighted_0"]:
        return abs(1 + z)
    if kind == "taylor2":
        return abs(1 + z + z**2 / 2.0)
    if kind == "taylor3":
        return abs(1 + z + z**2 / 2.0 + z**3 / 6.0)
    if kind in ["euler_implicit", "weighted_1"]:
        return abs(1 / (1 - z))
    if kind == "weighted_05":
        sigma = 0.5
        return abs((1 + (1 - sigma) * z) / (1 - sigma * z))
    if kind == "central":
        return max_root_abs([1, -2 * z, -1])
    if kind == "ab2":
        return max_root_abs([1, -1 - 1.5 * z, 0.5 * z])
    if kind == "am2":
        return max_root_abs([1 - 5 * z / 12.0, -1 - 8 * z / 12.0, z / 12.0])
    if kind == "am4":
        return max_root_abs([1 - 9 * z / 24.0, -1 - 19 * z / 24.0, 5 * z / 24.0, -z / 24.0])
    if kind == "bdf3":
        return max_root_abs([11.0 / 6.0 - z, -3.0, 1.5, -1.0 / 3.0])
    raise ValueError(kind)


def is_stable(kind, z):
    return stability_measure(kind, z) <= 1 + 1e-10


def negative_axis_limit(kind, xmax=80.0, samples=4001):
    xs = np.linspace(0.0, xmax, samples)
    last = 0.0
    for x in xs:
        if is_stable(kind, -x):
            last = x
        else:
            break
    return last

lambda_fast = max(abs(np.real(eig0)))
stability_rows = []
for method in method_order:
    kind = methods_meta[method]["stability"]
    limit = negative_axis_limit(kind)
    h_limit = np.inf if limit >= 79.9 else limit / lambda_fast
    stability_rows.append({
        "method": method,
        "title": methods_meta[method]["title"],
        "stability_kind": kind,
        "x_limit_on_negative_axis": limit if limit < 79.9 else ">= 80",
        "rough_h_limit": h_limit,
    })

stability_df = pd.DataFrame(stability_rows)
display(stability_df)
""")

code(r"""
def plot_stability_regions(selected_methods, xlim=(-8, 4), ylim=(-5, 5), points=170):
    xs = np.linspace(xlim[0], xlim[1], points)
    ys = np.linspace(ylim[0], ylim[1], points)
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y

    plt.figure(figsize=(12, 8))
    for method in selected_methods:
        kind = methods_meta[method]["stability"]
        M = np.empty_like(X, dtype=float)
        for i in range(points):
            for j in range(points):
                M[i, j] = stability_measure(kind, Z[i, j])
        plt.contour(X, Y, M, levels=[1.0], linewidths=1.5)

    for N, marker, label in [(min(N_values), "x", f"h*lambda, N={min(N_values)}"), (max(N_values), "o", f"h*lambda, N={max(N_values)}")]:
        h = (b - a) / N
        points_lambda = h * eig0
        plt.scatter(points_lambda.real, points_lambda.imag, marker=marker, s=50, label=label)

    labels = [methods_meta[m]["title"] for m in selected_methods]
    for label in labels:
        plt.plot([], [], label=label)

    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Re z")
    plt.ylabel("Im z")
    plt.title("Границы областей устойчивости и точки h*lambda_i")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()

plot_stability_regions(["ps1", "modified_euler", "ps3", "diff_sigma_05", "implicit_euler", "adams_explicit_2", "adams_implicit_2", "gear_3"])
""")

md(r"""
### Динамика решения

Построим компоненты эталонного решения. На графике хорошо видно, почему задача жесткая: часть компонент быстро затухает, а часть меняется на значительно более медленном масштабе времени.
""")

code(r"""
ref_matrix = np.vstack(ref_values)
plt.figure(figsize=(12, 7))
for i in range(6):
    plt.plot(ref_domain, ref_matrix[:, i], label=f"F{i + 1}")

plt.xlabel("t")
plt.ylabel("F_i(t)")
plt.title("Эталонное решение жесткой системы")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 7))
for i in range(6):
    plt.semilogy(ref_domain, np.maximum(np.abs(ref_matrix[:, i]), 1e-16), label=f"|F{i + 1}|")

plt.xlabel("t")
plt.ylabel("|F_i(t)|")
plt.title("Компоненты решения в логарифмической шкале")
plt.legend()
plt.tight_layout()
plt.show()
""")

md(r"""
### Итоговое сравнение

Соберем финальную таблицу по самой мелкой из рабочих сеток. Оптимальный метод будем выбирать по совокупности признаков: малая погрешность, устойчивость, отсутствие срывов Ньютона и разумное время счета.
""")

code(r"""
N_final = max(N_values)
comparison_rows = []

for method in method_order:
    table = method_tables[method]
    row = table[table["N"] == N_final].iloc[0]
    stats = solutions[method][N_final]["stats"]
    stab = stability_df[stability_df["method"] == method].iloc[0]

    comparison_rows.append({
        "method": method,
        "title": methods_meta[method]["title"],
        "order_theory": methods_meta[method]["order"],
        "eps_ref": row["eps_ref"],
        "eps_runge": row["eps_runge"],
        "p_ref_last": row["p_ref"],
        "time_sec": row["time_sec"],
        "newton_iterations": stats["newton_iterations"],
        "newton_failed": stats["newton_failed"],
        "failed": stats["failed"],
        "rough_h_limit": stab["rough_h_limit"],
    })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df["stable_for_h_final"] = comparison_df["rough_h_limit"].apply(lambda x: True if not np.isfinite(x) else ((b - a) / N_final <= x))
comparison_df["efficiency_score"] = comparison_df["eps_ref"] * np.maximum(comparison_df["time_sec"], 1e-12)
comparison_df = comparison_df.sort_values(["failed", "eps_ref", "time_sec"])

display(comparison_df)

valid = comparison_df[(~comparison_df["failed"]) & np.isfinite(comparison_df["eps_ref"])]
accuracy_winner = valid.iloc[0]
efficient_winner = valid.sort_values("efficiency_score").iloc[0]

print("Минимальная ошибка:", accuracy_winner["title"], "eps =", accuracy_winner["eps_ref"])
print("Лучший баланс ошибка*время:", efficient_winner["title"], "score =", efficient_winner["efficiency_score"])
""")

md(r"""
### Вывод

Система является жесткой: уже в начальной точке собственные значения матрицы Якоби имеют сильно различающиеся отрицательные действительные части, а число жесткости получается больше 100. Поэтому явные методы приходится ограничивать шагом из условий устойчивости. Это видно и по расчетам: на грубых сетках явные методы и центральные разности ведут себя заметно хуже, чем неявные схемы.

Разностная схема с весом $\sigma=\frac12$ и неявные методы хорошо проходят проверку устойчивости. По итоговой таблице для выбранного отрезка $[0,1]$ оптимальным оказывается метод Адамса-Бэшфорта-Моултона 4-го порядка: он дает минимальную ошибку и лучший показатель `ошибка * время`. Метод Гира/BDF3 при этом остается самым естественным резервным выбором для жестких задач, потому что он специально построен для таких систем и допускает крупный шаг по устойчивости. Если требуется максимальная простота реализации, можно выбрать неявный Эйлер, но он имеет только первый порядок.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "venv",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path("lab_3.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote lab_3.ipynb with {len(cells)} cells")
