import numpy as np
from numpy import pi, float_power as fpower, sin, cos, fabs
from enviroment import Test, Material

to_norm = 1
stef_bolc = 5.67036713 if to_norm else 1.0

__Hs = lambda t: stef_bolc * fpower(t, 4)

w = 100.0 if to_norm else 1.0


def CreateBoundFunc(func_num: int, limits: list, material: Material) -> list:

    coef = 0.01 * (material.tmax - material.tmin) if to_norm else 1.0
    d = 0.01 * material.tmin if to_norm else 1.0
    tcc = material.thermal_cond

    u_test = [
        lambda x, y: 0.5 * coef * (x + y) + d,
        lambda x, y: 0.5 * coef * (x**2 + y) + d,
        lambda x, y: 0.5 * coef * (x**2 + y**2) + d,
        lambda x, y: coef * x**4 * y**3 + d,
        lambda x, y: coef * cos(x) * sin(y) + d,
        lambda x, y: coef * sin(pi * x) * sin(pi * y) + d,
    ]

    du = [
        [lambda x, y: 0.5 * coef, lambda x, y: 0.5 * coef],
        [lambda x, y: 0.5 * coef * 2 * x, lambda x, y: 0.5 * coef],
        [lambda x, y: 0.5 * coef * 2 * x, lambda x, y: 0.5 * coef * 2 * y],
        [
            lambda x, y: coef * 4 * x**3 * y**3,
            lambda x, y: coef * 3 * x**4 * y**2,
        ],
        [lambda x, y: -coef * sin(x) * sin(y), lambda x, y: coef * cos(x) * cos(y)],
        [
            lambda x, y: pi * coef * cos(pi * x) * sin(pi * y),
            lambda x, y: pi * coef * sin(pi * x) * cos(pi * y),
        ],
    ]

    d2u = [
        [lambda x, y: 0 * x * y, lambda x, y: 0 * x * y],
        [lambda x, y: 0.5 * coef * 2, lambda x, y: 0 * x * y],
        [
            lambda x, y: 0.5 * coef * 2 * (1.0 + x - x),
            lambda x, y: 0.5 * coef * 2 * (1.0 + x - x),
        ],
        [lambda x, y: coef * 12 * x**2 * y**3, lambda x, y: coef * 6 * x**4 * y],
        [lambda x, y: -coef * cos(x) * sin(y), lambda x, y: -coef * cos(x) * sin(y)],
        [
            lambda x, y: -fpower(pi, 2) * coef * sin(pi * x) * sin(pi * y),
            lambda x, y: -fpower(pi, 2) * coef * sin(pi * x) * sin(pi * y),
        ],
    ]

    normed = w
    f = lambda x, y: tcc * normed * (-d2u[func_num][0](x, y) - d2u[func_num][1](x, y))
    u, dux, duy = u_test[func_num], *du[func_num]

    def g(x, y):
        if x == 0:
            if y == 0:
                return 0.5 * (
                    2 * stef_bolc * fabs(u(0, 0)) * fpower(u(0, 0), 3)
                    - tcc * normed * dux(0, 0)
                    - tcc * normed * duy(0, 0)
                )
            elif y == limits[1]:
                return 0.5 * (
                    2 * stef_bolc * fabs(u(0, limits[1])) * fpower(u(0, limits[1]), 3)
                    - tcc * normed * dux(0, limits[1])
                    + tcc * normed * duy(0, limits[1])
                )
            else:
                return stef_bolc * fabs(u(x, y)) * fpower(
                    u(x, y), 3
                ) - tcc * normed * dux(x, y)
        elif x == limits[0]:
            if y == 0:
                return 0.5 * (
                    2 * stef_bolc * fabs(u(limits[0], 0)) * fpower(u(limits[0], 0), 3)
                    + tcc * normed * dux(limits[0], 0)
                    - tcc * normed * duy(limits[0], 0)
                )
            elif y == limits[1]:
                return 0.5 * (
                    2
                    * stef_bolc
                    * fabs(u(limits[0], limits[1]))
                    * fpower(u(limits[0], limits[1]), 3)
                    + tcc * normed * dux(limits[0], limits[1])
                    + tcc * normed * duy(limits[0], limits[1])
                )
            else:
                return stef_bolc * fabs(u(x, y)) * fpower(
                    u(x, y), 3
                ) + tcc * normed * dux(x, y)
        elif y == 0:
            return stef_bolc * fabs(u(x, y)) * fpower(u(x, y), 3) - tcc * normed * duy(
                x, y
            )
        elif y == limits[1]:
            return stef_bolc * fabs(u(x, y)) * fpower(u(x, y), 3) + tcc * normed * duy(
                x, y
            )
        return 0

    return [f, g]

def g_for_comp(x, y, tmin, tmax):
    coef = tmax - tmin
    d = tmin
    if y == 0.0 and 0.0 <= x <= 0.3:
        return __Hs(d + coef * sin(pi * x / 0.3))
    elif y == 1.0 and 0.7 <= x <= 1.0:
        return __Hs(d + coef * sin(pi * (1.0 - x) / 0.3))
    else: return __Hs(tmin)

def SpecialBoundFunc(func_num: int, limits: list, material: Material) -> list:
    tmax, tmin = material.tmax * 0.01, material.tmin * 0.01
    coef = tmax - tmin
    d = tmin
    gs = [
        #lambda x, y: __Hs(d + coef * sin(pi * x / 0.3))
        #if y == 0.0 and 0.0 <= x <= 0.3
        #else __Hs(tmin),
        lambda x, y: g_for_comp(x, y, tmin, tmax),
        lambda x, y: __Hs(tmax) if y == 0.0 else __Hs(tmin),
        lambda x, y: __Hs(tmin),
        lambda x, y: __Hs(tmin),
    ]
    gs.append(gs[0])
    gs.append(gs[1])
    gs.append(lambda x, y: __Hs(tmax) if (y == 0.0 and np.fabs(x - 0.5) <= 0.02) else 0.0)

    gcirc = (
        lambda x0, y0, r0: lambda x, y: np.hypot(x - x0, y - y0) / r0
        if np.hypot(x - x0, y - y0) <= r0
        else 0.0
    )
    n_circ = 8
    r = 0.5
    r_s = 0.15
    circles = [
        gcirc(
            r * cos(2 * pi * i / n_circ) + 0.5, r * sin(2 * pi * i / n_circ) + 0.5, r_s
        )
        for i in range(n_circ)
    ]
    check_dot = lambda x, y: sum(circ(x, y) for circ in circles) == 0.0
    _f = (
        lambda x, y: __Hs(
            1.3 * tmax * cos(0.5 * pi * sum(circ(x, y) for circ in circles))
        )
        if not check_dot(x, y)
        else 0.0
    )

    f = [
        lambda x, y: 0,
        lambda x, y: 0,
        lambda x, y: __Hs(tmax * cos(0.5 * pi * np.hypot(x - 0.5, y - 0.5) / 0.25))
        if np.hypot(x - 0.5, y - 0.5) <= 0.25
        else 0.0,
        lambda x, y: _f(x, y),
    ]
    f.append(f[-1])
    f.append(f[2])
    f.append(f[0])
    return [f[func_num], gs[func_num]]


def getBoundary(test: Test, f_off=False, g_off=False) -> list:
    func_num, material, cells, cell_size, limits = list(test._asdict().values())[1:]
    h = limits[0] / (cells[0] * (cell_size - 1))
    if func_num < 0:
        f, g = SpecialBoundFunc(-1 - func_num, limits, material)
    else:
        f, g = CreateBoundFunc(func_num, limits, material)
    if g_off:
        g = lambda x, y: 0.0
    if f_off:
        f = lambda x, y: 0.0

    F = np.zeros((cells[0], cells[1], cell_size, cell_size))
    G = np.zeros_like(F)
    for i in range(cells[0]):
        for j in range(cells[1]):
            for i2 in range(cell_size):
                for j2 in range(cell_size):
                    F[i, j, i2, j2] = f(
                        (i * (cell_size - 1) + i2) * h, (j * (cell_size - 1) + j2) * h
                    )
    for k in range(cells[0]):
        for k2 in range(cell_size):
            G[k, 0, k2, 0] = g((k * (cell_size - 1) + k2) * h, 0)
            G[k, -1, k2, -1] = g((k * (cell_size - 1) + k2) * h, limits[1])
    for k in range(cells[1]):
        for k2 in range(cell_size):
            G[0, k, 0, k2] = g(0, (k * (cell_size - 1) + k2) * h)
            G[-1, k, -1, k2] = g(limits[0], (k * (cell_size - 1) + k2) * h)

    return [F, G]
