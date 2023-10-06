import numpy as np
import pickle
from numpy import float_power as fpower, fabs
from scipy.interpolate import LinearNDInterpolator
from time import strftime
import functools

from boundary import stef_bolc, w
from enviroment import Material

from custom_numerics.wraps import timer
from custom_numerics.draw import draw1D, draw2D, drawHeatmap


class BalanceScheme:
    def __init__(
        self,
        F: np.ndarray,
        G: np.ndarray,
        material: Material,
        limits: list,
        cells: list,
        cell_size: int,
        name: str,
        folder="",
        log=False
    ):
        self.F, self.G = F, G

        self.material = material
        self.tcc_n = material.thermal_cond * w
        self.zlim = []  # [material.tmin, material.tmax]

        self.U = 0.5 * (material.tmin + material.tmax) * np.ones_like(self.F)
        self.dU = np.zeros_like(self.F)

        self.limits = limits
        self.cells = cells
        self.cell_size = cell_size
        self.sigma = stef_bolc
        self.h = limits[0] / ((cell_size - 1) * cells[0])
        self.h2 = fpower(self.h, 2)
        self.newt_err = []
        self.cg_err = []

        self.test_name = name
        self.folder = folder
        self.log_file = None
        self.logFlag = log

    def dot(self, u: np.ndarray) -> np.ndarray:
        res = np.zeros_like(u)
        h = self.h
        h2 = self.h2
        sigma = self.sigma
        tcc_n_h2 = self.tcc_n / h2
        
        HeatStream = lambda v: sigma * fabs(v) * fpower(v, 3)

        res[::, ::, 1:-1, 1:-1] = -tcc_n_h2*(u[::, ::, 2:, 1:-1] + u[::, ::, :-2, 1:-1] - 4*u[::, ::, 1:-1, 1:-1] + \
            u[::, ::, 1:-1, 2:] + u[::, ::, 1:-1, :-2])
        
        res[::, ::, 0, 1:-1] = -2*tcc_n_h2*(u[::, ::, 1, 1:-1] - u[::, ::, 0, 1:-1]) + \
                2*HeatStream(u[::, ::, 0, 1:-1])/h - \
                    tcc_n_h2*(u[::, ::, 0, 2:] - 2*u[::, ::, 0, 1:-1] + u[::, ::, 0, :-2])
        res[::, ::, 1:-1, 0] = -2*tcc_n_h2*(u[::, ::, 1:-1, 1] - u[::, ::, 1:-1, 0]) + \
                2*HeatStream(u[::, ::, 1:-1, 0])/h - \
                    tcc_n_h2*(u[::, ::, 2:, 0] - 2*u[::, ::, 1:-1, 0] + u[::, ::, :-2, 0])
        res[::, ::, -1, 1:-1] = 2*tcc_n_h2*(u[::, ::, -1, 1:-1] - u[::, ::, -2, 1:-1]) + \
                2*HeatStream(u[::, ::, -1, 1:-1])/h - \
                    tcc_n_h2*(u[::, ::, -1, 2:] - 2*u[::, ::, -1, 1:-1] + u[::, ::, -1, :-2])
        res[::, ::, 1:-1, -1] = 2*tcc_n_h2*(u[::, ::, 1:-1, -1] - u[::, ::, 1:-1, -2]) + \
                2*HeatStream(u[::, ::, 1:-1, -1])/h - \
                    tcc_n_h2*(u[::, ::, 2:, -1] - 2*u[::, ::, 1:-1, -1] + u[::, ::, :-2, -1])

        res[::, ::, 0, 0] = 4*HeatStream(u[::, ::, 0, 0])/h - \
            2*tcc_n_h2*(u[::, ::, 0, 1] - 2*u[::, ::, 0, 0] + u[::, ::, 1, 0])
        res[::, ::, 0, -1] = 4*HeatStream(u[::, ::, 0, -1])/h - \
            2*tcc_n_h2*(u[::, ::, 0, -2] - 2*u[::, ::, 0, -1] + u[::, ::, 1, -1])
        res[::, ::, -1, -1] = 4*HeatStream(u[::, ::, -1, -1])/h - \
            2*tcc_n_h2*(u[::, ::, -1, -2] - 2*u[::, ::, -1, -1] + u[::, ::, -2, -1])
        res[::, ::, -1, 0] = 4*HeatStream(u[::, ::, -1, 0])/h - \
            2*tcc_n_h2*(u[::, ::, -1, 1] - 2*u[::, ::, -1, 0] + u[::, ::, -2, 0])
        
        _G = self.G.copy()

        # inside joints
        _G[1:, ::, 0, 1:-1] = HeatStream(u[:-1, ::, -1, 1:-1])
        _G[:-1, ::, -1, 1:-1] = HeatStream(u[1:, ::, 0, 1:-1])
        _G[::, 1:, 1:-1, 0] = HeatStream(u[::, :-1, 1:-1, -1])
        _G[::, :-1, 1:-1, -1] = HeatStream(u[::, 1:, 1:-1, 0])

        # inside corners
        _G[:-1, :-1, -1, -1] = HeatStream(u[1:, :-1, 0, -1]) + HeatStream(u[:-1, 1:, -1, 0])
        _G[1:, :-1, 0, -1] = HeatStream(u[:-1, :-1, -1, -1]) + HeatStream(u[1:, 1:, 0, 0])
        _G[1:, 1:, 0, 0] = HeatStream(u[1:, :-1, 0, -1]) + HeatStream(u[:-1, 1:, -1, 0])
        _G[:-1, 1:, -1, 0] = HeatStream(u[1:, 1:, 0, 0]) + HeatStream(u[:-1, :-1, -1, -1])

        # side corners
        _G[0, :-1, 0, -1] = _G[0, :-1, 0, -1] + HeatStream(u[0, 1:, 0, 0])
        _G[0, 1:, 0, 0] = _G[0, 1:, 0, 0] + HeatStream(u[0, :-1, 0, -1])
        _G[-1, :-1, -1, -1] = _G[-1, :-1, -1, -1] + HeatStream(u[-1, 1:, -1, 0])
        _G[-1, 1:, -1, 0] = _G[-1, 1:, -1, 0] + HeatStream(u[-1, :-1, -1, -1])
        _G[:-1, 0, -1, 0] = _G[:-1, 0, -1, 0] + HeatStream(u[1:, 0, 0, 0])
        _G[1:, 0, 0, 0] = _G[1:, 0, 0, 0] + HeatStream(u[:-1, 0, -1, 0])
        _G[:-1, -1, -1, -1] = _G[:-1, -1, -1, -1] + HeatStream(u[1:, -1, 0, -1])
        _G[1:, -1, 0, -1] = _G[1:, -1, 0, -1] + HeatStream(u[:-1, -1, -1, -1])

        _G[0, 0, 0, 0] *= 2
        _G[-1, 0, -1, 0] *= 2
        _G[-1, -1, -1, -1] *= 2
        _G[0, -1, 0, -1] *= 2

        res -= self.F + (2 / h) * _G
        
        return res

    def _dot(self, du: np.ndarray) -> np.ndarray:
        res = np.zeros_like(du)
        U = self.U
        h = self.h
        h2 = self.h2
        sigma = self.sigma
        tcc_n = self.tcc_n

        #internal area
        res[::, ::, 1:-1, 1:-1] = tcc_n*(-(du[::, ::, 2:, 1:-1] + du[::, ::, :-2, 1:-1] - 2*du[::, ::, 1:-1, 1:-1])/h2 - \
            (du[::, ::, 1:-1, 2:] + du[::, ::, 1:-1, :-2] - 2*du[::, ::, 1:-1, 1:-1])/h2)
        
        dHeatStream = lambda v, dv: 4*sigma*fabs(v)*fpower(v, 2)*dv
        #all sides
        res[::, ::, 0, 1:-1] = -2*tcc_n*(du[::, ::, 1, 1:-1] - du[::, ::, 0, 1:-1])/h2 + \
                2/h*dHeatStream(U[::, ::, 0, 1:-1], du[::, ::, 0, 1:-1]) - \
                    tcc_n*(du[::, ::, 0, 2:] - 2*du[::, ::, 0, 1:-1] + du[::, ::, 0, :-2])/h2
        res[::, ::, 1:-1, 0] = -2*tcc_n*(du[::, ::, 1:-1, 1] - du[::, ::, 1:-1, 0])/h2 + \
                2/h*dHeatStream(U[::, ::, 1:-1, 0], du[::, ::, 1:-1, 0]) - \
                    tcc_n*(du[::, ::, 2:, 0] - 2*du[::, ::, 1:-1, 0] + du[::, ::, :-2, 0])/h2
        res[::, ::, -1, 1:-1] = 2*tcc_n*(du[::, ::, -1, 1:-1] - du[::, ::, -2, 1:-1])/h2 + \
                2/h*dHeatStream(U[::, ::, -1, 1:-1], du[::, ::, -1, 1:-1]) - \
                    tcc_n*(du[::, ::, -1, 2:] - 2*du[::, ::, -1, 1:-1] + du[::, ::, -1, :-2])/h2
        res[::, ::, 1:-1, -1] = 2*tcc_n*(du[::, ::, 1:-1, -1] - du[::, ::, 1:-1, -2])/h2 + \
                2/h*dHeatStream(U[::, ::, 1:-1, -1], du[::, ::, 1:-1, -1]) - \
                    tcc_n*(du[::, ::, 2:, -1] - 2*du[::, ::, 1:-1, -1] + du[::, ::, :-2, -1])/h2
        
        #inner sides
        res[1:, ::, 0, 1:-1] -= 2/h*dHeatStream(U[:-1, ::, -1, 1:-1], du[:-1, ::, -1, 1:-1])
        res[:-1, ::, -1, 1:-1] -= 2/h*dHeatStream(U[1:, ::, 0, 1:-1], du[1:, ::, 0, 1:-1])
        res[::, 1:, 1:-1, 0] -= 2/h*dHeatStream(U[::, :-1, 1:-1, -1], du[::, :-1, 1:-1, -1])
        res[::, :-1, 1:-1, -1] -= 2/h*dHeatStream(U[::, 1:, 1:-1, 0], du[::, 1:, 1:-1, 0])
        
        #all corners
        res[::, ::, 0, 0] = 4/h*dHeatStream(U[::, ::, 0, 0], du[::, ::, 0, 0]) - \
            2*tcc_n*(du[::, ::, 0, 1] - 2*du[::, ::, 0, 0] + du[::, ::, 1, 0])/h2
        res[::, ::, 0, -1] = 4/h*dHeatStream(U[::, ::, 0, -1], du[::, ::, 0, -1]) - \
            2*tcc_n*(du[::, ::, 0, -2] - 2*du[::, ::, 0, -1] + du[::, ::, 1, -1])/h2
        res[::, ::, -1, -1] = 4/h*dHeatStream(U[::, ::, -1, -1], du[::, ::, -1, -1]) - \
            2*tcc_n*(du[::, ::, -1, -2] - 2*du[::, ::, -1, -1] + du[::, ::, -2, -1])/h2
        res[::, ::, -1, 0] = 4/h*dHeatStream(U[::, ::, -1, 0], du[::, ::, -1, 0]) - \
            2*tcc_n*(du[::, ::, -1, 1] - 2*du[::, ::, -1, 0] + du[::, ::, -2, 0])/h2
        
        #inner corners
        res[:-1, :-1, -1, -1] -= 2/h*(dHeatStream(U[1:, :-1, 0, -1], du[1:, :-1, 0, -1]) + \
            dHeatStream(U[:-1, 1:, -1, 0], du[:-1, 1:, -1, 0]))
        res[1:, :-1, 0, -1] -= 2/h*(dHeatStream(U[:-1, :-1, -1, -1], du[:-1, :-1, -1, -1]) + \
            dHeatStream(U[1:, 1:, 0, 0], du[1:, 1:, 0, 0]))
        res[1:, 1:, 0, 0] -= 2/h*(dHeatStream(U[1:, :-1, 0, -1], du[1:, :-1, 0, -1]) + \
            dHeatStream(U[:-1, 1:, -1, 0], du[:-1, 1:, -1, 0]))
        res[:-1, 1:, -1, 0] -= 2/h*(dHeatStream(U[1:, 1:, 0, 0], du[1:, 1:, 0, 0]) + \
            dHeatStream(U[:-1, :-1, -1, -1], du[:-1, :-1, -1, -1]))
        
        # outer corners
        res[0, :-1, 0, -1] -= 2 / h * dHeatStream(U[0, 1:, 0, 0], du[0, 1:, 0, 0])
        res[0, 1:, 0, 0] -= 2 / h * dHeatStream(U[0, :-1, 0, -1], du[0, :-1, 0, -1])
        res[-1, :-1, -1, -1] -= 2 / h * dHeatStream(U[-1, 1:, -1, 0], du[-1, 1:, -1, 0])
        res[-1, 1:, -1, 0] -= 2 / h * dHeatStream(U[-1, :-1, -1, -1], du[-1, :-1, -1, -1])
        res[:-1, 0, -1, 0] -= 2 / h * dHeatStream(U[1:, 0, 0, 0], du[1:, 0, 0, 0])
        res[1:, 0, 0, 0] -= 2 / h * dHeatStream(U[:-1, 0, -1, 0], du[:-1, 0, -1, 0])
        res[:-1, -1, -1, -1] -= 2 / h * dHeatStream(U[1:, -1, 0, -1], du[1:, -1, 0, -1])
        res[1:, -1, 0, -1] -= 2 / h * dHeatStream(U[:-1, -1, -1, -1], du[:-1, -1, -1, -1])

        return res

    def scal(self, x: np.ndarray, y: np.ndarray) -> np.float64:
        res, h2 = 0.0, self.h2
        res = np.sum(x[::, ::, 1:-1, 1:-1] * y[::, ::, 1:-1, 1:-1]) * h2
        res += (0.5 * h2 * (
                np.sum(x[::, ::, 1:-1, 0] * y[::, ::, 1:-1, 0])
                + np.sum(x[::, ::, 1:-1, -1] * y[::, ::, 1:-1, -1])
                + np.sum(x[::, ::, 0, 1:-1] * y[::, ::, 0, 1:-1])
                + np.sum(x[::, ::, -1, 1:-1] * y[::, ::, -1, 1:-1])
            )
        )
        res += (0.25 * h2 * np.sum((
                    x[::, ::, 0, 0] * y[::, ::, 0, 0]
                    + x[::, ::, -1, 0] * y[::, ::, -1, 0]
                    + x[::, ::, 0, -1] * y[::, ::, 0, -1]
                    + x[::, ::, -1, -1] * y[::, ::, -1, -1]
                )
            )
        )
        return res

    def Norm(self, x: np.ndarray):
        return np.sqrt(self.scal(x, x))

    def log(func):
        @functools.wraps(func)
        def _logwrapper(self, *args, **kwargs):
            if self.logFlag:
                self.log_file = open(f"{self.folder}/logs/{self.test_name}.log", 'a')
            result = func(self, *args, **kwargs)
            if self.logFlag:
                self.log_file.close()
                self.log_file = None
            return result
        return _logwrapper

    @timer
    def BiCGstab(self, eps: np.float64):
        self.dU = np.zeros_like(self.dU)
        self.cg_err.append([])
        b = -self.dot(self.U) #probably with minus
        b_norm = self.Norm(b)
        r = b
        rt = np.copy(r)
        p = np.copy(r)
        beta = 0
        n_iter = 0
        err = self.Norm(r)
        self.cg_err[-1].append(err)
        NMAX = 25000
        _dotp = np.zeros_like(p)
        while err > eps * b_norm and n_iter < NMAX:
            scal_r_rt = self.scal(r, rt)
            if n_iter > 0:
                beta = alpha / w * scal_r_rt / self.scal(r0, rt)
                p = r + beta * (p - w * _dotp)
            _dotp = self._dot(p)
            alpha = scal_r_rt / self.scal(_dotp, rt)
            s = r - alpha * _dotp
            _dots = self._dot(s)
            w = self.scal(_dots, s) / self.scal(_dots, _dots)
            self.dU += alpha * p + w * s
            r0 = np.copy(r)
            r = s - w * _dots
            n_iter += 1
            err = self.Norm(r)
            self.cg_err[-1].append(err)
        _err = self.Norm(self.dU)
        self.newt_err.append(_err)
        self._log(f"[{len(self.newt_err):02}] Newt err - {_err:e} || [{n_iter:05d}] ", end="")
        return _err

    @log
    @timer
    def Compute(self, eps: np.float64, u0: np.ndarray = None) -> np.ndarray:
        if (u0 is not None) and self.U.shape != u0.shape:
            self.U = self.Linearize(u0)
        elif u0:
            self.U = u0
        self.U *= 0.01
        self._log(f"[{0:02}] Compute started")
        _err = self.BiCGstab(1e-4)
        self.U += self.dU
        while _err > eps:
            _err = self.BiCGstab(1e-4)
            if len(self.cg_err[-1]) > 24999:
                break
            self.U += self.dU
        self._log(f"[{len(self.newt_err)+1:02}] Compute over")
        self.U = w * self.U
        return self.U

    def Linearize(self, array: np.ndarray) -> np.ndarray:
        cells, _, cell_size, _ = array.shape
        avg_res = np.zeros((cells, cells))
        pts = np.zeros((cells**2, 2))
        for i in range(cells):
            for j in range(cells):
                avg_res[i, j] = array[i, j].sum() / cell_size**2
                pts[i * cells + j] = np.array(
                    [self.limits[0] * i / (cells - 1), self.limits[1] * j / (cells - 1)]
                )
        interp = LinearNDInterpolator(pts, avg_res.reshape(cells**2))

        res = np.zeros(self.U.size)
        res_ = res.reshape(self.U.shape)
        cells, _, cell_size, _ = self.U.shape
        for i in range(cells):
            for j in range(cells):
                for i2 in range(cell_size):
                    for j2 in range(cell_size):
                        res_[i, j, i2, j2] = interp(
                            (i * (cell_size - 1) + i2) * self.h,
                            (j * (cell_size - 1) + j2) * self.h,
                        )
        return res_

    def _flatten1(self, data: np.ndarray) -> np.ndarray:
        h = self.h
        limits = self.limits
        cells = self.cells
        cell_size = self.cell_size - 1
        x = np.arange(0, limits[0] + h, h)
        y = np.arange(0, limits[1] + h, h)
        res = np.zeros(shape=(x.size, y.size))
        for i in range(0, x.size, cell_size):
            for j in range(0, y.size, cell_size):
                if i == x.size - 1:
                    if j == y.size - 1:
                        res[i, -1] = data[-1, -1, -1, -1]
                    else:
                        res[i, j : j + cell_size] = data[
                            i // cell_size - 1, j // cell_size, -1, :-1
                        ]
                elif j == y.size - 1:
                    res[i : i + cell_size, -1] = data[
                        i // cell_size, j // cell_size - 1, :-1, -1
                    ]
                else:
                    res[i : i + cell_size, j : j + cell_size] = data[
                        i // cell_size, j // cell_size, :-1, :-1
                    ]
        return res

    def _flatten2(self, data: np.ndarray) -> np.ndarray:
        h = self.h
        limits = self.limits
        cells = self.cells
        cell_size = self.cell_size - 1
        x = np.arange(0, limits[0] + h, h)
        y = np.arange(0, limits[1] + h, h)
        res = np.zeros(shape=(x.size, y.size))
        for i in range(x.size):
            for j in range(y.size):
                k = 1
                res[i, j] = data[
                    i // cell_size - (1 if i == x.size - 1 else 0),
                    j // cell_size - (1 if j == y.size - 1 else 0),
                    i % cell_size if i != x.size - 1 else -1,
                    j % cell_size if j != y.size - 1 else -1,
                ]
                if i % cell_size == 0:
                    if i != 0 and i != x.size - 1:
                        res[i, j] += data[
                            i // cell_size - 1,
                            j // cell_size - (1 if j == y.size - 1 else 0),
                            -1,
                            j % cell_size if j != y.size - 1 else -1,
                        ]
                        k += 1
                if j % cell_size == 0:
                    if j != 0 and j != y.size - 1:
                        res[i, j] += data[
                            i // cell_size - (1 if i == x.size - 1 else 0),
                            j // cell_size - 1,
                            i % cell_size if i != x.size - 1 else -1,
                            -1,
                        ]
                        k += 1
                if k == 3:
                    res[i, j] += data[i // cell_size - 1, j // cell_size - 1, -1, -1]
                    k += 1
                res[i, j] /= k
        return res

    def show_res(self, code=0, show_plot=True, heatmap=True):
        # 0 - compute result
        # 1 - error
        # 2 - F
        zlim = self.zlim
        if code == 0:
            data = self.U
        elif code == 1:
            data = fabs(self.dU)
        elif code == 2:
            data = self.F
        else:
            data = np.zeros_like(self.F)
        data = self._flatten1(data)
        if not heatmap:
            draw2D(
                data,
                [0, self.limits[0]],
                self.folder + "/" + self.test_name,
                show_plot=show_plot,
                zlim=zlim,
            )
        else:
            drawHeatmap(
                data,
                [0, self.limits[0]],
                self.folder + "/" + self.test_name + f"_({code})",
                show_plot=show_plot,
                zlim=zlim,
            )

    def trace_newt_err(self, show_plot=1):
        draw1D(
            [np.array(self.newt_err)],
            [1.0, len(self.newt_err)],
            f"{self.folder}/CG/{self.test_name}(Newton error plot)",
            yscale="log",
            show_plot=show_plot,
        )

    def trace_cg_err(self, show_plot=1):
        for i in range(len(self.cg_err)):
            draw1D(
                [np.array(self.cg_err[i])],
                [1.0, len(self.cg_err[i])],
                f"{self.folder}/CG/{self.test_name}(CG iter_{i+1:03d})",
                yscale="log",
                show_plot=show_plot,
            )
    
    def _log(self, line: str, **kwargs) -> None:
        log = f"{strftime('%H:%M:%S')} {line}"
        if self.log_file is None:
            print(log, **kwargs)
        else:
            print(log, file=self.log_file, **kwargs)

    def save(self):
        with open(f"{self.folder}/bin/{self.test_name}.bin", "wb") as file:
            pickle.dump(self, file)
