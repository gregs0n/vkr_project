import numpy as np
import pickle
from numpy import float_power as fpower, fabs
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse.linalg import *
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
        folder=""
    ):
        self.F, self.G = F, G

        self.square_shape = (cells[0], cells[1], cell_size, cell_size)
        self.linear_shape = cells[0]*cells[1]*cell_size*cell_size,

        self.material = material
        self.tcc_n = material.thermal_cond * w
        self.zlim = [300, 600]  # [material.tmin, material.tmax]

        self.U = 0.5 * (material.tmin + material.tmax) * np.ones(self.linear_shape)
        self.dU = np.zeros_like(self.U)

        self.limits = limits
        self.cells = cells
        self.cell_size = cell_size
        self.sigma = stef_bolc
        self.h = limits[0] / ((cell_size - 1) * cells[0])
        self.h2 = fpower(self.h, 2)
        self.newt_err = []

        self.test_name = name
        self.folder = folder

    def operator(self, u_linear: np.ndarray) -> np.ndarray:
        u = u_linear.reshape(self.square_shape)
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
        
        return res.reshape(self.linear_shape)

    def jacobian(self, du_linear: np.ndarray) -> np.ndarray:
        du = du_linear.reshape(self.square_shape)
        res = np.zeros_like(du)
        U = self.U.reshape(self.square_shape)
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

        return res.reshape(self.linear_shape)

    @timer
    def Compute(self, eps: np.float64, u0: np.ndarray = None) -> np.ndarray:
        if u0:
            self.U = u0.reshape(self.linear_shape)
        self.U *= 0.01
        A = LinearOperator((self.linear_shape[0], self.linear_shape[0]), matvec=self.jacobian)
        R = -self.operator(self.U)
        dU, exit_code = bicgstab(
            A,
            R,
            rtol=1.0e-4,
            atol=1.0e-6,
            x0=R,
        )
        if exit_code:
            print(f"jacobian failed with exit code: {exit_code}")
            exit()
        err = np.abs(dU).max()
        self.newt_err.append(err)
        while err > eps:
            self.U += dU
            R = -self.operator(self.U)
            dU, exit_code = tfqmr(
                A,
                R,
                rtol=1.0e-4,
                atol=1.0e-6,
                x0=dU,
            )
            if exit_code:
                print(f"jacobian failed with exit code: {exit_code}")
                exit()
            err = np.abs(dU).max()
            self.newt_err.append(err)
            #print(err)
        self.U = (w * self.U).reshape(self.square_shape)
        return self.U

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
    
    def _flatten3(self, data: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=self.cells)
        for i_cell in range(self.cells[0]):
            for j_cell in range(self.cells[1]):
                cur_cell = data[i_cell, j_cell]
                res[i_cell, j_cell] += np.sum(cur_cell[1:-1, 1:-1])*self.h2
                res[i_cell, j_cell] += self.h2*0.5*(
                    np.sum(cur_cell[1:-1, 0]) +
                    np.sum(cur_cell[1:-1, -1]) + 
                    np.sum(cur_cell[0, 1:-1]) + 
                    np.sum(cur_cell[-1, 1:-1])
                )
                res[i_cell, j_cell] += self.h2 * 0.25 * (
                    cur_cell[0, 0] + cur_cell[0, -1] + 
                    cur_cell[-1, 0] + cur_cell[-1, -1]
                )
                res[i_cell, j_cell] *= (self.cells[0]/self.limits[0])**2
        return res

    def show_res(self, code=0, show_plot=True, heatmap=True):
        # 0 - compute result
        # 1 - error
        # 2 - F
        zlim = self.zlim
        #if code == 0:
        #    data = self.U
        #elif code == 1:
        #    data = fabs(self.dU)
        #elif code == 2:
        #    data = self.F
        #else:
        #    data = np.zeros_like(self.F)
        if code == 3:
            data = self._flatten3(self.U)
        elif code == 1:
            data = self._flatten1(self.U)
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

    def save(self):
        np.save(f"{self.folder}/bin/{self.test_name}", self._flatten3(self.U))
        #with open(f"{self.folder}/bin/{self.test_name}.bin", "wb") as file:
        #    pickle.dump(self, file)
