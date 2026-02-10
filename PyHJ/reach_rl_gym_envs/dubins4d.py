from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import spaces
from matplotlib.patches import Circle

from PyHJ.utils import evaluate_V


class Dubins_Env_4D(gym.Env):
    # TODO: 1. baseline over approximation; 2. our critic loss drop faster
    def __init__(self):
        self.render_mode = None
        self.dt = 0.05
        self.u_max = 1.25
        self.a_max = 1
        self.v_max = 3
        self.high = np.array([1.1, 1.1, 1.1, 1.1, 0])
        self.low = np.array([-1.1, -1.1, -1.1, -1.1, self.v_max])
        self.observation_space = spaces.Box(
            low=self.low, high=self.high, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )  # joint action space

        self.constraint = [0.0, 0.0, 0.5]  # constraint: [x, y, r]

    def step(self, action):
        # action is in -1, 1. Scale by self.u_max

        self.state[0] = self.state[0] + self.dt * self.state[3] * self.state[4]
        self.state[1] = self.state[1] + self.dt * self.state[2] * self.state[4]

        theta = np.arctan2(self.state[2], self.state[3])  # sin, cos
        theta_next = theta + self.dt * (action[0] * self.u_max)
        # this trick ensure the periodic state is continuous. this is important so the inputs the nn have no discontinuities
        self.state[2] = np.sin(theta_next)
        self.state[3] = np.cos(theta_next)

        self.state[4] = self.state[4] + self.dt * (action[1] * self.a_max)
        self.state[4] = np.clip(self.state[4], 0, self.v_max)

        # l(x) = (x-x0)^2 + (y-y0)^2 - r^2
        rew = (
            (self.state[0] - self.constraint[0]) ** 2
            + (self.state[1] - self.constraint[1]) ** 2
            - self.constraint[2] ** 2
        )

        terminated = False
        truncated = False
        if any(self.state[:2] > self.high[:2]) or any(self.state[:2] < self.low[:2]):
            terminated = True

        info = {}
        return self.state.astype(np.float32), rew, terminated, truncated, info

    def reset(
        self,
        initial_state=None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        if initial_state is None:
            theta = np.random.uniform(low=0, high=2 * np.pi)

            self.state = np.random.uniform(
                low=[-1.1, -1.1, -1.1, -1.1, 0.0],
                high=[1.1, 1.1, 1.1, 1.1, self.v_max],
                size=(5,),
            )
            self.state[2] = np.sin(theta)
            self.state[3] = np.cos(theta)
        else:
            self.state = initial_state
        self.state = self.state.astype(np.float32)
        return self.state, {}

    def get_eval_plot(self, policy, critic):
        nx, ny = 51, 51
        thetas = [0, np.pi / 6, np.pi / 3, np.pi / 2]
        vs = [1.0, 1.5, 3.0]
        fig1, axes1 = plt.subplots(len(vs), len(thetas))
        fig2, axes2 = plt.subplots(len(vs), len(thetas))
        X, Y = np.meshgrid(
            np.linspace(-1.1, 1.1, nx, endpoint=True),
            np.linspace(-1.1, 1.1, ny, endpoint=True),
        )
        for i in range(len(thetas)):
            for j in range(len(vs)):
                V = np.zeros_like(X)
                for ii in range(nx):
                    for jj in range(ny):
                        tmp_point = torch.tensor(
                            [
                                X[ii, jj],
                                Y[ii, jj],
                                np.sin(thetas[i]),
                                np.cos(thetas[i]),
                                vs[j],
                            ]
                        )
                        V[ii, jj] = evaluate_V(tmp_point, policy, critic)

                axes1[j, i].imshow(V > 0, extent=(-1.1, 1.1, -1.1, 1.1), origin="lower")
                axes2[j, i].imshow(
                    V,
                    extent=(-1.1, 1.1, -1.1, 1.1),
                    vmin=-1.0,
                    vmax=1.0,
                    origin="lower",
                )
                const = Circle(
                    (-0.0, -0.0), 0.5, color="red", fill=False
                )  # fill=False makes it a ring
                # Add the circle to the axes
                axes1[j, i].add_patch(const)
                const2 = Circle(
                    (-0.0, -0.0), 0.5, color="red", fill=False
                )  # fill=False makes it a ring
                # Add the circle to the axes
                axes2[j, i].add_patch(const2)
                axes1[0, i].set_title(
                    "theta = {}".format(np.round(thetas[i], 2)),
                    fontsize=12,
                )
                axes2[0, i].set_title(
                    "theta = {}".format(np.round(thetas[i], 2)),
                    fontsize=12,
                )
                axes1[j, 0].set_ylabel(
                    "v = {}".format(np.round(vs[j], 2)),
                    fontsize=12,
                )
                axes2[j, 0].set_ylabel(
                    "v = {}".format(np.round(vs[j], 2)),
                    fontsize=12,
                )

        return fig1, fig2

    def get_eval_plot_f1(self, policy, critic):
        grid = np.load("/home/kensuke/HJRL/new_BRT_v1_w1.25.npy")

        plot_idxs = [0, 7, 14, 20]

        fig1, axes1 = plt.subplots(1, len(plot_idxs))
        fig2, axes2 = plt.subplots(1, len(plot_idxs))
        X, Y = np.meshgrid(
            np.linspace(-1.1, 1.1, grid.shape[0], endpoint=True),
            np.linspace(-1.1, 1.1, grid.shape[1], endpoint=True),
        )
        thetas_lin = np.linspace(0, 2 * np.pi, grid.shape[2], endpoint=True)

        tp, fp, fn, tn = 0, 0, 0, 0
        for i in range(len(thetas_lin)):
            V = np.zeros_like(X)
            for ii in range(grid.shape[0]):
                for jj in range(grid.shape[1]):
                    tmp_point = torch.tensor(
                        [
                            X[ii, jj],
                            Y[ii, jj],
                            np.sin(thetas_lin[i]),
                            np.cos(thetas_lin[i]),
                        ]
                    )
                    V[ii, jj] = evaluate_V(tmp_point, policy, critic)

            V_grid = grid[:, :, i]
            tp_grid = np.sum((V > 0) & (V_grid.T > 0))
            fp_grid = np.sum((V > 0) & (V_grid.T < 0))
            fn_grid = np.sum((V < 0) & (V_grid.T > 0))
            tn_grid = np.sum((V < 0) & (V_grid.T < 0))
            tp += tp_grid
            fp += fp_grid
            fn += fn_grid
            tn += tn_grid

            prec_grid = tp_grid / (tp_grid + fp_grid) if (tp_grid + fp_grid) > 0 else 0
            rec_grid = tp_grid / (tp_grid + fn_grid) if (tp_grid + fn_grid) > 0 else 0
            f1_grid = (
                2 * (prec_grid * rec_grid) / (prec_grid + rec_grid)
                if (prec_grid + rec_grid) > 0
                else 0
            )

            if i in plot_idxs:
                plot_idx = plot_idxs.index(i)
                axes1[plot_idx].imshow(
                    V > 0, extent=(-1.1, 1.1, -1.1, 1.1), origin="lower"
                )
                axes2[plot_idx].imshow(
                    V,
                    extent=(-1.1, 1.1, -1.1, 1.1),
                    vmin=-1.0,
                    vmax=1.0,
                    origin="lower",
                )
                axes1[plot_idx].set_title(
                    "theta = {}".format(np.round(thetas_lin[i], 2)),
                    fontsize=12,
                )
                axes2[plot_idx].set_title(
                    "f1 = {}".format(np.round(f1_grid, 2)),
                    fontsize=12,
                )

                axes1[plot_idx].contour(
                    grid[:, :, i].T,
                    levels=[0.0],
                    colors="purple",
                    linewidths=2,
                    origin="lower",
                    extent=[-1.1, 1.1, -1.1, 1.1],
                )

        print("TP: {}, FP: {}, FN: {}, TN: {}".format(tp, fp, fn, tn))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        return fig1, fig2, f1
