# for math operations
# Python reachability toolbox

import hj_reachability as hj

# for accelerated computing
import jax.numpy as jnp

# for plotting
import numpy as np
import torch
from hj_reachability import dynamics, sets

speed = 0.5

# @title 3D Dubins Car Dynamics


class Dubins3D(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(
        self,
        max_turn_rate=1.25,
        control_mode="max",
        disturbance_mode="min",
        control_space=None,
        disturbance_space=None,
    ):
        self.speed = speed
        if control_space is None:
            control_space = sets.Box(
                jnp.array([-max_turn_rate]), jnp.array([max_turn_rate])
            )
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([0, 0]), jnp.array([0, 0]))
        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        _, _, psi = state
        v = self.speed
        return jnp.array([v * jnp.cos(psi), v * jnp.sin(psi), 0.0])

    def control_jacobian(self, state, time):
        x, y, _ = state
        return jnp.array(
            [
                [0],
                [0],
                [1],
            ]
        )

    def disturbance_jacobian(self, state, time):
        return jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )


def dubins_dynamics_tensor(
    current_state: torch.Tensor, action: torch.Tensor, dt: float
) -> torch.Tensor:
    """
    current_state: shape(num_samples, dim_x)
    action: shape(num_samples, dim_u)

    Implemented discrete time dynamics with RK-4.

    return:
    next_state: shape(num_samples, dim_x)
    """

    def one_step_dynamics(state, action):
        """Compute the derivatives [dx/dt, dy/dt, dtheta/dt]."""
        x_dot = speed * torch.cos(state[:, 2])
        y_dot = speed * torch.sin(state[:, 2])
        theta_dot = action[:, 0]
        return torch.stack([y_dot, x_dot, theta_dot], dim=1)

    # k1
    k1 = one_step_dynamics(current_state, action)
    # k2
    mid_state_k2 = current_state + 0.5 * dt * k1
    k2 = one_step_dynamics(mid_state_k2, action)
    # k3
    mid_state_k3 = current_state + 0.5 * dt * k2
    k3 = one_step_dynamics(mid_state_k3, action)
    # k4
    end_state_k4 = current_state + dt * k3
    k4 = one_step_dynamics(end_state_k4, action)
    # Combine k1, k2, k3, k4 to compute the next state
    next_state = current_state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    next_state[..., -1] = next_state[..., -1] % (2 * np.pi)
    return next_state


class DubinsHJSolver:
    def __init__(self, nx=51, ny=51, nt=51):
        # Define the dynamical system
        self.dyn_sys = Dubins3D()

        # Define the computation grid
        self.grid_min = np.array([-1.1, -1.1, 0.0])  # in meters
        self.grid_max = np.array([1.1, 1.1, 2 * np.pi])  # in meters
        self.num_cells = (nx, ny, nt)  # in cells
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(self.grid_min, self.grid_max), self.num_cells, periodic_dims=2
        )

        self.solver_settings = hj.SolverSettings.with_accuracy(
            "very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
        )

    # Run the solver
    def solve(self, constraints, constraints_shape):
        time = 0.0
        target_time = -2.8

        constraints = constraints[:-1]
        assert constraints.shape[-1] == constraints_shape, (
            "Constraints should be masked"
        )

        x_c, y_c, radius = constraints
        self.failure_lx = (
            jnp.linalg.norm(np.array([x_c, y_c]) - self.grid.states[..., :2], axis=-1)
            - radius
        )

        target_values = hj.step(
            self.solver_settings,
            self.dyn_sys,
            self.grid,
            time,
            self.failure_lx,
            target_time,
        )
        return target_values
