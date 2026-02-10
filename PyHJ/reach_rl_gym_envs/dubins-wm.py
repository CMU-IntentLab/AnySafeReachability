import re
from collections import defaultdict
from typing import Optional

import einops
import gymnasium as gym
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from generate_data_traj_cont import get_frame
from gymnasium import spaces
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm

matplotlib.use("Agg")
from skimage import measure

from PyHJ.reach_rl_gym_envs.dubins import Dubins_Env
from PyHJ.reach_rl_gym_envs.utils.dubins_gt_solver import DubinsHJSolver
from PyHJ.reach_rl_gym_envs.utils.env_eval_utils import get_metrics
from PyHJ.utils.eval_utils import evaluate_V, find_a


class Dubins_WM_Env(gym.Env):
    # TODO: 1. baseline over approximation; 2. our critic loss drop faster
    def __init__(self, params):
        if len(params) == 1:
            config = params[0]
        else:
            wm = params[0]
            past_data = params[1]
            config = params[2]
            self.set_wm(wm, past_data, config)

        self.config = config

        self.render_mode = None
        self.time_step = 0.05
        self.high = np.array(
            [
                1.1,
                1.1,
                np.pi,
            ]
        )
        self.low = np.array([-1.1, -1.1, -np.pi])
        self.device = "cuda:0"
        self.num_constraints = 1
        self.pass_semantic_constraint = config.pass_semantic_constraint
        self.pass_semantic_state = config.pass_semantic_state
        self.pass_constraint = config.safety_margin_type == "cos_sim"

        self.state_shape = (
            config.pa["sz_embedding"] if self.pass_semantic_state else 544
        )

        self.constraint_shape = (
            config.pa["sz_embedding"]
            if self.pass_semantic_constraint
            else self.state_shape
        )
        if self.pass_constraint:
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.state_shape,),
                        dtype=np.float32,
                    ),
                    "constraints": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.constraint_shape + 1,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.state_shape,),
                dtype=np.float32,
            )

        image_size = config.size[0]  # 128
        img_obs_space = gym.spaces.Box(
            low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
        )
        obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        bool_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
        obs_dict = {
            "is_first": bool_space,
            "is_last": bool_space,
            "is_terminal": bool_space,
        }
        if "image" in config.encoder["cnn_keys"]:
            obs_dict["image"] = img_obs_space

        if "obs_state" in config.encoder["mlp_keys"]:
            obs_dict["obs_state"] = obs_space

        self.observation_space_full = gym.spaces.Dict(obs_dict)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )  # joint action space
        self.image_size = config.size[0]
        self.turnRate = config.turnRate

        self.solver = DubinsHJSolver(nx=config.nx, ny=config.ny, nt=config.nt)
        if hasattr(self, "wm"):
            self.select_constraints()

        self.safety_margin_type = config.safety_margin_type
        self.nominal_policy_type = "turn_right"

    def set_wm(self, wm, past_data, config):
        self.device = config.device
        self.encoder = wm.encoder.to(self.device)
        self.wm = wm.to(self.device)
        self.data = past_data

        if config.dyn_discrete:
            self.feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            self.feat_size = config.dyn_stoch + config.dyn_deter

    def step(self, action):
        with torch.no_grad():
            init = {k: v[:, -1] for k, v in self.latent.items()}
            ac_torch = (
                torch.tensor([[action]], dtype=torch.float32).to(self.device)
                * self.turnRate
            )
            self.latent = self.wm.dynamics.imagine_with_action(ac_torch, init)
            self.feat = self.wm.dynamics.get_feat(self.latent)
            rew, cont = self.safety_margin(self.feat)  # rew is negative if unsafe

            self.feat = self.feat.detach().cpu().numpy()

            if cont < 0.75:
                terminated = True
            else:
                terminated = False
            truncated = False
            if self.pass_constraint:
                self.obs = {
                    "state": np.copy(self.feat).flatten()
                    if not self.pass_semantic_state
                    else np.copy(
                        self.wm.semantic_encoder(
                            torch.tensor(
                                self.feat, device=self.device, dtype=torch.float32
                            )
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    ).flatten(),
                    "constraints": self.constraint_sem
                    if self.pass_semantic_constraint
                    else self.constraint_feat,  # Semantic embedding of the constraints
                }
            else:
                self.obs = (
                    np.copy(self.feat).flatten()
                    if not self.pass_semantic_state
                    else np.copy(
                        self.wm.semantic_encoder(
                            torch.tensor(
                                self.feat, device=self.device, dtype=torch.float32
                            )
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    ).flatten()
                )
            info = {"is_first": False, "is_terminal": terminated}
        return self.obs, rew, terminated, truncated, info

    def reset(
        self,
        initial_state=None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self.init_traj = next(self.data)
        self.privileged_state = self.init_traj["privileged_state"][:, -1]
        data = self.wm.preprocess(self.init_traj)
        embed = self.encoder(data)
        self.latent, _ = self.wm.dynamics.observe(
            embed, data["action"], data["is_first"]
        )

        for k, v in self.latent.items():
            self.latent[k] = v[:, [-1]]
        self.feat = self.wm.dynamics.get_feat(self.latent).detach().cpu().numpy()

        self.select_constraints()
        if self.pass_constraint:
            self.obs = {
                "state": np.copy(self.feat).flatten()
                if not self.pass_semantic_state
                else np.copy(
                    self.wm.semantic_encoder(
                        torch.tensor(self.feat, device=self.device, dtype=torch.float32)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                ).flatten(),
                "constraints": self.constraint_sem
                if self.pass_semantic_constraint
                else self.constraint_feat,  # Semantic embedding of the constraints
            }
        else:
            self.obs = (
                np.copy(self.feat).flatten()
                if not self.pass_semantic_state
                else np.copy(
                    self.wm.semantic_encoder(
                        torch.tensor(self.feat, device=self.device, dtype=torch.float32)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                ).flatten()
            )
        return self.obs, {
            "is_first": True,
            "is_terminal": False,
        }

    def safety_margin(self, feat):
        g_xList = []

        cont = self.wm.heads["cont"](feat)

        self.safety_margin_type = self.config.safety_margin_type

        if self.safety_margin_type == "learned":
            with torch.no_grad():  # Disable gradient calculation
                outputs = torch.tanh(self.wm.heads["margin"](feat))
                g_xList.append(outputs.detach().cpu().numpy())

            safety_margin = np.array(g_xList).squeeze()
        elif self.safety_margin_type == "cos_sim":
            # [1 1 512]
            feat_sem = (
                self.wm.semantic_encoder(feat.to(torch.float32)).detach().cpu().numpy()
            )

            with torch.no_grad():
                constraints = self.constraint_sem[..., :-1]  # (Z)
                constraints = einops.repeat(
                    constraints, "Z -> B Z", B=feat_sem.shape[0]
                )
                feat_sem = feat_sem.reshape(constraints.shape)  # (B Z)

                numerator = np.sum(feat_sem * constraints, axis=-1)  # (B)
                denominator = np.linalg.norm(feat_sem, axis=-1) * np.linalg.norm(  # (B)
                    constraints, axis=-1
                )
                metric = -numerator / (denominator + 1e-8)  # (B)
                # metric = metric
                # metric = np.tanh(20 * metric)
                # assert metric.ndim == 2, f"Expected dimension 2, got {metric.shape}"
                # assert metric.shape[1] == 1, (
                #     f"Expected second dimension 1, got {metric.shape[1]}"
                # )
                # safety_margin = np.min(metric, axis=-1)  # (B)
                safety_margin = metric  # (B)
                if self.config.safety_margin_hard_threshold:
                    safety_margin[
                        safety_margin > self.config.safety_margin_threshold
                    ] = 1.0
                    safety_margin[
                        safety_margin <= self.config.safety_margin_threshold
                    ] = -1.0

        else:
            raise ValueError(
                "Unknown safety margin type: {}".format(self.safety_margin_type)
            )

        return safety_margin, cont.mean.squeeze().detach().cpu().numpy()

    def select_one_constraint(self, in_distribution=True, env_dist_type=None):
        env_dist_type = (
            self.config.env_dist_type if env_dist_type is None else env_dist_type
        )
        if env_dist_type == "fc":
            in_distribution_set = [
                np.array([-0.5, -0.5, 0.5, 1.0]),
                np.array([0.5, -0.5, 0.5, 1.0]),
                np.array([-0.5, 0.5, 0.5, 1.0]),
                np.array([0.5, 0.5, 0.5, 1.0]),
            ]
            if in_distribution:
                i = np.random.randint(0, len(in_distribution_set))
                gt_constraint = in_distribution_set[i]
                constraint_state = np.array([*gt_constraint[:2], 0.0])
            else:
                constraint_state = np.array([0.0, 0.0, 0.0])
                gt_constraint = np.array([0.0, 0.0, 0.5, 1.0])
                return constraint_state, gt_constraint

        elif bool(re.fullmatch(r"c\d{3}", env_dist_type)):
            # e.g., c001, c002, ..., c999
            match = re.fullmatch(r"c(\d{3})", env_dist_type)
            N = int(match.group(1))
            k = int(np.sqrt(N))
            assert k * k == N, "N must be a perfect square"

            # Generate 1D coordinates (exclusive of edges)
            coords = np.linspace(-1 + 2 / (k + 2), 1 - 2 / (k + 2), k)

            # Create 2D grid
            X, Y = np.meshgrid(coords, coords)

            # Stack into (N, 2) array of (x, y) pairs
            centers = np.stack([X.ravel(), Y.ravel()], axis=-1)

            radius = 0.5
            theta = 0

            if in_distribution:
                i = np.random.randint(0, len(centers))
                center = centers[i]
                # constraint state is a random state in the circle

                constraint_state = np.array(
                    [
                        center[0],
                        center[1],
                        theta,
                    ]
                )
                gt_constraint = np.array(
                    [constraint_state[0], constraint_state[1], radius, 1.0]
                )
            else:  # Out of distribution
                state = np.random.uniform(-1.0, 1.0, size=2)
                gt_constraint = np.array([state[0], state[1], radius, 1.0])

                if self.config.pass_prototype:
                    # NOTE: Using prototype, find closest center and pass this to value function
                    dists = np.linalg.norm(
                        einops.repeat(state, "D -> B D", B=centers.shape[0])
                        - centers[:, :2],
                        axis=1,
                    )
                    closest_center = centers[np.argmin(dists)]
                    constraint_state = np.array(
                        [closest_center[0], closest_center[1], theta]
                    )
                else:
                    constraint_state = np.array([state[0], state[1], theta])

        elif env_dist_type == "4c":  # four circles
            centers = [
                np.array([-0.5, -0.5, 0.50, 1.0]),
                np.array([0.5, -0.5, 0.50, 1.0]),
                np.array([-0.5, 0.5, 0.50, 1.0]),
                np.array([0.5, 0.5, 0.50, 1.0]),
            ]
            if in_distribution:
                i = np.random.randint(0, len(centers))
                center = centers[i][:2]
                # constraint state is a random state in the circle
                # gt_constraint is a circle of radius 0.50
                radius = np.random.uniform(low=0.0, high=0.50)
                theta = np.random.uniform(low=0.0, high=2 * np.pi)
                constraint_state = np.array(
                    [
                        center[0] + radius * np.cos(theta),
                        center[1] + radius * np.sin(theta),
                        0.0,
                    ]
                )
                # gt_constraint = centers[i]
                gt_constraint = np.array(
                    [constraint_state[0], constraint_state[1], 0.5, 1.0]
                )

            else:  # Out of distribution
                radius = 0.5
                centers_array = np.array(centers)
                while True:
                    constraint_state = np.random.uniform(-1, 1, size=2)
                    # Compute distances to each center
                    dists = np.linalg.norm(
                        einops.repeat(
                            constraint_state, "D -> B D", B=centers_array.shape[0]
                        )
                        - centers_array[:, :2],
                        axis=1,
                    )
                    # Accept point if it's outside all four circles
                    if np.all(dists > radius):
                        constraint_state = np.append(constraint_state, 1.0)
                        break
                gt_constraint = np.array(
                    [
                        constraint_state[0],
                        constraint_state[1],
                        radius,
                        1.0,
                    ]
                )

        elif env_dist_type == "fcfe":
            in_distribution_set = [
                np.array([-0.5, -0.5, 0.5, 1.0]),
                np.array([0.5, -0.5, 0.5, 1.0]),
                np.array([-0.5, 0.5, 0.5, 1.0]),
                np.array([0.5, 0.5, 0.5, 1.0]),
                np.array([-0.5, 0.0, 0.5, 1.0]),
                np.array([0.5, 0.0, 0.5, 1.0]),
                np.array([0.0, -0.5, 0.5, 1.0]),
                np.array([0.0, 0.5, 0.5, 1.0]),
            ]
            if in_distribution:
                i = np.random.randint(0, len(in_distribution_set))
                gt_constraint = in_distribution_set[i]
                constraint_state = np.array([*gt_constraint[:2], 0.0])
            else:
                gt_constraint = np.array([0.0, 0.0, 0.5, 1.0])
                constraint_state = np.array([0.0, 0.0, 0.0])
                return constraint_state, gt_constraint

        elif env_dist_type == "rh":
            if in_distribution:
                gt_constraint = np.array(
                    [
                        np.random.uniform(low=0.0, high=1.0),
                        np.random.uniform(low=-1.0, high=1.0),
                        np.random.uniform(low=0.1, high=0.5),
                        1.0,  # This is used to say that this constraint is active
                    ]
                )
                constraint_state = np.array(
                    [
                        *gt_constraint[:2],
                        0.0,
                    ]
                )
            else:
                gt_constraint = np.array(
                    [
                        np.random.uniform(low=-1.0, high=0.0),
                        np.random.uniform(low=-1.0, high=1.0),
                        np.random.uniform(low=0.1, high=0.5),
                        1.0,  # This is used to say that this constraint is active
                    ]
                )
                constraint_state = np.array(
                    [
                        *gt_constraint[:2],
                        0.0,
                    ]
                )

        elif env_dist_type == "br":
            if in_distribution:
                gt_constraint = np.array(
                    [
                        np.random.uniform(low=-1.0, high=1.0),
                        np.random.uniform(low=-1.0, high=1.0),
                        np.random.uniform(
                            low=0.3, high=0.5
                        ),  # Big radii are in distribution
                        1.0,  # This is used to say that this constraint is active
                    ]
                )
                constraint_state = np.array(
                    [
                        *gt_constraint[:2],
                        0.0,  # theta
                    ]
                )
            else:
                gt_constraint = np.array(
                    [
                        np.random.uniform(low=-1.0, high=1.0),
                        np.random.uniform(low=-1.0, high=1.0),
                        np.random.uniform(
                            low=0.1, high=0.3
                        ),  # Small radii are out of distribution
                        1.0,  # This is used to say that this constraint is active
                    ]
                )
                constraint_state = np.array(
                    [
                        *gt_constraint[:2],
                        0.0,  # theta
                    ]
                )
        elif env_dist_type == "v":
            # Eval and test set are the same here
            constraint_state = np.array([0.0, 0.0, 0.0])
            gt_constraint = np.append(
                constraint_state[:2],
                np.array([0.5, 1.0]),
            )
        elif env_dist_type == "v*":
            # Eval and test set are the same here
            constraint_state = np.array([0.5, 0.5, 0.0])
            gt_constraint = np.append(
                constraint_state[:2],
                np.array([0.5, 1.0]),
            )
        elif env_dist_type == "uni":
            # Eval and test set are the same here
            constraint_state = np.array(
                [
                    np.random.uniform(low=-1.0, high=1.0),
                    np.random.uniform(low=-1.0, high=1.0),
                    np.random.uniform(low=0.0, high=2 * np.pi),  # theta
                ]
            )
            gt_constraint = np.append(
                constraint_state[:2],
                np.array([0.5, 1.0]),  # Default radius and active status,
            )

        elif env_dist_type == "uni_small":
            # Eval and test set are the same here
            constraint_state = np.array(
                [
                    np.random.uniform(low=-0.5, high=0.5),
                    np.random.uniform(low=-0.5, high=0.5),
                    np.random.uniform(low=0.0, high=2 * np.pi),  # theta
                ]
            )
            gt_constraint = np.append(
                constraint_state[:2],
                np.array([0.5, 1.0]),  # Default radius and active status,
            )

        elif env_dist_type == "ds":  # Distribution from dataset
            data = next(self.data)
            constraint_state = np.array(data["privileged_state"])[0, -1]
            while (
                constraint_state[0] < -0.5
                or constraint_state[0] > 0.5
                or constraint_state[1] < -0.5
                or constraint_state[1] > 0.5
            ):
                data = next(self.data)
                constraint_state = np.array(data["privileged_state"])[0, -1]
            gt_constraint = np.append(
                constraint_state[:2],
                np.array([0.5, 1.0]),  # Default radius and active status,
            )
            data = self.wm.preprocess(data)
            embed = self.encoder(data)
            latent, _ = self.wm.dynamics.observe(
                embed, data["action"], data["is_first"]
            )

            for k, v in latent.items():
                latent[k] = v[:, [-1]]
            feat = self.wm.dynamics.get_feat(latent).detach().cpu().numpy().squeeze()
            return feat, constraint_state, gt_constraint
        else:
            raise ValueError(
                "Unknown distribution type: {}".format(self.distribution_type)
            )

        assert len(gt_constraint) == 4, (
            "Constraint should have 4 elements, (x, y, radius, u)"
        )
        assert len(constraint_state) == 3, (
            "Constraint state should have 3 elements (x, y, theta)"
        )
        return constraint_state, gt_constraint

    def select_constraints(self, in_distribution=True):
        # constraint_state is different from constraint
        # constraint_state is the state of the agent to produce the constraint image
        # constraint is the grouund truth constraint as (x,y,radius,u)
        if self.config.env_dist_type not in ["prox"]:
            constraints_info = self.select_one_constraint(
                in_distribution=in_distribution
            )
            if len(constraints_info) == 2:
                constraint_state, gt_constraint = constraints_info
                constraint_state = torch.tensor(constraint_state, dtype=torch.float32)
                # constraint_state[..., -1] = 0  # Set theta to 0 for the constraint image
                img = get_frame(states=constraint_state, config=self.config)
                self.constraint_img = img

                feat_c = self.get_latent(
                    wm=self.wm,
                    thetas=constraint_state[-1].reshape(-1),
                    imgs=[img],
                    compute_lz=False,
                )
            elif len(constraints_info) == 3:  # feat_c is passed
                __, constraint_state, gt_constraint = constraints_info
                constraint_state = torch.tensor(constraint_state, dtype=torch.float32)
                img = get_frame(states=constraint_state, config=self.config)
                self.constraint_img = img

                feat_c = self.get_latent(
                    wm=self.wm,
                    thetas=constraint_state[-1].reshape(-1),
                    imgs=[img],
                    compute_lz=False,
                )
            self.constraint_feat = np.array(np.append(feat_c, 1.0))  # .reshape(
            #     self.num_constraints, -1
            # )
            self.constraint_sem = np.append(  # Semantic embedding of the constraints
                self.wm.semantic_encoder(
                    torch.tensor(
                        np.array(feat_c), device=self.device, dtype=torch.float32
                    )
                )
                .detach()
                .cpu()
                .numpy(),
                1.0,
            )  # .reshape(self.num_constraints, -1)

            self.gt_constraint = np.array(gt_constraint)  # .reshape(
            #     self.num_constraints, -1
            # )  # Store the ground truth constraints
        elif self.config.env_dist_type == "prox":
            if in_distribution:
                i = np.random.randint(
                    0, len(self.wm.proxies) - 1
                )  # NOTE: last class is safe
                self.constraint_sem = np.append(
                    self.wm.proxies[i].detach().cpu().numpy(), 1.0
                ).reshape(self.num_constraints, -1)
                centers = [
                    np.array([-0.5, -0.5, 0.5, 1.0]),
                    np.array([0.5, -0.5, 0.5, 1.0]),
                    np.array([-0.5, 0.5, 0.5, 1.0]),
                    np.array([0.5, 0.5, 0.5, 1.0]),
                ]
                self.gt_constraint = centers[i].reshape(self.num_constraints, -1)
                img = get_frame(
                    states=torch.tensor([*self.gt_constraint[0][:2], 0.0]),
                    config=self.config,
                )
                self.constraint_img = img
            else:
                constraint_state, gt_constraint = self.select_one_constraint(
                    in_distribution=in_distribution, env_dist_type="4c"
                )
                constraint_state = torch.tensor(constraint_state, dtype=torch.float32)
                # constraint_state[..., -1] = 0  # Set theta to 0 for the constraint image
                img = get_frame(states=constraint_state, config=self.config)
                self.constraint_img = img
                feat_c = self.get_latent(
                    wm=self.wm,
                    thetas=constraint_state[-1].reshape(-1),
                    imgs=[img],
                    compute_lz=False,
                )
                self.constraint_feat = np.array(np.append(feat_c, 1.0)).reshape(
                    self.num_constraints, -1
                )
                self.constraint_sem = (
                    np.append(  # Semantic embedding of the constraints
                        self.wm.semantic_encoder(
                            torch.tensor(
                                np.array(feat_c),
                                device=self.device,
                                dtype=torch.float32,
                            )
                        )
                        .detach()
                        .cpu()
                        .numpy(),
                        1.0,
                    ).reshape(self.num_constraints, -1)
                )

                self.gt_constraint = np.array(gt_constraint).reshape(
                    self.num_constraints, -1
                )  # Store the ground truth constraints

    def get_latent(self, wm, thetas, imgs, compute_lz=True):
        thetas = np.expand_dims(np.expand_dims(thetas, 1), 1)
        imgs = np.expand_dims(imgs, 1)
        dummy_acs = np.zeros((np.shape(thetas)[0], 1))
        firsts = np.ones((np.shape(thetas)[0], 1))
        lasts = np.zeros((np.shape(thetas)[0], 1))
        cos = np.cos(thetas)
        sin = np.sin(thetas)
        states = np.concatenate([cos, sin], axis=-1)
        chunks = 21
        if np.shape(imgs)[0] > chunks:
            bs = int(np.shape(imgs)[0] / chunks)
        else:
            bs = int(np.shape(imgs)[0] / chunks)
        for i in range(chunks):
            if i == chunks - 1:
                data = {
                    "obs_state": states[i * bs :],
                    "image": imgs[i * bs :],
                    "action": dummy_acs[i * bs :],
                    "is_first": firsts[i * bs :],
                    "is_terminal": lasts[i * bs :],
                }
            else:
                data = {
                    "obs_state": states[i * bs : (i + 1) * bs],
                    "image": imgs[i * bs : (i + 1) * bs],
                    "action": dummy_acs[i * bs : (i + 1) * bs],
                    "is_first": firsts[i * bs : (i + 1) * bs],
                    "is_terminal": lasts[i * bs : (i + 1) * bs],
                }

            data = wm.preprocess(data)
            embeds = wm.encoder(data)
            if i == 0:
                embed = embeds
            else:
                embed = torch.cat([embed, embeds], dim=0)

        data = {
            "obs_state": states,
            "image": imgs,
            "action": dummy_acs,
            "is_first": firsts,
            "is_terminal": lasts,
        }
        data = wm.preprocess(data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])

        feat = wm.dynamics.get_feat(post).detach()
        if compute_lz:
            lz, __ = self.safety_margin(feat)  # lz is the safety margin
            return feat.squeeze().cpu().numpy(), np.array(lz)
        else:
            return feat.squeeze().cpu().numpy()

    def get_eval_plot(self, cache, thetas, policy, config, in_distribution=True):
        nx, ny, nt = config.nx, config.ny, config.nt
        show_constraint = self.safety_margin_type == "cos_sim"
        if show_constraint:
            fig1, axes1 = plt.subplots(
                3, len(thetas) + 1, figsize=(3 * (len(thetas) + 1), 10)
            )
            fig2, axes2 = plt.subplots(
                3, len(thetas) + 1, figsize=(3 * (len(thetas) + 1), 10)
            )
            fig3, axes3 = plt.subplots(
                3, len(thetas) + 1, figsize=(3 * (len(thetas) + 1), 10)
            )
        else:
            fig1, axes1 = plt.subplots(3, len(thetas), figsize=(3 * len(thetas), 10))
            fig2, axes2 = plt.subplots(3, len(thetas), figsize=(3 * len(thetas), 10))
            fig3, axes3 = plt.subplots(3, len(thetas), figsize=(3 * len(thetas), 10))

        # constraint = np.array([0.0, 0.0, 0.5, 1.0]).reshape(1, -1)
        self.select_constraints(in_distribution=in_distribution)
        constraint = self.gt_constraint
        # constraint[:, 2] = 0.5  # Force radius to 0.5
        gt_values = self.solver.solve(
            constraints=constraint,
            constraints_shape=3,
        )

        all_metrics = []

        if show_constraint:
            for axes, fig in zip([axes1, axes2, axes3], [fig1, fig2, fig3]):
                for ax in axes[:, 0]:
                    ax.imshow(
                        self.constraint_img,
                        extent=(-1.5, 1.5, -1.5, 1.5),
                    )
                    ax.set_title("Constraint Reference Image")
            self.constraint_img

        for i in range(len(thetas)):
            theta = thetas[i]
            graph_index = i + 1 if show_constraint else i
            idxs, imgs_prev, thetas_prev = cache[theta]
            with torch.no_grad():
                feat, lz = self.get_latent(
                    wm=self.wm,
                    thetas=thetas_prev,
                    imgs=imgs_prev,
                )
                if self.pass_constraint:
                    obs = {
                        "state": feat
                        if not self.pass_semantic_state
                        else self.wm.semantic_encoder(
                            torch.tensor(feat, device=self.device, dtype=torch.float32)
                        )
                        .detach()
                        .cpu()
                        .numpy(),
                        "constraints": einops.repeat(
                            self.constraint_sem
                            if self.pass_semantic_constraint
                            else self.constraint_feat,
                            "C -> N C",
                            N=feat.shape[0],
                        ),
                    }
                else:
                    obs = (
                        feat
                        if not self.pass_semantic_state
                        else self.wm.semantic_encoder(
                            torch.tensor(feat, device=self.device, dtype=torch.float32)
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                V = evaluate_V(obs=obs, policy=policy, critic=policy.critic)
            V = np.minimum(V, lz)

            nt_index = int(
                np.round((thetas[i] / (2 * np.pi)) * (nt - 1))
            )  # Convert theta to index in the grid

            V = V.reshape((nx, ny)).T  # Reshape to match the grid
            metrics = get_metrics(
                rl_values=V, gt_values=gt_values[:, :, nt_index].T, config=self.config
            )
            # trivial solution
            # metrics = get_metrics(
            #     rl_values=lz.reshape((nx, ny)).T,
            #     gt_values=gt_values[:, :, nt_index].T,
            # )
            # print(metrics)
            # exit()
            all_metrics.append(metrics)

            # Find contours for gt and rl Value functions
            contours_rl = measure.find_contours(
                np.array(V > self.config.safety_margin_threshold).astype(
                    float
                )  # , level=0.0
            )
            contours_gt = measure.find_contours(
                np.array(gt_values[:, :, nt_index].T > 0).astype(float)  # , level=0.0
            )
            # contours_safety_margin = measure.find_contours(
            #     np.array(lz.reshape((nx, ny)).T > self.config.safety_filter_eps).astype(
            #         float
            #     ),
            #     level=0.5,
            # )

            # Show sub-zero level set
            axes1[0, graph_index].imshow(
                V > self.config.safety_margin_threshold,
                extent=(-1.1, 1.1, -1.1, 1.1),
                origin="lower",
            )
            axes1[2, graph_index].imshow(
                gt_values[:, :, nt_index].T > 0,
                extent=(-1.1, 1.1, -1.1, 1.1),
                origin="lower",
            )
            # Show value functions
            axes2[0, graph_index].imshow(
                V,
                extent=(-1.1, 1.1, -1.1, 1.1),
                vmin=-1.0,
                vmax=1.0,
                origin="lower",
            )
            axes2[2, graph_index].imshow(
                gt_values[:, :, nt_index].T,
                extent=(-1.1, 1.1, -1.1, 1.1),
                vmin=-1.0,
                vmax=1.0,
                origin="lower",
            )

            # Plot safety margin
            axes3[0, graph_index].imshow(
                lz.reshape((nx, ny)).T,
                extent=(-1.1, 1.1, -1.1, 1.1),
                vmin=-1.0,
                vmax=1.0,
                origin="lower",
            )
            # GT safety margin
            axes3[2, graph_index].imshow(
                self.solver.failure_lx[:, :, nt_index].T,
                extent=(-1.1, 1.1, -1.1, 1.1),
                vmin=-1.0,
                vmax=1.0,
                origin="lower",
            )

            # Plot contours for RL Value function
            for contour in contours_rl:
                for axes in [
                    axes1,
                    axes2,
                ]:  # Plot in Continuous plot and binary avoid plot
                    extent = 1.1
                    [
                        ax.plot(
                            contour[:, 1] * (2 * extent / (nx - 1)) - extent,
                            contour[:, 0] * (2 * extent / (ny - 1)) - extent,
                            color="blue",
                            linewidth=2,
                            label=f"RL Value Contour (eps={self.config.safety_filter_eps:.2f})",
                        )
                        for ax in axes[:, graph_index]
                    ]

            metric = np.array(lz.reshape((nx, ny)).T)
            x = np.linspace(-1.1, 1.1, metric.shape[1])
            y = np.linspace(-1.1, 1.1, metric.shape[0])
            X, Y = np.meshgrid(x, y)

            contour = axes3[1, graph_index].contour(
                X, Y, metric, levels=5, colors="black", linewidths=1
            )
            axes3[1, graph_index].clabel(contour, inline=True, fontsize=8, fmt="%.2f")

            for axes in [axes1, axes2]:
                contour = axes[1, graph_index].contour(
                    X, Y, V, levels=5, colors="black", linewidths=1
                )
                axes[1, graph_index].clabel(
                    contour, inline=True, fontsize=8, fmt="%.2f"
                )

            # Plot contours for GT Value function
            for contour in contours_gt:
                for axes in [axes1, axes2]:
                    extent = 1.1
                    [
                        ax.plot(
                            contour[:, 1] * (2 * extent / (nx - 1)) - extent,
                            contour[:, 0] * (2 * extent / (ny - 1)) - extent,
                            color="orange",
                            linewidth=2,
                            label="GT Value Contour",
                        )
                        for ax in axes[:, graph_index]
                    ]

            # for contour in contours_safety_margin:
            #     for axes in [axes1, axes2, axes3]:
            #         [
            #             ax.plot(
            #                 contour[:, 1] * (2.0 / (nx - 1)) - 1.0,
            #                 contour[:, 0] * (2.0 / (ny - 1)) - 1.0,
            #                 color="orange",
            #                 linewidth=2,
            #                 label=f"Safety Margin Contour (eps={self.config.safety_margin_threshold:.2f})",
            #             )
            #             for ax in axes[:, graph_index]
            #         ]

            # Add constraint patch
            x_c, y_c, radius, u = self.gt_constraint
            if u == 0.0:
                break
            for axes in [axes1, axes2, axes3]:
                [
                    ax.add_patch(
                        Circle(
                            (x_c, y_c),
                            radius,
                            color="red",
                            fill=False,
                            label="Constraint",
                        )
                    )
                    for ax in axes[:, graph_index]
                ]

            for axes in [axes1, axes2, axes3]:
                for j in range(3):
                    F1 = (
                        2
                        * metrics["TP"]
                        / (2 * metrics["TP"] + metrics["FP"] + metrics["FN"] + 1e-8)
                    )
                    FPR = metrics["FP"] / (metrics["FP"] + metrics["TN"] + 1e-8)
                    label = rf"$\theta$={thetas[i]:.2f}, F1={F1:.2f}, FPR={FPR:.2f}"
                    if j == 1:
                        label = rf"Topo Map, $\theta$={thetas[i]:.2f}, AUC={metrics['AUC']:.2f}"
                    elif j == 2:
                        label = rf"GT, $\theta$={thetas[i]:.2f}"
                    axes[j, graph_index].set_title(
                        label,
                        fontsize=12,
                    )

        for axes in [axes1, axes2, axes3]:
            for ax in axes.flat:
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_aspect("equal")

        for fig, axes in zip([fig1, fig2, fig3], [axes1, axes2, axes3]):
            handles, labels = [], []
            for ax in axes.flat:
                h, label = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(label)

            # Remove duplicates while preserving order
            unique = dict(zip(labels, handles))

            fig.tight_layout(
                pad=0.5, rect=[0, 0, 1, 0.92]
            )  # Adjust spacing between subplots

            # If still overlapping, fine-tune spacing:
            fig.subplots_adjust(hspace=0.05)

            # Create a single, global legend
            fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=3)

        # plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for the legend

        aggregated = defaultdict(list)
        for metrics in all_metrics:
            for key, value in metrics.items():
                aggregated[key].append(value)

        # Compute averages
        aggregate_metrics = {key: np.sum(values) for key, values in aggregated.items()}

        return (
            fig1,
            fig2,
            fig3,
            aggregate_metrics,
        )

    def get_eval_metrics(self, cache, thetas, policy, config, in_distribution=True):
        nx, ny, nt = config.nx, config.ny, config.nt

        # constraint = np.array([0.0, 0.0, 0.5, 1.0]).reshape(1, -1)
        constraint = self.select_constraints(in_distribution=in_distribution)
        gt_constraint = self.gt_constraint

        gt_values = self.solver.solve(
            constraints=gt_constraint,
            constraints_shape=3,
        )

        all_metrics = []

        for i in range(len(thetas)):
            theta = thetas[i]
            idxs, imgs_prev, thetas_prev = cache[theta]
            with torch.no_grad():
                feat, lz = self.get_latent(
                    wm=self.wm,
                    thetas=thetas_prev,
                    imgs=imgs_prev,
                )
                if self.pass_constraint:
                    obs = {
                        "state": feat
                        if not self.pass_semantic_state
                        else self.wm.semantic_encoder(
                            torch.tensor(feat, device=self.device, dtype=torch.float32)
                        )
                        .detach()
                        .cpu()
                        .numpy(),
                        "constraints": einops.repeat(
                            self.constraint_sem
                            if self.pass_semantic_constraint
                            else self.constraint_feat,
                            "C -> N C",
                            N=feat.shape[0],
                        ),
                    }
                else:
                    obs = (
                        feat
                        if not self.pass_semantic_state
                        else self.wm.semantic_encoder(
                            torch.tensor(feat, device=self.device, dtype=torch.float32)
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                V = evaluate_V(obs=obs, policy=policy, critic=policy.critic)
            V = np.minimum(V, lz)

            nt_index = int(
                np.round((thetas[i] / (2 * np.pi)) * (nt - 1))
            )  # Convert theta to index in the grid

            V = V.reshape((nx, ny)).T  # Reshape to match the grid
            metrics = get_metrics(
                rl_values=V, gt_values=gt_values[:, :, nt_index].T, config=self.config
            )
            # trivial solution
            # metrics = get_metrics(
            #     rl_values=lz.reshape((nx, ny)).T,
            #     gt_values=gt_values[:, :, nt_index].T,
            # )
            # print(metrics)
            # exit()
            all_metrics.append(metrics)

        aggregated = defaultdict(list)
        for metrics in all_metrics:
            for key, value in metrics.items():
                aggregated[key].append(value)

        # Compute averages
        averaged_metrics = {key: np.sum(values) for key, values in aggregated.items()}

        return averaged_metrics

    def nominal_policy(self):
        if self.nominal_policy_type == "turn_right":
            return np.array([-1.0], dtype=np.float32)
        else:
            raise ValueError(f"Unknown nominal policy type: {self.nominal_policy_type}")

    def get_trajectory(self, policy):
        gt_env = Dubins_Env(
            nominal_policy=self.nominal_policy_type, dist_type=self.config.env_dist_type
        )
        obs, __ = self.reset()
        priv_state = self.privileged_state.squeeze()
        while (
            priv_state[0] < -1.0
            or priv_state[0] > 1.0
            or priv_state[1] < -1.0
            or priv_state[1] > 1.0
            # Check if already in constraint
            or np.linalg.norm(priv_state[:2] - self.gt_constraint[:2])
            < self.gt_constraint[2]
            or evaluate_V(obs=obs, policy=policy, critic=policy.critic)
            < 0.1 + self.config.safety_margin_threshold
        ):
            obs, __ = self.reset()
            priv_state = self.privileged_state.squeeze()

        # import ipdb

        # ipdb.set_trace()
        # data = self.init_traj
        # import ipdb

        # ipdb.set_trace()

        gt_state = torch.tensor(
            [priv_state[0], priv_state[1], np.sin(priv_state[2]), np.cos(priv_state[2])]
        )
        obs_gt, _ = gt_env.reset(initial_state=gt_state.cpu().numpy())
        gt_env.constraint = self.gt_constraint.copy()
        done_gt = False
        imgs_traj = []
        imgs_imagined = []
        t = 0

        while not done_gt:  # Rollout trajectory with safety filtering
            theta = np.arctan2(obs_gt["state"][2], obs_gt["state"][3])
            state = torch.tensor([obs_gt["state"][0], obs_gt["state"][1], theta])

            with torch.no_grad():
                if isinstance(obs, dict):
                    frame = self.wm.heads["decoder"](
                        torch.tensor(self.feat.flatten(), device=self.device)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )["image"].mode()[0, 0]
                else:
                    frame = self.wm.heads["decoder"](
                        torch.tensor(self.feat.flatten(), device=self.device)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )["image"].mode()[0, 0]
                imgs_imagined.append(frame.cpu().numpy())
                # V, _ = self.safety_margin(
                #     torch.tensor(obs["state"], device=self.device).unsqueeze(0)
                # )
                V = evaluate_V(obs=obs, policy=policy, critic=policy.critic)
                V = V.squeeze()
            if V < self.config.safety_filter_eps:
                unsafe = True
                action = find_a(obs=obs, policy=policy)
            else:
                unsafe = False
                action = self.nominal_policy()

            obs, rew, done, _, info = self.step(action)
            obs_gt, rew_gt, done_gt, _, _ = gt_env.step(action)

            # Closed loop
            title_kwargs = {
                "Nominal Policy": self.nominal_policy_type,
                "Eps": self.config.safety_filter_eps,
                # "Time": t,
                "V": f"{V:.2f}",
                "R": f"{rew_gt:.2f}",
                "A": f"{action.squeeze().item():.2f}",
                "C": f"{done}",
            }
            img = gt_env.render(unsafe=unsafe, title_kwargs=title_kwargs)

            if t > 64:
                done_gt = True
            t += 1

            imgs_traj.append(img)

        imgs = np.array(imgs_traj)

        imgs_imagined = np.array(imgs_imagined) * 255.0

        imgs_imagined = (
            F.interpolate(
                einops.rearrange(torch.tensor(imgs_imagined), "T H W C -> T C H W"),
                size=(imgs.shape[1], imgs.shape[2]),
                mode="bilinear",
                align_corners=False,
            )
            .cpu()
            .numpy()
        )
        imgs = einops.rearrange(imgs, "T H W C -> T C H W")
        imgs = np.concatenate([imgs, imgs_imagined], axis=3)
        gt_env.close()
        return imgs

    def get_success_rate(self, policy):
        gt_env = Dubins_Env(
            nominal_policy=self.nominal_policy_type, dist_type=self.config.env_dist_type
        )
        unsuccessful = 0
        timeout = 0
        out_of_bounds = 0

        total = 250
        fig, ax = plt.subplots(figsize=(5, 5))
        avg_t = 0
        for idx in tqdm(range(total)):
            gt_env.reset()
            trajectory = []
            obs, __ = self.reset()
            priv_state = self.privileged_state.squeeze()

            while (
                # Check if in bounding box
                priv_state[0] < -1.0
                or priv_state[0] > 1.0
                or priv_state[1] < -1.0
                or priv_state[1] > 1.0
                # Check if already in constraint
                or np.linalg.norm(priv_state[:2] - self.gt_constraint[:2])
                < self.gt_constraint[2]
                or evaluate_V(obs=obs, policy=policy, critic=policy.critic)
                < 0.1 + self.config.safety_margin_threshold
            ):
                obs, __ = self.reset()
                priv_state = self.privileged_state.squeeze()

            gt_state = torch.tensor(
                [
                    priv_state[0],
                    priv_state[1],
                    np.sin(priv_state[2]),
                    np.cos(priv_state[2]),
                ]
            )
            obs_gt, _ = gt_env.reset(initial_state=gt_state.cpu().numpy())
            # TODO: Set constraints of gt_env to the same as self.constraint
            gt_env.constraint = self.gt_constraint.copy()
            done_gt = False

            t = 0

            while not done_gt:  # Rollout trajectory with safety filtering
                trajectory.append(gt_env.state[:2])
                with torch.no_grad():
                    V = evaluate_V(obs=obs, policy=policy, critic=policy.critic)
                    V = V.squeeze()
                if V < self.config.safety_filter_eps:
                    action = find_a(obs=obs, policy=policy)
                else:
                    action = self.nominal_policy()

                # action = find_a(obs=obs, policy=policy)

                obs, rew, done, _, info = self.step(action)
                obs_gt, rew_gt, done_gt, _, _ = gt_env.step(action)
                if done_gt:
                    out_of_bounds += 1

                if rew_gt < 0:
                    unsuccessful += 1
                    done_gt = True

                if t > 64:
                    done_gt = True
                    timeout += 1
                t += 1

            avg_t += t

        gt_env.close()
        print("Average time: ", avg_t / total)
        print(
            f"Out of Bounds: {out_of_bounds / total}, Timeout: {timeout / total}, Unsuccessful: {unsuccessful / total}"
        )
        return 1 - unsuccessful / total
