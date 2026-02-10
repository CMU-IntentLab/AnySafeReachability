import io
from typing import Optional

import gymnasium as gym
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import spaces
from PIL import Image


class Dubins_Image(gym.Env):
    def __init__(self, config):
        self.render_mode = None
        self.time_step = 0.05
        self.high = np.array(
            [
                1.5,
                1.5,
                np.pi,
            ]
        )
        self.low = np.array([-1.5, -1.5, -np.pi])
        self.device = "cuda:0"

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )  # joint action space
        self.turnRate = config.turnRate
        self.v = config.speed
        self.dt = config.dt
        self.config = config

    def reset(
        self,
        initial_state=None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = np.random.uniform(low=self.low, high=self.high)

        self.state = torch.tensor(self.state, dtype=torch.float32)
        image = self.get_frame(self.state, config=self.config)

        return (self.state, image)

    def step(self, action):
        action = action.detach() * self.turnRate
        actual_ac = action
        # ac_prob =  random.random()
        # if ac_prob < 0.2 :
        # 	actual_ac = -action
        # else:
        # 	actual_ac = action

        states_next = torch.rand(3)
        states_next[0] = self.state[0] + self.v * self.dt * torch.cos(self.state[2])
        states_next[1] = self.state[1] + self.v * self.dt * torch.sin(self.state[2])
        states_next[2] = self.state[2] + self.dt * actual_ac

        if torch.abs(states_next[0]) > 1.5 or torch.abs(states_next[1]) > 1.5:
            done = True
        else:
            done = False

        self.state = states_next
        image = self.get_frame(self.state, config=self.config)

        truncated = False
        info = {}

        return (self.state, image), 0, done, truncated, info

    def get_frame(self, states, config):
        dt = config.dt
        v = config.speed
        fig, ax = plt.subplots()
        plt.xlim([config.x_min, config.x_max])
        plt.ylim([config.y_min, config.y_max])
        plt.axis("off")
        fig.set_size_inches(1, 1)
        circle = patches.Circle(
            [config.obs_x, config.obs_y],
            config.obs_r,
            edgecolor="#b3b3b3",
            facecolor="#b3b3b3",
            linewidth=2,
        )
        # Add the circle patch to the axis
        ax.add_patch(circle)
        plt.quiver(
            states[0],
            states[1],
            dt * v * torch.cos(states[2]),
            dt * v * torch.sin(states[2]),
            angles="xy",
            scale_units="xy",
            minlength=0,
            width=0.1,
            scale=0.15,
            color="black",
            zorder=3,
        )

        plt.scatter(states[0], states[1], s=20, color="black", zorder=3)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=config.size[0])
        buf.seek(0)

        # Load the buffer content as an RGB image
        img = Image.open(buf).convert("RGB")
        img_array = np.array(img)
        plt.close(fig)

        return img_array


class Dubins_WM_Env(gym.Env):
    def __init__(self, config):
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

        if config.dyn_discrete:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    1,
                    1,
                    1536,
                ),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    1,
                    1,
                    544,
                ),
                dtype=np.float32,
            )

        image_size = 128
        img_obs_space = spaces.Box(
            low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
        )
        bool_space = spaces.Box(low=0.0, high=1.0, shape=(1,))
        self.observation_space_full = spaces.Dict(
            {
                "image": img_obs_space,
                "is_first": bool_space,
                "is_last": bool_space,
                "is_terminal": bool_space,
            }
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )  # joint action space
        self.image_size = 128
        self.turnRate = config.turnRate
        self.failure_threshold = 0.4  # 0.5

    def set_wm(self, config, wm, dataset):
        self.config = config
        self.wm = wm
        self.dataset = dataset

        if self.config.dyn_discrete:
            self.feat_size = (
                self.config.dyn_stoch * self.config.dyn_discrete + self.config.dyn_deter
            )
        else:
            self.feat_size = self.config.dyn_stoch + self.config.dyn_deter

    def step(self, action):
        with torch.no_grad():
            action = action.detach() * self.turnRate  # action unnormalization.

            failure, cont = self.safety_margin(self.feat)
            reward = 1.5 * torch.tanh(self.failure_threshold - failure).item()

            # TODO: Batch
            if cont.item() < 0.75:
                done = True
            else:
                done = False

            # update feat
            # action = action.reshape((self.config.batch_size, self.config.batch_length, -1))
            self.latent = self.wm.dynamics.img_step(self.latent, action.unsqueeze(-1))
            self.feat = self.wm.dynamics.get_feat(self.latent)

            truncated = False
            info = {}

        return self.feat.cpu().numpy(), reward, done, truncated, info

    def step_offline(self):
        init_traj = next(self.dataset)

        with torch.no_grad():
            data = self.wm.preprocess(init_traj)
            embed = self.wm.encoder(data)
            latent, _ = self.wm.dynamics.observe(
                embed, data["action"], data["is_first"]
            )

            state, state_next = {}, {}
            for k, v in latent.items():
                state[k] = v[:, [0]]
                state_next[k] = v[:, [-1]]

            feat = self.wm.dynamics.get_feat(state)
            feat_next = self.wm.dynamics.get_feat(state_next)

            action = data["action"][:, 0] / self.turnRate  # action normalization.
            assert abs(action.item()) <= 1.0, f"Action out of bounds: {action.item()}"
            # check any of the actions are out of bounds
            # assert np.all(np.abs(action.cpu().numpy()) <= 1.0), "Action out of bounds"

            # reward = 2 * (0.5 - data["margin"][:, 0])

            failure, cont = self.safety_margin(feat)
            reward = 1.5 * torch.tanh(self.failure_threshold - failure).item()

            if cont.item() < 0.75:
                done = True
                truncated = False
            else:
                done = False
                truncated = bool(data["is_first"][:, -1].item())

            info = {}

        return (
            feat.cpu().numpy(),
            action,
            feat_next.cpu().numpy(),
            reward,
            done,
            truncated,
            info,
        )

    def step_offline_batch(self):
        init_traj = next(self.dataset)

        with torch.no_grad():
            data = self.wm.preprocess(init_traj)
            embed = self.wm.encoder(data)
            latent, _ = self.wm.dynamics.observe(
                embed, data["action"], data["is_first"]
            )

            state, state_next = {}, {}
            for k, v in latent.items():
                state[k] = v[:, 0:-1]
                state_next[k] = v[:, 1:]

            feat = self.wm.dynamics.get_feat(state)
            feat_next = self.wm.dynamics.get_feat(state_next)

            action = data["action"][:, 0:-1] / self.turnRate  # action normalization.
            # check any of the actions are out of bounds
            assert np.all(np.abs(action.cpu().numpy()) <= 1.0), "Action out of bounds"

            failure, cont = self.safety_margin(feat)
            reward = 1.5 * torch.tanh(self.failure_threshold - failure)

            truncated = data["is_first"][:, 1:]
            done = (cont < 0.75).squeeze()
            truncated[done] = False

            info = {}

        return (
            feat.cpu().numpy(),
            action,
            feat_next.cpu().numpy(),
            reward,
            done,
            truncated,
            info,
        )

    def reset(
        self,
        initial_state=None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        init_traj = next(self.dataset)

        with torch.no_grad():
            data = self.wm.preprocess(init_traj)
            embed = self.wm.encoder(data)
            self.latent, _ = self.wm.dynamics.observe(
                embed, data["action"], data["is_first"]
            )

            for k, v in self.latent.items():
                self.latent[k] = v[:, [-1]]
            self.feat = self.wm.dynamics.get_feat(self.latent)

        return self.feat.cpu().numpy()

    def reset_batch(self):
        super().reset()
        init_traj = next(self.dataset)

        with torch.no_grad():
            data = self.wm.preprocess(init_traj)
            embed = self.wm.encoder(data)
            self.latent, _ = self.wm.dynamics.observe(
                embed, data["action"], data["is_first"]
            )

            self.feat = self.wm.dynamics.get_feat(self.latent)
            self.feat = self.feat.reshape(
                (self.config.batch_size * self.config.batch_length, -1)
            )

        return self.feat.cpu().numpy()

    def step_batch(self, action):
        with torch.no_grad():
            action = action.detach() * self.turnRate  # action unnormalization.

            failure, cont = self.safety_margin(self.feat)
            reward = 1.5 * torch.tanh(self.failure_threshold - failure)

            done = cont < 0.75

            # update feat
            action = action.reshape(
                (self.config.batch_size, self.config.batch_length, -1)
            )
            self.latent = self.wm.dynamics.img_step(self.latent, action)

            truncated = False
            info = {}

            # TODO if done: reset the environment for that individual indices
            if done.any():
                # Reset the latent state for done indices
                init_traj = next(self.dataset)
                data = self.wm.preprocess(init_traj)
                embed = self.wm.encoder(data)
                temp_latent, _ = self.wm.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )

                done_idxs = done[:, 0].nonzero(as_tuple=False).view(-1).tolist()
                for i in done_idxs:
                    # alter the latent state for the done indices
                    batch_idx = i // self.config.batch_length
                    temp_idx = i % self.config.batch_length
                    for k, v in temp_latent.items():
                        self.latent[k][batch_idx, temp_idx] = v[batch_idx][
                            batch_idx, temp_idx
                        ]

            self.feat = self.wm.dynamics.get_feat(self.latent)
            self.feat = self.feat.reshape(
                (self.config.batch_size * self.config.batch_length, -1)
            )

        return self.feat.cpu().numpy(), reward, done, truncated, info

    def safety_margin(self, feats):
        with torch.no_grad():  # Disable gradient calculation
            failure_dist = self.wm.heads["margin"](feats)
            failure = failure_dist.mean  # \in [0, 1] (Bernouli Dist)
            cont = self.wm.heads["cont"](feats).mean

            return failure, cont
