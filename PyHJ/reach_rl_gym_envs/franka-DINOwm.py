import os
import sys
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch.utils.data import DataLoader

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)

from dino_wm.models.dino_models import normalize_acs, select_xyyaw_from_state


class Franka_DINOWM_Env(gym.Env):
    # TODO: 1. baseline over approximation; 2. our critic loss drop faster
    def __init__(self, params, pass_constraint=True, device="cuda:0"):
        self.device = device
        self.set_wm(*params)
        self.pass_constraint = pass_constraint

        if self.pass_constraint:
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(397,),
                        dtype=np.float32,
                    ),
                    "constraints": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(512 + 1,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(397,),
                dtype=np.float32,
            )
        self.action_space = spaces.Box(low=-0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.front_hist = None
        self.state_hist = None
        if self.pass_constraint:
            self.constraint_type = "database"  # "prox" - proxies, "database"
            self.select_constraint()

    def _reset_loader(self):
        self.data = iter(DataLoader(self.dataset, batch_size=1, shuffle=True))
        self.data_const = iter(
            DataLoader(self.const_dataset, batch_size=1, shuffle=True)
        )

    def set_wm(self, wm, past_data, const_data):
        self.wm = wm.to(self.device)
        self.dataset = past_data
        self.const_dataset = const_data
        self._reset_loader()

    def step(self, action):
        ac_torch = torch.tensor([[action]], dtype=torch.float32).to(
            self.device
        )  # *self.scalar

        self.ac_hist = torch.cat([self.ac_hist[:, 1:], ac_torch], dim=1)
        rew = np.inf
        # Latent: [1 T N Z]
        latent = self.wm.forward_features(
            self.front_hist, self.state_hist, self.ac_hist
        )

        # inp1: [1 T N S], state: [1 T 3]
        inp1, state = (
            self.wm.front_head(latent),
            self.wm.state_pred(latent),
        )
        self.front_hist = torch.cat([self.front_hist[:, 1:], inp1[:, [-1]]], dim=1)
        self.state_hist = torch.cat([self.state_hist[:, 1:], state[:, [-1]]], dim=1)

        if self.pass_constraint:
            rew = self.safety_margin_pa(latent)  # rew is negative if unsafe
        else:
            rew = self.safety_margin_classifier(latent)  # rew is negative if unsafe

        self.latent = latent[:, [-1]].mean(dim=2).detach().cpu().numpy()
        terminated = False
        truncated = False
        info = {"is_first": False, "is_terminal": terminated}
        state = np.copy(self.latent).flatten()

        # Pass constraint info if required
        if self.pass_constraint:  # AnySafe
            obs = {
                "state": state,
                "constraints": self.constraint["semantic_feat"].cpu().numpy(),
            }
            assert state.shape == self.observation_space["state"].shape, (
                "State shape mismatch with observation space, got {}."
            ).format(state.shape)
            assert (
                obs["constraints"].shape == self.observation_space["constraints"].shape
            ), "Constraint shape mismatch with observation space."
        else:  # Latent Safe
            obs = state
            assert state.shape == self.observation_space.shape, (
                "State shape mismatch with observation space, got {}."
            ).format(state.shape)
        return obs, rew, terminated, truncated, info

    def reset(
        self,
        initial_state=None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        try:
            data = next(self.data)
        except StopIteration:
            # Reset the DataLoader and reshuffle
            self._reset_loader()
            data = next(self.data)

        # inputs1: [1 T N P], acs: [1 T 3], states: [1 T 3]
        inputs1 = data["cam_zed_embd"][[0], :].to(self.device)
        acs = data["action"][[0], :].to(self.device)
        acs = normalize_acs(acs, device=self.device)
        states = select_xyyaw_from_state(data["state"][[0], :]).to(self.device)

        # [1 T N Z]
        self.latent = self.wm.forward_features(inputs1, states, acs)
        # inp1: [1 T N S], state: [1 T 3]
        inp1, state = (
            self.wm.front_head(self.latent),
            self.wm.state_pred(self.latent),
        )
        # front_hist: [1 T N P], state_hist: [1 T 3], ac_hist: [1 T 3]
        self.front_hist = torch.cat([inputs1[:, 1:], inp1[:, [-1]]], dim=1)
        self.state_hist = torch.cat([states[:, 1:], state[:, [-1]]], dim=1)
        self.ac_hist = acs

        if self.pass_constraint:
            self.select_constraint()

        state = np.copy(
            self.latent[:, [-1]].mean(dim=2).flatten().detach().cpu().numpy()
        )

        if self.pass_constraint:
            obs = {
                "state": state,
                "constraints": self.constraint["semantic_feat"].cpu().numpy(),
            }
            assert (
                obs["constraints"].shape == self.observation_space["constraints"].shape
            ), "Constraint shape mismatch with observation space."
            assert state.shape == self.observation_space["state"].shape, (
                "State shape mismatch with observation space, got {}."
            ).format(state.shape)
        else:
            obs = state
            assert state.shape == self.observation_space.shape, (
                "State shape mismatch with observation space, got {}."
            ).format(state.shape)

        return obs, {
            "is_first": True,
            "is_terminal": False,
        }

    def safety_margin_classifier(self, latent):
        g_xList = []

        with torch.no_grad():  # Disable gradient calculation
            outputs = torch.tanh(2 * self.wm.fail_pred(latent)[0, -1])
            g_xList.append(outputs.detach().cpu().numpy())

        safety_margin = np.array(g_xList).squeeze()

        return safety_margin

    def safety_margin_pa(self, latent):
        g_xList = []

        with torch.no_grad():
            inp1 = self.wm.front_head(latent)
            state = self.wm.state_pred(latent)
            # [1 T S]
            semantic_features = self.wm.semantic_embed(inp1=inp1, state=state)
            # if self.constraint is None:
            #     const = self.wm.proxies.to(self.device).detach()  # [M Z]
            # else:
            const = (
                self.constraint["semantic_feat"].unsqueeze(0).to(self.device)[..., :-1]
            )  # [1, Z]

            assert const.requires_grad is False, "Proxies should not require gradients."

            queries_norm = F.normalize(
                semantic_features.squeeze(), p=2, dim=1
            )  # [T, Z]

            const_norm = F.normalize(const, p=2, dim=1)  # [1, Z]

            # Compute cosine similarity
            cos_sim_matrix = queries_norm @ const_norm.T  # [T, 1]
            # outputs = torch.tanh(2 * -cos_sim_matrix[-1])
            outputs = -cos_sim_matrix[-1]

            g_xList.append(outputs.detach().cpu().numpy())

        safety_margin = np.array(g_xList).squeeze()
        return safety_margin

    def select_constraint(self):
        if self.constraint_type == "prox":
            num_p = self.wm.proxies.shape[0]
            idx = np.random.randint(0, num_p)
            self.constraint = {
                "semantic_feat": torch.cat(  # Concatenated with 1 means the constraint is active
                    [self.wm.proxies[idx], torch.tensor([1.0]).to(self.device)], dim=0
                ).detach(),
            }
        elif self.constraint_type == "database":
            try:
                data = next(self.data_const)
            except StopIteration:
                # Reset the DataLoader and reshuffle
                self._reset_loader()
                data = next(self.data_const)
            if data["failure"][0, -1, -1] == -1:
                self.select_constraint()

            # cam_zed_embd: [1 1 N P], state: [1 1 3]
            cam_zed_embd = data["cam_zed_embd"][[0], -1:].to(self.device)
            state = select_xyyaw_from_state(data["state"][[0], -1:]).to(self.device)
            # semantic_feat: [Z]
            semantic_feat = self.wm.semantic_embed(
                inp1=cam_zed_embd, state=state
            ).squeeze()
            self.constraint = {
                "semantic_feat": torch.cat(  # Concatenated with 1 means the constraint is active
                    [semantic_feat, torch.tensor([1.0]).to(self.device)], dim=0
                ).detach(),
            }
        else:
            raise NotImplementedError(
                f"Constraint type {self.constraint_type} not implemented."
            )

        assert (
            self.constraint["semantic_feat"].shape
            == self.observation_space["constraints"].shape
        ), "Constraint shape mismatch with observation space."
