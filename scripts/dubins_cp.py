import argparse
import collections
import os
import random
import sys
from datetime import datetime

import einops
import gymnasium  # as gym
import numpy as np
import ruamel.yaml as yaml
import torch
import torch.nn.functional as F

# note: need to include the dreamerv3 repo for this
from termcolor import cprint

import PyHJ

dreamer_dir = os.path.abspath("/home/sunny/AnySafe_Reachability/dreamerv3_torch")
sys.path.append(dreamer_dir)
saferl_dir = os.path.abspath("/home/sunny/AnySafe_Reachability/PyHJ")
sys.path.append(saferl_dir)
print(sys.path)
import models
import tools
from dreamer import make_dataset

# note: need to include the dreamerv3 repo for this
from tqdm import *

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.extend(
    [
        base_dir,
    ]
)

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # set random seed for all gpus

device = "cuda:0"

# Backbone Model
LOG_DIR = "logs_pa"


def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--expt_name", type=str, default=None)
    parser.add_argument("--resume_run", type=bool, default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    # environment parameters
    config, remaining = parser.parse_known_args()

    if not config.resume_run:
        curr_time = datetime.now().strftime("%m%d/%H%M%S")
        config.expt_name = (
            f"{curr_time}_{config.expt_name}" if config.expt_name else curr_time
        )
    else:
        assert config.expt_name, "Need to provide experiment name to resume run."

    yml = yaml.YAML(typ="safe", pure=True)
    with open("/home/sunny/AnySafe_Reachability/configs.yaml", "r") as f:
        configs = yml.load(f)

    name_list = ["defaults", *config.configs] if config.configs else ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args(remaining)

    final_config.logdir = f"{final_config.logdir}"
    # final_config.time_limit = HORIZONS[final_config.task.split("_")[-1]]

    print("---------------------")
    cprint(f"Experiment name: {config.expt_name}", "red", attrs=["bold"])
    cprint(f"Task: {final_config.task}", "cyan", attrs=["bold"])
    cprint(f"Logging to: {final_config.logdir + '/PyHJ'}", "cyan", attrs=["bold"])
    print("---------------------")
    return final_config


dummy_variable = PyHJ

config = get_args()
# config = tools.set_wm_name(config)

if config.sem["gpu_id"] != -1:
    torch.cuda.set_device(config.sem["gpu_id"])


env = gymnasium.make(config.task, params=[config])

config.num_actions = (
    env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
)
model = models.WorldModel(env.observation_space_full, env.action_space, 0, config)
ckpt_path = "logs/checkpoints_sem/encoder_task_dubins-wm.pth"
model.load_state_dict(torch.load(ckpt_path), strict=False)

# model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

config = tools.set_wm_name(config)

for name, param in model.named_parameters():
    param.requires_grad = False  # Just eval

# Decoder
decoder = torch.nn.Sequential(
    torch.nn.Linear(config.sem["sz_embedding"], 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 544),
).to(device)

offline_eps = collections.OrderedDict()
# config.sem["batch_size"] = 1
# config.sem["batch_length"] = 2
# tools.fill_expert_dataset_dubins(config, offline_eps, is_val_set=False)
# offline_dataset = make_dataset(offline_eps, config)
# train_len = len(offline_eps) // config.batch_length

expert_val_eps = collections.OrderedDict()
tools.fill_expert_dataset_dubins(config, expert_val_eps, is_val_set=True)
config.batch_size = 100
eval_dataset = make_dataset(expert_val_eps, config)
eval_len = len(expert_val_eps) // config.batch_length

# Dataset Loader and Sampler
BS = 300  # config.sem["sz_batch"]  # batch size
BL = config.batch_length  # batch length

losses_per_epoch = {
    "loss": [],
    "mae_loss": [],
    # "ae_loss": [],
}
auc_per_epoch = []

pbar = tqdm(
    enumerate(eval_dataset),
    total=eval_len,
    position=0,
    leave=False,
)

all_pos_pairs = []

epsilon = 0.5  # Radius threshold

for batch_idx, data in pbar:
    if batch_idx >= eval_len:
        break

    # Take only the last timestep
    state = torch.tensor(data["privileged_state"][:, :, :2]).to(device)  # [B T 2]
    state = einops.rearrange(state, "B T Z -> (B T) Z")  # [B*T, 2]

    diff = state.unsqueeze(0) - state.unsqueeze(1)  # [B, B, 2]
    dists = torch.norm(diff, dim=2)
    # labels_gt = torch.clip(1 - dists, -1, 1)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        # semantic_features: [B T 512]
        # feat_gt: [B T 544]
        semantic_features, feat_gt = model.semantic_embed(data)
        # feat_pred = decoder(semantic_features)

        # Normalize along the embedding dimension
        # (B T 512)
        sem_norm = F.normalize(
            semantic_features, p=2, dim=-1
        )  # Each row becomes unit norm

        sem_norm = einops.rearrange(sem_norm, "B T Z -> (B T) Z")
        # Compute cosine similarity via dot product → shape [B, B]
        cos_sim = sem_norm @ sem_norm.T

        # Remove double comparisons and self comparison
        mask_double = torch.triu(torch.ones_like(cos_sim), diagonal=1).bool()
        cos_sim = cos_sim[mask_double]
        dists = dists[mask_double]

        pos_pairs = cos_sim[dists < epsilon]
        all_pos_pairs.extend(pos_pairs)

all_pos_pairs = torch.tensor(all_pos_pairs)
alphas = [0.1, 0.05, 0.01, 0.005, 0.001]
thresh = np.clip(1 - 1 / (np.sqrt(2)) * epsilon, -1, 1)
print(f"For Radius epsilon = {epsilon} with a normal threshold of: {-thresh}")
for alpha in alphas:
    delta = torch.quantile(all_pos_pairs.to(torch.float32), alpha).item()
    print(f"Delta for alpha={alpha}: {-delta}")
