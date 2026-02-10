import argparse
import os
import pickle
import sys

import einops
import gymnasium  # as gym
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import PyHJ
import wandb
from PyHJ.utils import WandbLogger

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../dreamerv3_torch")
)
sys.path.append(dreamer_dir)
saferl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "/PyHJ"))
sys.path.append(saferl_dir)
print(sys.path)
import collections
import io
import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import models
import ruamel.yaml as yaml
import tools

# note: need to include the dreamerv3 repo for this
from dreamer import make_dataset
from generate_data_traj_cont import get_frame
from PIL import Image
from termcolor import cprint

# NOTE: all the reach-avoid gym environments are in reach_rl_gym, the constraint information is output as an element of the info dictionary in gym.step() function
"""
    Note that, we can pass arguments to the script by using
    python run_training_ddpg.py --task ra_droneracing_Game-v6 --control-net 512 512 512 512 --disturbance-net 512 512 512 512 --critic-net 512 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9
    python run_training_ddpg.py --task ra_highway_Game-v2 --control-net 512 512 512 --disturbance-net 512 512 512 --critic-net 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9
    python run_training_ddpg.py --task ra_1d_Game-v0 --control-net 32 32 --disturbance-net 4 4 --critic-net 4 4 --epoch 10 --total-episodes 160 --gamma 0.9
    
    For learning the classical reach-avoid value function (baseline):
    python run_training_ddpg.py --task ra_droneracing_Game-v6 --control-net 512 512 512 512 --disturbance-net 512 512 512 512 --critic-net 512 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9 --is-game-baseline True
    python run_training_ddpg.py --task ra_highway_Game-v2 --control-net 512 512 512 --disturbance-net 512 512 512 --critic-net 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9 --is-game-baseline True
    python run_training_ddpg.py --task ra_1d_Game-v0 --control-net 32 32 --disturbance-net 4 4 --critic-net 4 4 --epoch 10 --total-episodes 160 --gamma 0.9 --is-game-baseline True

"""


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
    configs = yml.load(
        # (pathlib.Path(sys.argv[0]).parent / "../configs/config.yaml").read_text()
        (pathlib.Path(sys.argv[0]).parent / "../configs.yaml").read_text()
    )

    name_list = ["defaults", *config.configs] if config.configs else ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args(remaining)

    final_config.logdir = f"{final_config.logdir + '/PyHJ'}/{config.expt_name}"
    # final_config.time_limit = HORIZONS[final_config.task.split("_")[-1]]

    print("---------------------")
    cprint(f"Experiment name: {config.expt_name}", "red", attrs=["bold"])
    cprint(f"Task: {final_config.task}", "cyan", attrs=["bold"])
    cprint(f"Logging to: {final_config.logdir + '/PyHJ'}", "cyan", attrs=["bold"])
    print("---------------------")
    return final_config


dummy_variable = PyHJ

args = get_args()
config = args
config.grid_size = 2
config.nb_classes = config.grid_size**2 + 1

env = gymnasium.make(args.task, params=[config])
config.num_actions = (
    env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
)
wm = models.WorldModel(env.observation_space_full, env.action_space, 0, config)

config = tools.set_wm_name(config)

ckpt_path = "logs/checkpoints_pa/encoder_task_dubins-wm.pth"
# ckpt_path = "logs/dreamer_dubins/dubins_mlp_obs_state_cnn_image_lz_None_sc_F_arrow_0.15/rssm_ckpt.pt"
# checkpoint = torch.load(ckpt_path, weights_only=True)
# state_dict = {
#     k[14:]: v for k, v in checkpoint["agent_state_dict"].items() if "_wm" in k
# }

if not os.path.exists(ckpt_path):
    raise ValueError(f"Checkpoint path {ckpt_path} does not exist!")
wm.load_state_dict(torch.load(ckpt_path), strict=False)
# wm.load_state_dict(state_dict, strict=False)
wm.eval()

offline_eps = collections.OrderedDict()
config.batch_size = 1
config.batch_length = 2
tools.fill_expert_dataset_dubins(config, offline_eps)
offline_dataset = make_dataset(offline_eps, config)

env.set_wm(wm, offline_dataset, config)

log_path = os.path.join(
    args.logdir + "/PyHJ",
    args.task,
    "wm_actor_activation_{}_critic_activation_{}_game_gd_steps_{}_tau_{}_training_num_{}_buffer_size_{}_c_net_{}_{}_a1_{}_{}_gamma_{}".format(
        args.actor_activation,
        args.critic_activation,
        args.actor_gradient_steps,
        args.tau,
        args.training_num,
        args.buffer_size,
        args.critic_net[0],
        len(args.critic_net),
        args.control_net[0],
        len(args.control_net),
        args.gamma_pyhj,
    ),
)


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    return img.convert("RGB")


def make_cache(config, thetas):
    nx, ny = config.nx, config.ny
    cache = {}

    cache_file = os.path.join(log_path, "cache.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)

        if cache[0][0].shape[0] != nx * ny:
            print("Cache size mismatch, recreating cache...")
            cache = {}
        else:
            return cache

    # Make new cache
    for theta in thetas:
        v = np.zeros((nx, ny))
        xs = np.linspace(-1.1, 1.1, nx, endpoint=True)
        ys = np.linspace(-1.1, 1.1, ny, endpoint=True)
        key = theta
        print("creating cache for key", key)
        idxs, imgs_prev, thetas, thetas_prev = [], [], [], []
        xs_prev = xs - config.dt * config.speed * np.cos(theta)
        ys_prev = ys - config.dt * config.speed * np.sin(theta)
        theta_prev = theta
        it = np.nditer(v, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            x_prev = xs_prev[idx[0]]
            y_prev = ys_prev[idx[1]]
            thetas.append(theta)
            thetas_prev.append(theta_prev)
            imgs_prev.append(
                get_frame(torch.tensor([x_prev, y_prev, theta_prev]), config)
            )
            idxs.append(idx)
            it.iternext()
        idxs = np.array(idxs)
        theta_prev_lin = np.array(thetas_prev)
        cache[theta] = [idxs, imgs_prev, theta_prev_lin]

    # pickle file
    cache_file = os.path.join(log_path, "cache.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)

    return cache


def get_latent(
    wm,
    thetas,  # (N)
    imgs,  # N-List of (H, W, C)
):
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
    stoch = post["stoch"]  # z_t
    deter = post["deter"]  # h_t
    semantic = wm.semantic_encoder(feat)
    return feat.squeeze().cpu().numpy(), stoch, deter, semantic.squeeze().cpu().numpy()


def topographic_map(
    config,
    cache,
    thetas,
    constraint_states,
    similarity_metric,
    model=None,
    use_semantic=True,
):
    constraint_states = torch.tensor(constraint_states, dtype=torch.float32)

    constraint_imgs = []
    for constraint_state in constraint_states:
        constraint_state = torch.tensor(constraint_state, dtype=torch.float32)
        constraint_img = get_frame(states=constraint_state, config=config)  # (H, W, C)
        constraint_imgs.append(constraint_img)
    # constraint_imgs = torch.stack(constraint_imgs, dim=0)  # (N, H, W, C)

    with torch.no_grad():
        feat_c, stoch_c, deter_c, semantic_c = get_latent(  # [N, Z]
            wm, thetas=np.array(constraint_states[:, -1]), imgs=constraint_imgs
        )
        if feat_c.ndim == 1:
            feat_c = feat_c.reshape(1, -1)  # [1, Z]
            semantic_c = semantic_c.reshape(1, -1)  # [1, Z]

        if use_semantic:
            feature_c = semantic_c
        else:
            feature_c = feat_c

    idxs, __, __ = cache[thetas[0]]

    feature_c = einops.repeat(feature_c, "N C -> B N C", B=idxs.shape[0])  # [B, N, Z]

    fig, axes = plt.subplots(
        feature_c.shape[1],
        len(thetas) + 1,
        figsize=(3 * len(thetas), 3 * feature_c.shape[1]),
        constrained_layout=True,
    )

    for i in range(len(thetas)):
        theta = thetas[i]
        i += 1  # offset for constraint and safe images
        for ax in axes[:, i]:
            ax.set_title(f"theta = {theta:.2f}")
        idxs, imgs_prev, thetas_prev = cache[theta]
        with torch.no_grad():
            feat, stoch, deter, semantic = get_latent(
                wm, thetas_prev, imgs_prev
            )  # [B, Z]
            if use_semantic:
                feature = semantic  # [B, N, Z]
            else:
                feature = feat
            feature = einops.repeat(
                feature, "B C -> B N C", N=feature_c.shape[1]
            )  # [B, N, Z]
        if similarity_metric == "Cosine_Similarity":  # negative cosine similarity
            numerator = np.sum(feature * feature_c, axis=-1)  # (B, N)
            denominator = np.linalg.norm(feature, axis=-1) * np.linalg.norm(  # (B, N)
                feature_c, axis=-1
            )
            metric = -numerator / (denominator + 1e-8)  # (B, N)
            # if use_semantic:
            #     metric += 0.5
        elif similarity_metric == "Euclidean Distance":
            metric = -np.linalg.norm(feature - feature_c, axis=-1)  # (B, N)
        elif similarity_metric == "Learned":
            assert model is not None, (
                "Model must be provided for learned similarity metric."
            )
            feature = torch.tensor(feature, dtype=torch.float32)
            feature_c = torch.tensor(feature_c, dtype=torch.float32)
            metric = torch.tanh(model(feature, feature_c))  # (B, N)
            metric = metric.detach().cpu().numpy()  # (B, N)
        else:
            raise ValueError(
                f"Unknown similarity metric: {similarity_metric}. Supported: ['Cosine_Similarity', 'Euclidean Distance', 'Learned']"
            )

        if similarity_metric == "Cosine_Similarity":
            pred_vals = einops.rearrange(
                torch.tensor(metric[:, 0]), "(W H) -> H W", H=config.ny, W=config.nx
            )  # flatten for plotting
            grid_points = torch.meshgrid(
                torch.linspace(-1.1, 1.1, config.nx),
                torch.linspace(-1.1, 1.1, config.ny),
            )
            grid_points = torch.stack(grid_points, dim=-1)
            dists = torch.norm((grid_points - torch.tensor([0.0, 0.0])), dim=-1)
            gt_vals = dists > 0.5

            # Flatten
            pred_flat = pred_vals.flatten()
            gt_flat = gt_vals.flatten()

            # Get unique sorted prediction values (empirical thresholds)
            thresholds = torch.arange(
                pred_flat.min(), pred_flat.max() + 0.01, 0.01
            )  # [T]

            # Move to NumPy for sklearn
            pred_np = pred_flat.cpu().numpy()
            gt_np = gt_flat.cpu().numpy()

            # For all thresholds, compute predicted labels
            # Shape: [T, N]
            pred_labels = (pred_np[None, :] > thresholds[:, None].cpu().numpy()).astype(
                np.uint8
            )

            # Compute F1 for each threshold (vectorized)
            # (scikit-learn expects 1D inputs, so use list comprehension)
            accuracies = np.array(
                [np.mean(gt_np == pred_labels[i]) for i in range(len(thresholds))]
            )
            f1_scores = np.array(
                [f1_score(gt_np, pred_labels[i]) for i in range(len(thresholds))]
            )

            # Find the best threshold based on F1 score
            best_idx = f1_scores.argmax()
            best_thresh = thresholds[best_idx].item()
            best_accuracy = accuracies[best_idx]

            print(
                f"Best threshold: {best_thresh:.4f}, Best F1 Score: {f1_scores[best_idx]:.4f}"
            )

            # Pick the best threshold
            best_idx = accuracies.argmax()
            best_thresh = thresholds[best_idx].item()
            best_accuracy = accuracies[best_idx]

            print(
                f"Best threshold: {best_thresh:.4f}, Best accuracy: {best_accuracy:.4f}"
            )
            exit()

        metrics = einops.rearrange(metric, "(W H) N -> N H W", W=config.nx, H=config.ny)
        for j, metric in enumerate(metrics):
            axes[j, i].imshow(
                metric,
                extent=(-1.1, 1.1, -1.1, 1.1),
                vmin=-1,
                vmax=1,
                origin="lower",
            )

        # x = np.linspace(-1.1, 1.1, metric.shape[1])
        # y = np.linspace(-1.1, 1.1, metric.shape[0])
        # X, Y = np.meshgrid(x, y)

        # contour = axes[i].contour(X, Y, metric, levels=5, colors="black", linewidths=1)
        # axes[i].clabel(contour, inline=True, fontsize=8, fmt="%.2f")

    for j, constraint_img in enumerate(constraint_imgs):
        # Show the constraint image on the topographic map
        axes[j, 0].imshow(
            constraint_img,
            extent=(config.x_min, config.x_max, config.y_min, config.y_max),
        )
        axes[j, 0].set_title(f"Constraint Image {j}")

    # set axes limits
    for ax in axes.flat:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect("equal")

    fig.suptitle(f"Topographic Map using {similarity_metric}")
    plt.tight_layout(pad=1.0, h_pad=0.2)  # reduce vertical padding
    return fig


def topographic_map_proxies(
    config,
    cache,
    thetas,
    model=None,
):
    idxs, __, __ = cache[thetas[0]]

    fig, axes = plt.subplots(
        len(wm.proxies),
        len(thetas),
        figsize=(3 * len(thetas), 3 * len(wm.proxies)),
        constrained_layout=True,
    )

    for proxy_idx, proxy in enumerate(wm.proxies):
        feature_c = (
            einops.repeat(proxy, "C -> B N C", B=idxs.shape[0], N=1)
            .detach()
            .cpu()
            .numpy()
        )  # [B, N, Z]

        for i in range(len(thetas)):
            theta = thetas[i]
            axes[proxy_idx, i].set_title(f"theta = {theta:.2f}, proxy = {proxy_idx}")
            idxs, imgs_prev, thetas_prev = cache[theta]
            with torch.no_grad():
                feat, stoch, deter, semantic = get_latent(
                    wm, thetas_prev, imgs_prev
                )  # [B, Z]
                feature = einops.repeat(
                    semantic, "B C -> B N C", N=feature_c.shape[1]
                )  # [B, N, Z]

            numerator = np.sum(feature * feature_c, axis=-1)  # (B, N)
            denominator = np.linalg.norm(feature, axis=-1) * np.linalg.norm(  # (B, N)
                feature_c, axis=-1
            )
            metric_const = -numerator / (denominator + 1e-8)  # (B, N)
            metric = np.min(metric_const, axis=-1)  # (B,)

            metric = metric.reshape(config.nx, config.ny).T
            axes[proxy_idx, i].imshow(
                metric,
                extent=(-1.1, 1.1, -1.1, 1.1),
                vmin=-1,
                vmax=1,
                origin="lower",
            )
            # x = np.linspace(-1.1, 1.1, metric.shape[1])
            # y = np.linspace(-1.1, 1.1, metric.shape[0])
            # X, Y = np.meshgrid(x, y)

            # contour = axes[proxy_idx, i].contour(
            #     X, Y, metric, levels=5, colors="black", linewidths=1
            # )
            # axes[proxy_idx, i].clabel(contour, inline=True, fontsize=8, fmt="%.2f")

    # set axes limits
    for ax in axes.flat:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect("equal")

    fig.suptitle("Topographic Map using proxies")
    plt.tight_layout()
    return fig


thetas = [0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi, 3 * np.pi / 2]
cache = make_cache(config, thetas)
logger = None
warmup = 1


class SafetyMargin(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SafetyMargin, self).__init__()
        layers = []

        # Create hidden layers
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.SiLU())
            last_dim = hidden_dim

        # Final output layer (no activation here)
        layers.append(nn.Linear(last_dim, output_dim))
        # layers.append(nn.Tanh())  # Use Tanh to keep output in [-1, 1]

        self.model = nn.Sequential(*layers)

    def forward(self, z, z_const):
        input = torch.cat((z, z_const), dim=-1)
        return self.model(input)


safety_margin = SafetyMargin(input_dim=2 * 544, hidden_dims=[512, 256], output_dim=1)
# Load the pre-trained model
model_path = "safety_margin_model.pth"
if os.path.exists(model_path):
    safety_margin.load_state_dict(torch.load(model_path, map_location=config.device))
    print("Safety Margin model loaded successfully.")
else:
    print(f"Model file {model_path} not found. Using untrained model.")

similarity_metrics = ["Cosine_Similarity"]  # , "Euclidean Distance", "Learned"]

logger = WandbLogger(
    name=f"wm_Analysis_{config.wm_name}", config=config, project="WM Analysis"
)

for metric in similarity_metrics:
    for use_semantic in [True, False]:
        constraint_list = [
            [0.0, 0.0, 0.0],  # 0.0],  # x, y, theta
            [0.25, 0.25, np.pi / 2],
            [-0.25, -0.75, -np.pi / 2],
            [0.0, 0.75, np.pi / 2],
            [-0.5, 0.25, -np.pi / 2],
        ]
        fig = topographic_map(
            config=config,
            cache=cache,
            thetas=thetas,
            constraint_states=constraint_list,
            similarity_metric=metric,
            model=safety_margin if metric == "Learned" else None,
            use_semantic=use_semantic,
        )

        wandb.log(
            {
                f"{metric}_constraint{'_semantic' if use_semantic else ''}": wandb.Image(
                    fig
                ),
            }
        )

        plt.close(fig)

fig = topographic_map_proxies(config=config, cache=cache, thetas=thetas)
wandb.log(
    {
        "Proxy-Based": wandb.Image(fig),
    }
)

plt.close(fig)
