import argparse
import copy
import os
import pickle
import sys
from collections import defaultdict

import gymnasium  # as gym
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

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

import models
import ruamel.yaml as yaml
import tools

# note: need to include the dreamerv3 repo for this
from dreamer import make_dataset
from generate_data_traj_cont import get_frame
from PIL import Image
from termcolor import cprint

from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic

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

    final_config.logdir = f"{final_config.logdir}"
    # final_config.time_limit = HORIZONS[final_config.task.split("_")[-1]]

    print("---------------------")
    cprint(f"Experiment name: {config.expt_name}", "red", attrs=["bold"])
    cprint(f"Task: {final_config.task}", "cyan", attrs=["bold"])
    cprint(f"Logging to: {final_config.logdir + '/PyHJ'}", "cyan", attrs=["bold"])
    print("---------------------")
    return final_config


args = get_args()
config = args
config.nb_classes = 5


env = gymnasium.make(args.task, params=[config])
config.num_actions = (
    env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
)
wm = models.WorldModel(env.observation_space_full, env.action_space, 0, config)

ckpt_path = "logs/checkpoints_pa/encoder_task_dubins-wm.pth"
wm.load_state_dict(torch.load(ckpt_path), strict=False)
wm.eval()

offline_eps = collections.OrderedDict()
config.batch_size = 1
config.batch_length = 2
tools.fill_expert_dataset_dubins(config, offline_eps)
offline_dataset = make_dataset(offline_eps, config)

env.set_wm(wm, offline_dataset, config)


# check if the environment has control and disturbance actions:
assert hasattr(
    env, "action_space"
)  # and hasattr(env, 'action2_space'), "The environment does not have control and disturbance actions!"
if isinstance(env.observation_space, gymnasium.spaces.Dict):
    args.state_shape = (
        env.observation_space["state"].shape or env.observation_space["state"].n
    )
else:
    args.state_shape = env.observation_space.shape or env.observation_space.n
args.constraint_dim = env.constraint_shape
args.action_shape = env.action_space.shape or env.action_space.n
args.max_action = env.action_space.high[0]


train_envs = DummyVectorEnv(
    [
        lambda: gymnasium.make(args.task, params=[wm, offline_dataset, config])
        for _ in range(args.training_num)
    ]
)
test_envs = DummyVectorEnv(
    [
        lambda: gymnasium.make(args.task, params=[wm, offline_dataset, config])
        for _ in range(args.test_num)
    ]
)

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
train_envs.seed(args.seed)
test_envs.seed(args.seed)
# model

if args.actor_activation == "ReLU":
    actor_activation = torch.nn.ReLU
elif args.actor_activation == "Tanh":
    actor_activation = torch.nn.Tanh
elif args.actor_activation == "Sigmoid":
    actor_activation = torch.nn.Sigmoid
elif args.actor_activation == "SiLU":
    actor_activation = torch.nn.SiLU

if args.critic_activation == "ReLU":
    critic_activation = torch.nn.ReLU
elif args.critic_activation == "Tanh":
    critic_activation = torch.nn.Tanh
elif args.critic_activation == "Sigmoid":
    critic_activation = torch.nn.Sigmoid
elif args.critic_activation == "SiLU":
    critic_activation = torch.nn.SiLU

if args.critic_net is not None:
    critic_net = Net(
        args.state_shape,
        obs_inputs=["state", "constraint"]
        if args.safety_margin_type == "cos_sim"
        else ["state"],
        action_shape=args.action_shape,
        hidden_sizes=args.critic_net,
        constraint_dim=args.constraint_dim,
        constraint_embedding_dim=args.constraint_embedding_dim,
        hidden_sizes_constraint=args.control_net_const,
        activation=critic_activation,
        concat=True,
        device=args.device,
    )
else:
    # report error:
    raise ValueError("Please provide critic_net!")

critic = Critic(critic_net, device=args.device).to(args.device)
critic_optim = torch.optim.AdamW(
    critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay_pyhj
)

log_path = None

from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy

print(
    "DDPG under the Avoid annealed Bellman equation with no Disturbance has been loaded!"
)

actor_net = Net(
    args.state_shape,
    obs_inputs=["state", "constraint"]
    if args.safety_margin_type == "cos_sim"
    else ["state"],
    hidden_sizes=args.control_net,
    activation=actor_activation,
    device=args.device,
    constraint_dim=args.constraint_dim,
    constraint_embedding_dim=args.constraint_embedding_dim,
    hidden_sizes_constraint=args.control_net_const,
)
actor = Actor(
    actor_net, args.action_shape, max_action=args.max_action, device=args.device
).to(args.device)
actor_optim = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)


policy = DDPGPolicy(
    critic,
    critic_optim,
    tau=args.tau,
    gamma=args.gamma_pyhj,
    exploration_noise=GaussianNoise(sigma=args.exploration_noise),
    reward_normalization=args.rew_norm,
    estimation_step=args.n_step,
    action_space=env.action_space,
    actor=actor,
    actor_optim=actor_optim,
    actor_gradient_steps=args.actor_gradient_steps,
)

epoch_id = 6

state_type = "z_sem" if args.pass_semantic_state else "z"
constraint_type = "z_c_sem" if args.pass_semantic_constraint else "z_c"
policy.load_state_dict(
    torch.load(
        f"/home/sunny/AnySafe_Reachability/scripts/logs/dreamer_dubins/PyHJ/sim_{args.safety_margin_type}_dist_type_{args.env_dist_type}_V({state_type}, {constraint_type})_const_embd_{args.constraint_embedding_dim}/epoch_id_{epoch_id}/policy.pth"
    )
)

log_path = os.path.join(
    args.logdir + "/PyHJ", args.task, f"wm_dist_type_{args.env_dist_type}"
)


if args.continue_training_epoch is not None:
    epoch = args.continue_training_epoch
    policy.load_state_dict(
        torch.load(os.path.join(log_path + "/epoch_id_{}".format(epoch), "policy.pth"))
    )


def stop_fn(mean_rewards):
    return False


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    return img.convert("RGB")


def make_cache(config, thetas):
    nx, ny = config.nx, config.ny

    cache_file = os.path.join(log_path, "cache.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)

        if cache[0][0].shape[0] != nx * ny:
            print("Cache file exists but has different dimensions, recreating cache.")
        elif set(cache.keys()) != set(thetas):
            print("Cache file exists but has different keys, recreating cache.")
        else:
            print("Cache file exists and has correct dimensions and thetas, using it.")
            return cache
    else:
        print(f"Didn't find cache at {cache_file}, creating it.")

    cache = {}
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


thetas = [3 * np.pi / 2, 7 * np.pi / 4, 0, np.pi / 4, np.pi / 2, np.pi]
if args.debug:
    thetas = [0, np.pi / 6]
    print("Debug mode: using fewer thetas for debugging purposes.")
    args.step_per_epoch = 10
cache = make_cache(config, thetas)

# env.config.env_dist_type = "v"

for in_dist in [True]:
    for i in range(5):
        plot1, plot2, plot3, metric = env.get_eval_plot(
            cache=cache,
            thetas=thetas,
            config=config,
            policy=policy,
            in_distribution=in_dist,
        )
        plot1.savefig(f"results/plot_{i}.png", dpi=300, bbox_inches="tight")

    all_metrics = []
    for __ in range(50):
        all_metrics.append(
            copy.deepcopy(
                env.get_eval_metrics(
                    cache=cache,
                    thetas=thetas,
                    config=config,
                    policy=policy,
                    in_distribution=in_dist,
                )
            )
        )

    aggregated = defaultdict(list)

    for metrics in all_metrics:
        for key, value in metrics.items():
            aggregated[key].append(value)

    # Compute averages
    aggregate_metrics = {key: np.sum(values) for key, values in aggregated.items()}

    in_dist_label = "in_dist" if in_dist else "out_dist"

    TPR = aggregate_metrics["TP"] / (
        aggregate_metrics["TP"] + aggregate_metrics["FN"] + 1e-8
    )
    FPR = aggregate_metrics["FP"] / (
        aggregate_metrics["FP"] + aggregate_metrics["TN"] + 1e-8
    )
    FNR = aggregate_metrics["FN"] / (
        aggregate_metrics["FN"] + aggregate_metrics["TP"] + 1e-8
    )
    TNR = aggregate_metrics["TN"] / (
        aggregate_metrics["TN"] + aggregate_metrics["FP"] + 1e-8
    )
    Accuracy = (aggregate_metrics["TP"] + aggregate_metrics["TN"]) / (
        aggregate_metrics["TP"]
        + aggregate_metrics["FP"]
        + aggregate_metrics["FN"]
        + aggregate_metrics["TN"]
        + 1e-8
    )
    Balanced_Accuracy = 0.5 * (TPR + TNR)
    Precision = aggregate_metrics["TP"] / (
        aggregate_metrics["TP"] + aggregate_metrics["FP"] + 1e-8
    )
    Recall = TPR
    F1 = 2 * (Precision * Recall) / (Precision + Recall + 1e-8)
    Intersection = np.sum((aggregate_metrics["TP"]))
    Union = np.sum(
        (aggregate_metrics["TP"] + aggregate_metrics["FP"] + aggregate_metrics["FN"])
    )
    IOU = Intersection / (Union + 1e-8)
    aggregate_metrics.update(
        {
            "TPR": TPR,
            "FPR": FPR,
            "FNR": FNR,
            "TNR": TNR,
            "Accuracy": Accuracy,
            "Balanced Accuracy": Balanced_Accuracy,
            "Precision": Precision,
            "Recall": Recall,
            "F1": F1,
            "IOU": IOU,
        }
    )

success_rate = env.get_success_rate(policy=policy)
aggregate_metrics.update({"Success Rate": success_rate})
print("Averaged metrics over 50 runs:")
for key, value in aggregate_metrics.items():
    print(f"{key}: {value}")

# Save as mp4

# trajs = []

# for i in tqdm(range(30)):
#     traj_imgs = env.get_trajectory(policy=policy)
#     trajs.append(traj_imgs)

# trajs = np.concatenate(trajs, axis=0)
# # Save video as mp4 from numpy array
# video_frames = np.transpose(trajs, (0, 2, 3, 1))
# import imageio

# imageio.mimsave("output.mp4", video_frames, fps=20)

save_path = f"/home/sunny/AnySafe_Reachability/scripts/logs/dreamer_dubins/PyHJ/sim_{args.safety_margin_type}_dist_type_{args.env_dist_type}_V({state_type}, {constraint_type})_const_embd_{args.constraint_embedding_dim}/epoch_id_{epoch_id}"
with open(f"{save_path}/metrics.txt", "w") as f:
    for key, value in aggregate_metrics.items():
        f.write(f"{key}: {value}\n")

import ipdb

ipdb.set_trace()

# plt.close()
