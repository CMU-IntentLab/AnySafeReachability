import argparse
import copy
import os
import pickle
import sys
from collections import defaultdict

import gymnasium  # as gym
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

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

import wandb
from PyHJ.data import Collector, VectorReplayBuffer
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.trainer import offpolicy_trainer
from PyHJ.utils import TensorboardLogger, WandbLogger
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

# config = tools.set_wm_name(config)

# ckpt_path = config.rssm_ckpt_path
# ckpt_path = "logs/dreamer_dubins/dubins_mlp_obs_state_cnn_image_lz_None_sc_F_arrow_0.15/rssm_ckpt.pt"
# checkpoint = torch.load(ckpt_path)
# state_dict = {
#     k[14:]: v for k, v in checkpoint["agent_state_dict"].items() if "_wm" in k
# }
# wm.load_state_dict(state_dict, strict=False)
ckpt_path = "logs/checkpoints_sem/encoder_task_dubins-wm.pth"
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

state_type = "z_sem" if args.pass_semantic_state else "z"
if args.pass_semantic_constraint and not args.pass_prototype:
    constraint_type = "z_c_sem"
elif args.pass_semantic_constraint and args.pass_prototype:
    constraint_type = "p"
else:
    constraint_type = "z_c"

log_path = os.path.join(
    args.logdir + "/PyHJ",
    f"sim_{args.safety_margin_type}_dist_type_{args.env_dist_type}_V({state_type}, {constraint_type})_const_embd_{args.constraint_embedding_dim}",
)


# collector
train_collector = Collector(
    policy,
    train_envs,
    VectorReplayBuffer(args.buffer_size, len(train_envs)),
    exploration_noise=True,
)
test_collector = Collector(policy, test_envs)

if args.warm_start_path is not None:
    policy.load_state_dict(torch.load(args.warm_start_path))
    args.kwargs = args.kwargs + "warmstarted"

epoch = 0
# writer = SummaryWriter(log_path, filename_suffix="_"+timestr+"epoch_id_{}".format(epoch))
# logger = TensorboardLogger(writer)
# log_path = (
#     log_path
#     + "/noise_{}_actor_lr_{}_critic_lr_{}_batch_{}_step_per_epoch_{}_kwargs_{}_seed_{}".format(
#         args.exploration_noise,
#         args.actor_lr,
#         args.critic_lr,
#         args.batch_size_pyhj,
#         args.step_per_epoch,
#         args.kwargs,
#         args.seed,
#     )
# )


if args.continue_training_epoch is not None:
    epoch = args.continue_training_epoch
    policy.load_state_dict(
        torch.load(os.path.join(log_path + "/epoch_id_{}".format(epoch), "policy.pth"))
    )


if args.continue_training_logdir is not None:
    policy.load_state_dict(torch.load(args.continue_training_logdir))
    # epoch = int(args.continue_training_logdir.split('_')[-9].split('_')[0])
    epoch = args.continue_training_epoch


def save_best_fn(policy, epoch=epoch):
    torch.save(
        policy.state_dict(),
        os.path.join(log_path + "/epoch_id_{}".format(epoch), "policy.pth"),
    )


def stop_fn(mean_rewards):
    return False


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    return img.convert("RGB")


if not os.path.exists(log_path + "/epoch_id_{}".format(epoch)):
    print("Just created the log directory!")
    # print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
    os.makedirs(log_path + "/epoch_id_{}".format(epoch))


def make_cache(config, thetas):
    nx, ny = config.nx, config.ny

    cache_file = os.path.normpath(os.path.join(log_path, "..", "cache.pkl"))
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)

        if cache[0][0].shape[0] != nx * ny:
            print("Cache file exists but has different dimensions, recreating cache.")
        elif set(cache.keys()) != set(thetas):
            print("Cache file exists but has different keys, recreating cache.")
        else:
            print("Cache file exists and has correct dimensions, using it.")
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
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)
    return cache


if not os.path.exists(log_path + "/epoch_id_{}".format(epoch)):
    print("Just created the log directory!")
    # print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
    os.makedirs(log_path + "/epoch_id_{}".format(epoch))
thetas = [3 * np.pi / 2, 7 * np.pi / 4, 0, np.pi / 4, np.pi / 2, np.pi]
if args.debug:
    thetas = [0, np.pi / 6]
    print("Debug mode: using fewer thetas for debugging purposes.")
    args.step_per_epoch = 10
cache = make_cache(config, thetas)
logger = None
warmup = args.warmup
# plot1, plot2, plot3, metrics = env.get_eval_plot(
#     cache=cache, thetas=thetas, config=config, policy=policy
# )

for iter in range(warmup + args.total_episodes):
    if iter < warmup:
        policy._gamma = 0  # for warmup the value fn
        policy.warmup = True
    else:
        policy._gamma = config.gamma_pyhj
        policy.warmup = False

    if args.continue_training_epoch is not None:
        print(
            "epoch: {}, remaining epochs: {}".format(
                epoch // args.epoch, args.total_episodes - iter
            )
        )
    else:
        print(
            "epoch: {}, remaining epochs: {}".format(iter, args.total_episodes - iter)
        )
    epoch = epoch + args.epoch
    print("log_path: ", log_path + "/epoch_id_{}".format(epoch))
    if args.total_episodes > 1:
        writer = SummaryWriter(
            log_path + "/epoch_id_{}".format(epoch)
        )  # filename_suffix="_"+timestr+"_epoch_id_{}".format(epoch))
    else:
        if not os.path.exists(log_path + "/total_epochs_{}".format(epoch)):
            print("Just created the log directory!")
            print("log_path: ", log_path + "/total_epochs_{}".format(epoch))
            os.makedirs(log_path + "/total_epochs_{}".format(epoch))
        writer = SummaryWriter(
            log_path + "/total_epochs_{}".format(epoch)
        )  # filename_suffix="_"+timestr+"_epoch_id_{}".format(epoch))
    if logger is None:
        task_name = (
            args.task
        )  # .split("-")[:-1]  # Take everything before the last dash

        wb_name_args = [
            # f"{task_name}",
            # "DDPG",
            f"V({state_type}, {constraint_type})",
            f"dist_type_{config.env_dist_type}",
            f"sim_{config.safety_margin_type}_{config.safety_margin_threshold}{'*' if config.safety_margin_hard_threshold else ''}",
            "proto" if config.pass_prototype else None,
            f"critic_{config.critic_net}",
            f"control_{config.control_net}",
            # f"{config.wm_name}",
        ]
        wandb_name = ""
        for arg in wb_name_args:
            if arg is not None:
                wandb_name += f"{arg}_"
        wandb_name = wandb_name[:-1]  # Remove the last underscore
        logger = WandbLogger(name=wandb_name, project="Dubins", config=config)
        logger.load(writer)

        wandb.define_metric("num_epochs", step_metric="num_epochs")
        wandb.define_metric("*", step_metric="num_epochs")
    logger = TensorboardLogger(writer)

    # import pdb; pdb.set_trace()
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size_pyhj,
        update_per_step=args.update_per_step,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    )

    save_best_fn(policy, epoch=epoch)
    for in_dist in [True, False]:
        plot1, plot2, plot3, __ = env.get_eval_plot(
            cache=cache,
            thetas=thetas,
            config=config,
            policy=policy,
            in_distribution=in_dist,
        )
        all_metrics = []
        for __ in range(5):
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
        metrics = {}
        TPR = aggregate_metrics["TP"] / (
            aggregate_metrics["TP"] + aggregate_metrics["FN"] + 1e-8
        )
        FPR = aggregate_metrics["FP"] / (
            aggregate_metrics["FP"] + aggregate_metrics["TN"] + 1e-8
        )
        FNR = aggregate_metrics["FN"] / (
            aggregate_metrics["TP"] + aggregate_metrics["FN"] + 1e-8
        )
        TNR = aggregate_metrics["TN"] / (
            aggregate_metrics["TN"] + aggregate_metrics["FP"] + 1e-8
        )
        Accuracy = (aggregate_metrics["TP"] + aggregate_metrics["TN"]) / (
            aggregate_metrics["TP"]
            + aggregate_metrics["TN"]
            + aggregate_metrics["FP"]
            + aggregate_metrics["FN"]
            + 1e-8
        )
        Precision = aggregate_metrics["TP"] / (
            aggregate_metrics["TP"] + aggregate_metrics["FP"] + 1e-8
        )
        Recall = TPR
        F1 = 2 * (Precision * Recall) / (Precision + Recall + 1e-8)
        Balanced_Accuracy = 0.5 * (TPR + TNR)
        metrics.update(
            {
                "TPR": TPR,
                "FPR": FPR,
                "FNR": FNR,
                "TNR": TNR,
                "Accuracy": Accuracy,
                "Precision": Precision,
                "Recall": Recall,
                "F1": F1,
                "Balanced_Accuracy": Balanced_Accuracy,
            }
        )

        in_dist_label = "in_dist" if in_dist else "out_dist"
        wandb.log(
            {
                f"{in_dist_label}/binary_reach_avoid_plot": wandb.Image(plot1),
                f"{in_dist_label}/continuous_plot": wandb.Image(plot2),
                f"{in_dist_label}/safety_margin_function": wandb.Image(plot3),
                **{f"{in_dist_label}/metric/{k}": v for k, v in metrics.items()},
                "num_epochs": epoch - 1,
            },
        )

    traj_imgs = env.get_trajectory(policy=policy)
    wandb.log(
        {
            "trajectory": wandb.Video(
                np.array(traj_imgs),
                fps=10,
                format="mp4",
            ),
            "num_epochs": epoch - 1,
        }
    )

    plt.close()
