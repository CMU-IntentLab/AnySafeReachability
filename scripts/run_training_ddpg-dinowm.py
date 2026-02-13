import os
import sys

import gymnasium
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import argparse
import os
import sys

# from dreamer import make_dataset
# NOTE: all the reach-avoid gym environments are in reach_rl_gym, the constraint information is output as an element of the info dictionary in gym.step() function
from torch.utils.data import DataLoader
from tqdm import *

import wandb
from dino_wm.models.dino_models import VideoTransformer
from dino_wm.utils.utils import load_state_dict_flexible
from dino_wm.utils.test_loader import SplitTrajectoryDataset
from PyHJ.data import Collector, VectorReplayBuffer
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.trainer import offpolicy_trainer
from PyHJ.utils import WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic


def parse_args(parser):
    # int argument
    parser.add_argument("--class-id", type=int, help="An integer value.")

    # boolean argument with default
    parser.add_argument(
        "--latent-safe",
        action="store_true",  # Python 3.9+
        default=False,
    )
    return parser.parse_args()


parser = argparse.ArgumentParser(description="Example parser for an int and a boolean.")
args = parse_args(parser)

wm = VideoTransformer(
    image_size=(224, 224),
    dim=384,  # DINO feature dimension
    ac_dim=10,  # Action embedding dimension
    state_dim=3,  # State dimension
    depth=6,
    heads=16,
    mlp_dim=2048,
    num_frames=3,
    dropout=0.1,
)

if args.latent_safe:
    wm.load_state_dict(
        torch.load(
            f"dino_wm/checkpoints_latent_safe/class_{args.class_id}_best_classifier.pth"
        ),
        strict=False,
    )
else:
    # wm.load_state_dict(
    #     torch.load(
    #         "dino_wm/checkpoints_sem/encoder_priv_mrg_0.1.pth"
    #     ),
    #     strict=False,
    # )

    load_state_dict_flexible(
        wm,
        "dino_wm/checkpoints_sem/encoder_priv.pth",
    )
# wm.load_state_dict(
#     torch.load(
#         "dino_wm/checkpoints/best_classifier.pth"
#     )
# )
hdf5_file = "path/to/train/dataset/consolidated.h5"
hdf5_file_const = "path/to/constraint/dataset/consolidated.h5"
hdf5_file_test = "path/to/test/dataset/consolidated.h5"
bs = 1
bl = 20
device = "cuda:0"
H = 3
expert_data = SplitTrajectoryDataset(hdf5_file, 3, split="train", num_test=0)
constraint_data = SplitTrajectoryDataset(
    hdf5_file_const, 3, split="train", num_test=0, only_pass_labeled_examples=True
)

expert_loader = iter(DataLoader(expert_data, batch_size=1, shuffle=True))

env = gymnasium.make(
    "franka_wm_DINO-v0",
    params=[wm, expert_data, constraint_data],
    pass_constraint=not args.latent_safe,
    device=device,
)

if args.latent_safe:
    state_shape = env.observation_space.shape or env.observation_space.n
    constraint_shape = None
else:
    state_shape = env.observation_space["state"].shape or env.observation_space.n
    constraint_shape = (
        env.observation_space["constraints"].shape or env.observation_space.n
    )
action_shape = env.action_space.shape or env.action_space.n
max_action = env.action_space.high[0]

train_envs = DummyVectorEnv(
    [
        lambda: gymnasium.make(
            "franka_wm_DINO-v0",
            params=[wm, expert_data, constraint_data],
            pass_constraint=not args.latent_safe,
        )
        for _ in range(1)
    ]
)
test_envs = DummyVectorEnv(
    [
        lambda: gymnasium.make(
            "franka_wm_DINO-v0",
            params=[wm, expert_data, constraint_data],
            pass_constraint=not args.latent_safe,
        )
        for _ in range(1)
    ]
)


# seed
np.random.seed(0)
torch.manual_seed(0)
train_envs.seed(0)
test_envs.seed(0)
# model

actor_activation = torch.nn.ReLU
critic_activation = torch.nn.ReLU


critic_net = Net(
    state_shape=state_shape,
    obs_inputs=["state"] if args.latent_safe else ["state", "constraint"],
    action_shape=action_shape,
    hidden_sizes=[512, 512, 512, 512],
    constraint_dim=512,
    constraint_embedding_dim=512,
    hidden_sizes_constraint=[],
    activation=critic_activation,
    concat=True,
    device=device,
)


critic = Critic(critic_net, device=critic_net.device).to(critic_net.device)
critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3, weight_decay=1e-3)


from PyHJ.policy import avoid_DDPGPolicy_annealing_dinowm as DDPGPolicy

print(
    "DDPG under the Avoid annealed Bellman equation with no Disturbance has been loaded!"
)

actor_net = Net(
    state_shape,
    obs_inputs=["state"] if args.latent_safe else ["state", "constraint"],
    hidden_sizes=[512, 512, 512, 512],
    activation=actor_activation,
    device=device,
    constraint_dim=512,
    constraint_embedding_dim=512,
    hidden_sizes_constraint=[],
)
actor = Actor(actor_net, action_shape, max_action=max_action, device=device).to(device)
actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)

policy = DDPGPolicy(
    critic,
    critic_optim,
    tau=0.005,
    gamma=0.9999,
    exploration_noise=GaussianNoise(sigma=0.1),
    reward_normalization=False,
    estimation_step=1,
    action_space=env.action_space,
    actor=actor,
    actor_optim=actor_optim,
    actor_gradient_steps=1,
)

if args.latent_safe:
    log_path = os.path.join("logs/dinowm/latent_safe/class_{}".format(args.class_id))
else:
    log_path = os.path.join("logs/dinowm")


# collector
train_collector = Collector(
    policy,
    train_envs,
    VectorReplayBuffer(40000, len(train_envs)),
    exploration_noise=True,
)
test_collector = Collector(policy, test_envs)


epoch = 0


def save_best_fn(policy, epoch=epoch):
    if not os.path.exists(log_path + "/epoch_id_{}".format(epoch)):
        print("Just created the log directory!")
        # print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
        os.makedirs(log_path + "/epoch_id_{}".format(epoch))

    if args.latent_safe:
        torch.save(
            policy.state_dict(),
            os.path.join(
                log_path + "/epoch_id_{}".format(epoch),
                f"rotvec_policy_class_{args.class_id}.pth",
            ),
        )
    else:
        torch.save(
            policy.state_dict(),
            os.path.join(
                log_path + "/epoch_id_{}".format(epoch), "rotvec_policy_priv_180.pth"
            ),
        )


def stop_fn(mean_rewards):
    return False


warmup = 1
total_eps = 15
for iter in range(warmup + total_eps):
    if iter < warmup:
        policy._gamma = 0  # for warmup the value fn
        policy.warmup = True
        steps = 10000
    else:
        policy._gamma = 0.95
        policy.warmup = False
        steps = 40000

    print(
        "episodes: {}, remaining episodes: {}".format(iter, warmup + total_eps - iter)
    )
    epoch = epoch + 1
    print("log_path: ", log_path + "/epoch_id_{}".format(epoch))
    if total_eps > 1:
        writer = SummaryWriter(log_path)
    else:
        if not os.path.exists(log_path + "/total_epochs_{}".format(epoch)):
            print("Just created the log directory!")
            print("log_path: ", log_path + "/total_epochs_{}".format(epoch))
            os.makedirs(log_path + "/total_epochs_{}".format(epoch))
        writer = SummaryWriter(log_path)

    if args.latent_safe:
        logger = WandbLogger(
            project="Latent Safe",
            name="Reachability_RL_class_{}".format(args.class_id),
        )
    else:
        logger = WandbLogger(
            project="DINO Reachability", name="sweeper_reachability_RL"
        )
    logger.load(writer)

    # import pdb; pdb.set_trace()
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=1,
        step_per_epoch=steps,  # steps per epoch
        step_per_collect=8,  # step per collect
        episode_per_test=1,  # test num
        batch_size=512,  # batch size
        update_per_step=0.125,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    )

    save_best_fn(policy, epoch=epoch)
    wandb.log(result)
