import argparse
import os

import gymnasium as gym

# NOTE: all the reach-avoid gym environments are in reach_rl_gym, the constraint information is output as an element of the info dictionary in gym.step() function
import matplotlib.pyplot as plt
import numpy as np
import torch

import wandb
from PyHJ.env import DummyVectorEnv
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import ActorProb, Critic

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))


"""
    Note that, we can pass arguments to the script by using
    For learning our new reach-avoid value function:
    python run_training_sac.py --task ra_droneracing_Game-v6 --control-net 512 512 512 512 --disturbance-net 512 512 512 512 --critic-net 512 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9
    python run_training_sac.py --task ra_highway_Game-v2 --control-net 512 512 512 --disturbance-net 512 512 512 --critic-net 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9
    python run_training_sac.py --task ra_1d_Game-v0 --control-net 32 32 --disturbance-net 4 4 --critic-net 4 4 --epoch 10 --total-episodes 160 --gamma 0.9
    
    For learning the classical reach-avoid value function (baseline):
    python run_training_sac.py --task ra_droneracing_Game-v6 --control-net 512 512 512 512 --disturbance-net 512 512 512 512 --critic-net 512 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9 --is-game-baseline True
    python run_training_sac.py --task ra_highway_Game-v2 --control-net 512 512 512 --disturbance-net 512 512 512 --critic-net 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9 --is-game-baseline True
    python run_training_sac.py --task ra_1d_Game-v0 --control-net 32 32 --disturbance-net 4 4 --critic-net 4 4 --epoch 10 --total-episodes 160 --gamma 0.9 --is-game-baseline True
    
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="dubins-v0"
    )  # ra_droneracing_Game-v6, ra_highway_Game-v2, ra_1d_Game-v0
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=40000)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.0)
    parser.add_argument("--total-episodes", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=40000)
    parser.add_argument("--step-per-collect", type=int, default=8)
    parser.add_argument("--update-per-step", type=float, default=0.125)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--control-net", type=int, nargs="*", default=None
    )  # for control policy
    parser.add_argument(
        "--disturbance-net", type=int, nargs="*", default=None
    )  # for disturbance policy
    parser.add_argument(
        "--critic-net", type=int, nargs="*", default=None
    )  # for critic net
    parser.add_argument(
        "--constraint-embedding-dim", type=int, default=3
    )  # for constraint embedding
    parser.add_argument(
        "--control-net-const", type=int, nargs="*", default=(128, 128)
    )  # for control net constraint
    parser.add_argument("--training-num", type=int, default=8)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--rew-norm", action="store_true", default=False)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--continue-training-logdir", type=str, default=None)
    parser.add_argument("--continue-training-epoch", type=int, default=None)
    parser.add_argument("--actor-gradient-steps", type=int, default=1)
    parser.add_argument(
        "--is-game-baseline", type=bool, default=False
    )  # it will be set automatically
    parser.add_argument(
        "--is-game", type=bool, default=False
    )  # it will be set automatically
    parser.add_argument("--target-update-freq", type=int, default=400)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--actor-activation", type=str, default="ReLU")
    parser.add_argument("--critic-activation", type=str, default="ReLU")
    parser.add_argument("--kwargs", type=str, default="{}")
    parser.add_argument("--warm-start-path", type=str, default=None)
    # new sac params:
    parser.add_argument("--alpha", default=0.2)
    parser.add_argument("--auto-alpha", type=int, default=1)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--env-dist-type", type=str, default="uni")
    parser.add_argument(
        "--nominal-policy", type=str, default="random"
    )  # Only used for eval
    args = parser.parse_known_args()[0]
    return args


args = get_args()

env_kwargs = {"dist_type": args.env_dist_type, "nominal_policy": args.nominal_policy}

env = gym.make(args.task, **env_kwargs)
# check if the environment has control and disturbance actions:
assert hasattr(env, "action_space")
if isinstance(env.observation_space, gym.spaces.Dict):
    args.state_shape = (
        env.observation_space["state"].shape or env.observation_space["state"].n
    )
else:
    args.state_shape = env.observation_space.shape or env.observation_space.n
args.constraint_dim = env.constraint_shape
args.action_shape = env.action_space.shape or env.action_space.n
args.max_action = env.action_space.high[0]

args.action1_shape = env.action_space.shape or env.action_space.n
args.max_action1 = env.action_space.high[0]


train_envs = DummyVectorEnv(
    [lambda: gym.make(args.task, **env_kwargs) for _ in range(args.training_num)]
)
test_envs = DummyVectorEnv(
    [lambda: gym.make(args.task, **env_kwargs) for _ in range(args.test_num)]
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
        obs_inputs=["state", "constraint"],
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

# critic = Critic(critic_net, device=args.device).to(args.device)
# critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

critic1 = Critic(critic_net, device=args.device).to(args.device)
critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
critic2 = Critic(critic_net, device=args.device).to(args.device)
critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
# import pdb; pdb.set_trace()
log_path = None

from PyHJ.policy import avoid_SACPolicy_annealing as SACPolicy

print("SAC under the Avoid annealed Bellman equation has been loaded!")

actor1_net = Net(
    args.state_shape,
    obs_inputs=["state", "constraint"],
    hidden_sizes=args.control_net,
    activation=actor_activation,
    device=args.device,
    constraint_dim=args.constraint_dim,
    constraint_embedding_dim=args.constraint_embedding_dim,
    hidden_sizes_constraint=args.control_net_const,
)
actor1 = ActorProb(actor1_net, args.action1_shape, device=args.device).to(args.device)
actor1_optim = torch.optim.Adam(actor1.parameters(), lr=args.actor_lr)

if args.auto_alpha:
    target_entropy = -np.prod(env.action_space.shape)
    log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
    alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
    args.alpha = (target_entropy, log_alpha, alpha_optim)

policy = SACPolicy(
    critic1,
    critic1_optim,
    critic2,
    critic2_optim,
    tau=args.tau,
    gamma=args.gamma,
    alpha=args.alpha,
    exploration_noise=None,  # GaussianNoise(sigma=args.exploration_noise), # careful!
    deterministic_eval=True,
    estimation_step=args.n_step,
    action_space=env.action_space,
    actor1=actor1,
    actor1_optim=actor1_optim,
)

log_path = os.path.join(
    args.logdir,
    args.task,
    f"baseline_sac_avoid_env_dist_{args.env_dist_type}_training_num_{args.training_num}_buffer_size_{args.buffer_size}_gamma_{args.gamma}",
)


if args.warm_start_path is not None:
    policy.load_state_dict(torch.load(args.warm_start_path))
    args.kwargs = args.kwargs + "warmstarted"

# epoch = 0
# writer = SummaryWriter(log_path, filename_suffix="_"+timestr+"epoch_id_{}".format(epoch))
# logger = TensorboardLogger(writer)
log_path = (
    log_path
    + "/noise_{}_actor_lr_{}_critic_lr_{}_batch_{}_step_per_epoch_{}_kwargs_{}_seed_{}".format(
        args.exploration_noise,
        args.actor_lr,
        args.critic_lr,
        args.batch_size,
        args.step_per_epoch,
        args.kwargs,
        args.seed,
    )
)

import ipdb

ipdb.set_trace()

epoch_id = 8
save_path = os.path.join(log_path + "/epoch_id_{}".format(epoch_id), "policy.pth")
policy.load_state_dict(torch.load(save_path))
policy.eval()

if args.continue_training_epoch is not None:
    epoch = args.continue_training_epoch
    policy.load_state_dict(
        torch.load(os.path.join(log_path + "/epoch_id_{}".format(epoch), "policy.pth"))
    )


if args.continue_training_logdir is not None:
    policy.load_state_dict(torch.load(args.continue_training_logdir))
    # epoch = int(args.continue_training_logdir.split('_')[-9].split('_')[0])
    epoch = args.continue_training_epoch


def stop_fn(mean_rewards):
    return False


if not os.path.exists(log_path + "/epoch_id_{}".format(epoch)):
    print("Just created the log directory!")
    # print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
    os.makedirs(log_path + "/epoch_id_{}".format(epoch))

if args.continue_training_epoch is not None:
    print(
        "episodes: {}, remaining episodes: {}".format(
            epoch // args.epoch, args.total_episodes - iter
        )
    )
else:
    print(
        "episodes: {}, remaining episodes: {}".format(iter, args.total_episodes - iter)
    )
epoch = epoch + args.epoch
print("log_path: ", log_path + "/epoch_id_{}".format(epoch))

in_dist_plots = env.get_eval_plot(policy, policy.critic1, in_distribution=True)
plot1, plot2 = in_dist_plots[0], in_dist_plots[1]
wandb.log(
    {
        "in_dist/binary_reach_avoid_plot": wandb.Image(plot1),
        "in_dist/continuous_plot": wandb.Image(plot2),
    }
)
if len(in_dist_plots) == 3:
    metrics = in_dist_plots[2]
    metrics = {"in_dist/" + k: v for k, v in metrics.items()}
    wandb.log(metrics)

out_dist_plots = env.get_eval_plot(policy, policy.critic1, in_distribution=False)
plot1, plot2 = out_dist_plots[0], out_dist_plots[1]
wandb.log(
    {
        "out_dist/binary_reach_avoid_plot": wandb.Image(plot1),
        "out_dist/continuous_plot": wandb.Image(plot2),
    }
)
if len(out_dist_plots) == 3:
    metrics = out_dist_plots[2]
    metrics = {"out_dist/" + k: v for k, v in metrics.items()}
    wandb.log(metrics)

in_dist_imgs = env.generate_trajectory(policy=policy, in_distribution=True)
wandb.log({"in_dist/trajectory": [wandb.Video(in_dist_imgs, fps=10, format="mp4")]})

out_dist_imgs = env.generate_trajectory(policy=policy, in_distribution=False)
wandb.log({"out_dist/trajectory": [wandb.Video(out_dist_imgs, fps=10, format="mp4")]})

success_rate = env.get_success_rate(policy, in_distribution=False)
wandb.log({"out_dist/success_rate": success_rate})

success_rate = env.get_success_rate(policy, in_distribution=True)
wandb.log({"in_dist/success_rate": success_rate})

import ipdb

ipdb.set_trace()

plt.close()
