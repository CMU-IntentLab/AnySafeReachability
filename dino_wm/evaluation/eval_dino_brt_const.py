import os
import random
import sys

import einops
import numpy as np
import torch
from dino_wm.test_loader import SplitTrajectoryDataset
from torch.utils.data import DataLoader

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Import custom modules
import os
import sys

from dino_wm.dino_models import VideoTransformer, normalize_acs, select_xyyaw_from_state
from gymnasium import spaces
from PyHJ.exploration import GaussianNoise
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from torchvision import transforms
from tqdm import *
from tqdm import tqdm
from utils import load_state_dict_flexible

# Add directories to system path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.extend(
    [
        base_dir,
        os.path.join(base_dir, "model_based_irl_torch"),
        os.path.join(base_dir, "real_envs"),
    ]
)

# Load model
print(sys.path)
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")


def transition_from_data(data, transition, device, use_amp=True):
    data1 = data["cam_zed_embd"].to(device)
    data2 = data["cam_rs_embd"].to(device)

    inputs1 = data1[:, :-1]
    inputs2 = data2[:, :-1]

    states = data["state"].to(device)[:, :-1]
    acs = normalize_acs(data["action"].to(device)[:, :-1], device=device)

    with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=use_amp):
        with torch.no_grad():
            pred1, pred2, pred_state, pred_fail, semantic_feat = transition(
                inputs1, inputs2, states, acs
            )

    return pred1, pred2, pred_state, pred_fail, semantic_feat


def data_from_traj(traj):
    data = {}
    segment_length = traj["actions"].shape[0]
    # data["robot0_eye_in_hand_image"] = torch.tensor(
    #     np.array(traj["camera_0"][:]) * 255.0, dtype=torch.uint8
    # )
    data["agentview_image"] = torch.tensor(
        np.array(traj["camera_1"][:]) * 255.0, dtype=torch.uint8
    )
    # data["cam_rs_embd"] = torch.tensor(
    #     np.array(traj["cam_rs_embd"][:]), dtype=torch.float32
    # )
    data["cam_zed_embd"] = torch.tensor(
        np.array(traj["cam_zed_embd"][:]), dtype=torch.float32
    )
    data["state"] = torch.tensor(np.array(traj["states"][:]), dtype=torch.float32)
    data["action"] = torch.tensor(np.array(traj["actions"][:]), dtype=torch.float32)
    if "labels" in traj.keys():
        data["failure"] = torch.tensor(np.array(traj["labels"][:]), dtype=torch.float32)
    data["is_first"] = torch.zeros(segment_length)
    data["is_last"] = torch.zeros(segment_length)
    data["is_first"][0] = 1.0
    data["is_terminal"] = data["is_last"]
    data["discount"] = torch.ones(segment_length, dtype=torch.float32)
    return data


def evaluate_V(policy, latent, constraint, action, device):
    constraint = einops.repeat(
        torch.concat((constraint, torch.tensor([1.0], device=device))),
        "C -> B C",
        B=1,
    )  # 1 indicates constraint is active

    obs = {
        "state": latent[:, [-1]].mean(dim=2).reshape(1, -1),
        "constraints": constraint,
    }
    return (
        policy.critic(
            obs=obs,
            act=normalize_acs(
                action.to(device).unsqueeze(0)  # Next action
            ),
        )
        .detach()
        .squeeze()
        .cpu()
        .numpy()
    )


def confusion(pred, gt_labels, threshold=0.0):
    safe_data = torch.where(gt_labels != 0)
    unsafe_data = torch.where(gt_labels == 0)

    pos = pred[safe_data]
    neg = pred[unsafe_data]

    TP = torch.sum(pos > threshold).item()
    FN = torch.sum(pos < threshold).item()
    FP = torch.sum(neg > threshold).item()
    TN = torch.sum(neg < threshold).item()

    return torch.tensor([TP, FN, FP, TN])


# Define transforms
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform1 = transforms.Compose(
    [
        transforms.Resize(520),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.2),
    ]
)

DINO_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
    ]
)

norm_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = "cuda:0"

hdf5_file = "/home/sunny/data/sweeper/test/consolidated.h5"
test_dataset = SplitTrajectoryDataset(
    hdf5_file,
    3,
    split="train",
    num_test=0,
    only_pass_labeled_examples=True,
)
test_data_loader = iter(DataLoader(test_dataset, batch_size=16, shuffle=True))

constraint_dataset = SplitTrajectoryDataset(
    "/home/sunny/data/sweeper/test/consolidated.h5",
    3,
    split="train",
    num_test=0,
    only_pass_labeled_examples=True,
)
const_data_loader = iter(DataLoader(constraint_dataset, batch_size=16, shuffle=True))

BL = 4
transition = VideoTransformer(
    image_size=(224, 224),
    dim=384,
    ac_dim=10,
    state_dim=3,
    depth=6,
    heads=16,
    mlp_dim=2048,
    num_frames=BL - 1,
    dropout=0.1,
).to(device)
load_state_dict_flexible(
    transition, "../checkpoints_pa/encoder_mrg_0.1_alpha_32_num_ex_all_ul_F.pth"
)
# load_state_dict_flexible(transition, "../checkpoints/best_testing.pth")

# transition.load_state_dict(torch.load("../checkpoints/best_classifier.pth"))
# transition.load_state_dict(torch.load("../checkpoints/best_multi_classifier.pth"))
transition.eval()

actor_activation = torch.nn.ReLU
critic_activation = torch.nn.ReLU

critic_net = Net(
    state_shape=(397,),
    obs_inputs=["state", "constraint"],
    action_shape=(3,),
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
    (397,),
    obs_inputs=["state", "constraint"],
    hidden_sizes=[512, 512, 512, 512],
    activation=actor_activation,
    device=device,
    constraint_dim=512,
    constraint_embedding_dim=512,
    hidden_sizes_constraint=[],
)
actor = Actor(actor_net, (3,), max_action=1, device=device).to(device)
actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)

policy = DDPGPolicy(
    critic,
    critic_optim,
    tau=0.005,
    gamma=0.9999,
    exploration_noise=GaussianNoise(sigma=0.1),
    reward_normalization=False,
    estimation_step=1,
    action_space=spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
    actor=actor,
    actor_optim=actor_optim,
    actor_gradient_steps=1,
)
policy.load_state_dict(
    torch.load(
        "/home/sunny/AnySafe_Reachability/scripts/logs/dinowm/epoch_id_16/rotvec_policy.pth"
    )
)

num_iter = 1000

confusion_matrices = {}
for _class in range(3):
    confusion_matrices[f"class_{_class}_lz"] = torch.zeros(4)
    confusion_matrices[f"class_{_class}_value_fn"] = torch.zeros(4)

for iter_idx in tqdm(range(num_iter), desc="Calculating Accuracy", position=0):
    test_data = next(test_data_loader)
    const_data = next(const_data_loader)
    if len(test_data["cam_zed_embd"]) != 16:
        test_data_loader = iter(DataLoader(test_dataset, batch_size=16, shuffle=True))
        test_data = next(test_data_loader)
    if len(const_data["cam_zed_embd"]) != 16:
        const_data_loader = iter(
            DataLoader(constraint_dataset, batch_size=16, shuffle=True)
        )
        const_data = next(const_data_loader)
    mask = const_data["failure"][:, -1] != -1.0

    # [16, 512]
    constraints = transition.semantic_embed(
        inp1=const_data["cam_zed_embd"][:, -1:].to(device),
        state=select_xyyaw_from_state(const_data["state"][:, -1:]).to(device),
    ).squeeze()

    # latent: [B T 256 397]
    pred1, pred_state, pred_split, semantic_features, latent = transition(
        test_data["cam_zed_embd"].to(device),
        select_xyyaw_from_state(test_data["state"].to(device)),
        normalize_acs(test_data["action"].to(device)),
        return_latent=True,
    )

    obs = {
        "state": latent[:, -1].mean(dim=-2),
        "constraints": torch.concat(
            [
                constraints,
                torch.ones_like(constraints[:, -1], device=device).unsqueeze(1),
            ],
            dim=1,
        ),
    }

    # value_fn [16]
    value_fn = (
        policy.critic(
            obs=obs,
            act=normalize_acs(
                test_data["action"][:, [-1]].to(device)  # Next action
            ),
        )
        .detach()
        .squeeze()
    )

    value_fn = value_fn[mask]

    # [16, 512]
    semantic_features_test = transition.semantic_embed(
        inp1=test_data["cam_zed_embd"][:, -1:].to(device),
        state=select_xyyaw_from_state(test_data["state"][:, -1:]).to(device),
    ).squeeze()

    # [16, 512]
    semantic_features_const = transition.semantic_embed(
        inp1=const_data["cam_zed_embd"][:, -1:].to(device),
        state=select_xyyaw_from_state(const_data["state"][:, -1:]).to(device),
    ).squeeze()

    cos_sim = torch.nn.CosineSimilarity(dim=1)
    # [16]
    lz = -torch.tanh(2 * cos_sim(semantic_features_test, semantic_features_const))
    lz = lz[mask]

    for _class in range(3):
        class_mask = const_data["failure"][:, -1][mask] == (_class + 1)
        # represent safe and unsafe
        gt_labels = (
            test_data["failure"][:, -1][mask][class_mask] != (_class + 1)
        ).float()
        # TP, FN, FP, TN
        confusion_matrices[f"class_{_class}_lz"] += confusion(
            lz[class_mask], gt_labels, threshold=transition.thresholds[_class]
        )
        confusion_matrices[f"class_{_class}_value_fn"] += confusion(
            value_fn[class_mask], gt_labels, threshold=transition.thresholds[_class]
        )

    torch.set_printoptions(precision=0, sci_mode=False)  # no scientific notation
    for k, v in confusion_matrices.items():
        print(f"{k}")
        print(f"{v.reshape(2, 2).T}")
