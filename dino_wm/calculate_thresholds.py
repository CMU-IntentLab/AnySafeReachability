

import os
import random
import sys

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Import custom modules
from dino_wm.dino_models import VideoTransformer, normalize_acs, select_xyyaw_from_state
from dino_wm.test_loader import SplitTrajectoryDataset

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Import custom modules
import os
import sys

from gymnasium import spaces
from tqdm import *

from proxy_anchor.utils import load_state_dict_flexible
from PyHJ.exploration import GaussianNoise
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic

print(sys.path)
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")


def evaluate_V(policy, latent, constraint, action, device):
    # constraint: [B (C + 1)]
    constraint = einops.repeat(
        torch.concat((constraint, torch.tensor([1.0], device=device))),
        "C -> B C",
        B=latent.shape[0],
    )  # 1 indicates constraint is active

    obs = {
        "state": latent[:, [-1]].mean(dim=2).squeeze(),  # [B, 397]
        "constraints": constraint,  # [B, C + 1]
    }
    return (
        policy.critic(
            obs=obs,
            act=normalize_acs(
                action.to(device)  # Next action
            ),
        )
        .detach()
        .squeeze()
        .cpu()
        .numpy()
    )


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


def isolate_red_chocolates(images):
    """
    Isolate red chocolates from batched images.

    Args:
        images (torch.Tensor): Batched images of shape (B, H, W, 3), values in [0, 1]

    Returns:
        masks (torch.Tensor): Binary masks of shape (B, 1, H, W)
    """
    images = images.permute(0, 3, 1, 2)  # Change to (B, C, H, W)
    B, C, H, W = images.shape
    assert C == 3, "Images must have 3 channels (RGB)"

    # Convert RGB to HSV (vectorized)
    r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:3]
    maxc = torch.max(images, dim=1, keepdim=True)[0]
    minc = torch.min(images, dim=1, keepdim=True)[0]
    v = maxc
    s = (maxc - minc) / (maxc + 1e-6)

    # Hue calculation
    h = torch.zeros_like(maxc)
    mask = (maxc == r) & (maxc != minc)
    h[mask] = (60 * ((g - b) / (maxc - minc + 1e-6)))[mask]
    mask = (maxc == g) & (maxc != minc)
    h[mask] = (60 * ((b - r) / (maxc - minc + 1e-6) + 2))[mask]
    mask = (maxc == b) & (maxc != minc)
    h[mask] = (60 * ((r - g) / (maxc - minc + 1e-6) + 4))[mask]
    h = (h % 360) / 360.0  # normalize to [0,1]

    # Define red range in HSV
    red_mask = ((h < 0.05) | (h > 0.95)) & (s > 0.5) & (v > 0.2)

    return red_mask.float().squeeze()


# --- Average positions ---
def get_avg_positions(mask):
    B, H, W = mask.shape
    ys = torch.arange(H, device=mask.device).view(1, H, 1).expand(B, H, W)
    xs = torch.arange(W, device=mask.device).view(1, 1, W).expand(B, H, W)

    counts = mask.sum(dim=(1, 2)).clamp(min=1)
    avg_x = (xs * mask).sum(dim=(1, 2)) / counts
    avg_y = (ys * mask).sum(dim=(1, 2)) / counts

    return torch.stack([avg_x, avg_y], dim=1)


use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = "cuda:0"

x_class_boundaries = [0, 224 // 3, 224 * 2 // 3, 224]  # x boundaries for 3 classes
y_class_boundaries = [224 // 3, 224 * 2 // 3, 224]  # y boundaries for 3 classes
# 3 * 2 = 6 classes in total
nb_classes = (len(x_class_boundaries) - 1) * (len(y_class_boundaries) - 1)
label_to_str = {
    0: "Left Top",
    1: "Left Bottom",
    2: "Middle Top",
    3: "Middle Bottom",
    4: "Right Top",
    5: "Right Bottom",
}


def get_class_from_xy(labels):
    assert labels.shape[-1] == 2, "Labels should have shape (B, 2)"
    x_labels = torch.bucketize(
        labels[..., 0], torch.tensor(x_class_boundaries, device=device)
    ).unsqueeze(1)
    y_labels = torch.bucketize(
        labels[..., 1], torch.tensor(y_class_boundaries, device=device)
    ).unsqueeze(1)

    class_labels = (x_labels - 1) * (len(y_class_boundaries) - 1) + (y_labels - 1)
    class_labels[torch.logical_or(x_labels <= 0, y_labels <= 0)] = -1
    class_labels[
        torch.logical_or(
            labels[..., 0] < x_class_boundaries[0],
            labels[..., 0] >= x_class_boundaries[-1],
        )
    ] = -1
    class_labels[
        torch.logical_or(
            labels[..., 1] < y_class_boundaries[0],
            labels[..., 1] >= y_class_boundaries[-1],
        )
    ] = -1

    return class_labels


BL = 4
BS = 16
open_loop = True
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
    nb_classes=nb_classes,
).to(device)
load_state_dict_flexible(
    transition,
    "checkpoints_pa/encoder_mrg_0.1_alpha_32_bound_2x3.pth",
)
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
        "/home/sunny/AnySafe_Reachability/scripts/logs/dinowm/epoch_id_16/rotvec_policy_prox.pth"
    )
)

hdf5_file = "/home/sunny/data/sweeper/test/consolidated.h5"
hdf5_file_const = "/home/sunny/data/sweeper/train/consolidated.h5"
train_data = SplitTrajectoryDataset(
    hdf5_file,
    BL,
    split="train",
    num_test=0,
    provide_labels=True,
    only_pass_labeled_examples=True,
)
train_loader = DataLoader(train_data, batch_size=BS, shuffle=True, num_workers=4)

_class = 0
label_y = _class % (len(y_class_boundaries) - 1)
label_x = _class // (len(y_class_boundaries) - 1)


tot = len(train_data) // BS
max_batches = 200
tot = min(tot, max_batches)
lz_per_class = {_class: [] for _class in range(nb_classes)}
for i, data in tqdm(enumerate(train_loader), total=tot):
    if i >= max_batches:
        break
    __, __, __, __, latent = transition(
        data["cam_zed_embd"][:, :-1].to(device),
        select_xyyaw_from_state(data["state"][:, :-1]).to(device),
        normalize_acs(data["action"][:, :-1].to(device)),
        return_latent=True,
    )
    # V: [B] value of last frame
    for _class in range(nb_classes):
        constraint = transition.proxies[_class].detach()

        # V = evaluate_V(
        #     policy=policy,
        #     latent=latent,
        #     constraint=constraint,
        #     action=data["action"][:, -1],
        #     device=device,
        # )
        semantic_features = transition.semantic_embed(
            inp1=data["cam_zed_embd"].to(device),
            state=select_xyyaw_from_state(data["state"]).to(device),
        ).detach()
        lz = -np.tanh(
            2
            * F.cosine_similarity(
                semantic_features[:, -1],
                einops.repeat(
                    constraint,
                    "c -> b c",
                    b=semantic_features.shape[0],
                ),
            )
            .cpu()
            .numpy()
        )
        mask = (
            (get_class_from_xy(data["failure"][:, -1].to(device)) == _class)
            .squeeze()
            .cpu()
            .numpy()
        )
        lz_per_class[_class].append(lz[mask])

for _class in range(nb_classes):
    lz_per_class[_class] = np.concatenate(lz_per_class[_class])
    for p in [75, 90, 95, 99]:
        print(
            f"Class {_class} ({label_to_str[_class]}) ({p}%): {np.percentile(lz_per_class[_class], p)}"
        )
