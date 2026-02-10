import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Import custom modules
from dino_wm.dino_models import VideoTransformer, normalize_acs
from dino_wm.test_loader import SplitTrajectoryDataset

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

import torch


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

BL = 4
BS = 1024
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
    nb_classes=3,
).to(device)
# load_state_dict_flexible(transition, "../checkpoints_pa/encoder_0.1.pth")
# load_state_dict_flexible(transition, "../checkpoints/best_testing.pth")

transition.load_state_dict(torch.load("../checkpoints/best_testing.pth"), strict=False)
transition.eval()

hdf5_file = "/home/sunny/data/sweeper/test/consolidated.h5"
train_data = SplitTrajectoryDataset(
    hdf5_file,
    32,
    split="train",
    num_test=0,
    provide_labels=False,
    only_pass_labeled_examples=False,
)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)

coverage_total = torch.zeros((224, 224), dtype=torch.float32, device="cuda")  # if GPU

tot = len(train_data) // BS
for i, data in enumerate(train_loader):
    # [B, T, H, W, C] -> take last frame
    images = data["agentview_image"]  # [data['failure'][:, -1] == 1.0]
    images = images.to("cuda")  # keep on GPU

    import ipdb

    ipdb.set_trace()

    # images: torch.Tensor of shape (B, H, W, 3), values in [0,1]
    # masks: (B, H, W)
    masks = isolate_red_chocolates(images / 255.0)  # Normalize to [0, 1]

    # Convert bool to float and sum over batch
    coverage_total += masks[:, -1].float().sum(dim=0)
    print(f"Processed {i + 1}/{tot} batches", end="\r")

    center = get_avg_positions(masks)

    idx = random.randint(0, images.shape[0] - 1)

    # Save the coverage map
    # plt.figure(figsize=(20, 10))
    # plt.imshow(np.concatenate([images[idx].cpu().numpy() / 255.0, masks_image[idx].cpu().numpy()], axis=1))
    # plt.plot(center[idx, 0].cpu(), center[idx, 1].cpu(), "bx", markersize=12)
    # plt.show()

last_image = images[-1]  # Get the last image for visualization

# Normalize
# coverage_total /= coverage_total.max()

# Move to CPU for plotting
coverage_total = coverage_total.cpu()

# plot the coverage map over the last image
fig, ax = plt.subplots(figsize=(10, 10))
img1 = ax.imshow(last_image.cpu().numpy())
img2 = ax.imshow(coverage_total.cpu().numpy(), vmax=5, alpha=0.5, cmap="viridis")

ax.set_title("Coverage Map Overlay")
ax.axis("off")

# Attach colorbar to the same axes so it matches the height
cbar = fig.colorbar(img2, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Coverage Intensity")
