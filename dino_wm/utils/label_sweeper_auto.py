import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms

# Global variables
current_idx = 0
images = []
labels = {}
current_traj = ""


def crop_top_middle(image):
    top = 30
    left = 28
    height = 192
    width = 192
    return F.crop(image, top, left, height, width)


crop_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Lambda(lambda img: crop_top_middle(img)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

import sys

import torch

# Import custom modules
from torchvision import transforms

print(sys.path)
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")


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


def auto_label(images):
    # [B, T, H, W, C] -> take last frame
    images = images[:].to("cuda")  # keep on GPU

    # images: torch.Tensor of shape (B, H, W, 3), values in [0,1]
    # masks: (B, H, W)
    # print("images shape:", images.shape)
    assert images.min() >= 0 and images.max() <= 1, "Images must be in [0, 1] range"
    masks = isolate_red_chocolates(images)  # Normalize to [0, 1]
    # print("Mask shape:", masks.shape)

    center = get_avg_positions(masks)
    return center


def process_trajectory(traj_file):
    """Load images from a trajectory file and set up labels."""
    global images, labels, current_idx, current_traj

    current_traj = os.path.splitext(os.path.basename(traj_file))[0]
    print(f"Processing trajectory: {current_traj}")
    labels = {}

    # Load images
    images = []
    with h5py.File(traj_file, "r") as hf:
        data = hf["data"]
        assert "camera_1" in data, (
            f"Expected 'camera_1' dataset in the HDF5 file {traj_file}."
        )
        for i in range(data["camera_1"][:].shape[0]):
            front = data["camera_1"][i]

            joint = np.concatenate([front], axis=1)
            joint = crop_transform(joint).permute(1, 2, 0)  # Convert to CxHxW format
            images.append(joint)

    # Initialize index
    current_idx = 0
    while current_idx < len(images) and current_idx in labels:
        current_idx += 1


def postprocess_trajectory(traj_file, labels, label_type):
    """Load images from a trajectory file and set up labels."""

    # write to done_file
    with h5py.File(traj_file, "r+") as hf:
        data_group = hf["data"]

        print(f"Assigning labels to {traj_file}.")
        # labels = np.array(list(labels.values()))
        print(f"Labels: {labels}")
        print(labels.shape)
        print(data_group["camera_1"].shape)
        if label_type in data_group:
            del data_group[label_type]
        data_group.create_dataset(label_type, data=np.array(labels))


# Initialize the plot
plt.ion()

if __name__ == "__main__":
    directory = "/path/to/dataset/"
    label_type = "xy_pos_label"
    reset_regardless_of_label = True
    start_idx = 0
    # Get all pickle files with "unsafe" in the filename
    hdf5_files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if "traj" in filename:
                hdf5_files.append(os.path.join(root, filename))
    # hdf5_files = [f for f in os.listdir(directory) if "traj" in f]
    hdf5_files = sorted(hdf5_files)
    print("total files:", len(hdf5_files))
    # Get the full paths

    tot = len(hdf5_files)
    don = 0

    for idx, traj_file in enumerate(hdf5_files):  # in range(10):
        # done_file = os.path.join(labeled_directory, traj_file)
        traj_file = os.path.join(directory, traj_file)

        if not os.path.exists(traj_file):
            print(f"File {traj_file} not found, skipping.")
            continue

        print(f"Processing {traj_file}...")
        # if "safe" in traj_file and "unsafe" not in traj_file:
        #     process_trajectory_safe(traj_file)
        # else:
        fig, ax = plt.subplots()
        fig.suptitle(f"Trajectory {don + 1}/{tot}")
        plt.subplots_adjust(bottom=0.2)
        fig.text(
            0.5,
            0.05,
            'Press "0" for not divided,\n"1" for divided,\nspace to rewind',
            ha="center",
            fontsize=12,
        )

        process_trajectory(traj_file)
        if images:
            labels = auto_label(torch.stack(images)).cpu().numpy()

        postprocess_trajectory(traj_file, labels, label_type=label_type)
        don += 1
        print(f"Done {don}/{tot}")
        print(f"Finished labeling for {traj_file}.")
