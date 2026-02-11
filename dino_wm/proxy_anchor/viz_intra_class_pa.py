import os
import random
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Import custom modules
from dino_wm.dino_models import VideoTransformer, normalize_acs
from dino_wm.utils import load_state_dict_flexible
from torchvision import transforms

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
        np.array(traj["camera_1"][:]) * 255.0, dtype=torch.uint8, device=device
    )
    # data["cam_rs_embd"] = torch.tensor(
    #     np.array(traj["cam_rs_embd"][:]), dtype=torch.float32
    # )
    data["cam_zed_embd"] = torch.tensor(
        np.array(traj["cam_zed_embd"][:]), dtype=torch.float32, device=device
    )
    data["state"] = torch.tensor(
        np.array(traj["states"][:]), dtype=torch.float32, device=device
    )
    data["action"] = torch.tensor(
        np.array(traj["actions"][:]), dtype=torch.float32, device=device
    )
    if "labels" in traj.keys():
        data["failure"] = torch.tensor(
            np.array(traj["labels"][:]), dtype=torch.float32, device=device
        )
    data["is_first"] = torch.zeros(segment_length, dtype=torch.float32, device=device)
    data["is_last"] = torch.zeros(segment_length, dtype=torch.float32, device=device)
    data["is_first"][0] = 1.0
    data["is_terminal"] = data["is_last"]
    data["discount"] = torch.ones(segment_length, dtype=torch.float32, device=device)
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

if __name__ == "__main__":
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = "cuda:0"

    hdf5_file = "/home/sunny/data/sweeper/test/consolidated.h5"
    database = {}
    with h5py.File(hdf5_file, "r") as hf:
        trajectory_ids = list(hf.keys())
        database = {
            i: data_from_traj(hf[traj_id]) for i, traj_id in enumerate(trajectory_ids)
        }

    BL = 4
    transition = VideoTransformer(
        image_size=(224, 224),
        dim=384,
        ac_dim=10,
        state_dim=8,
        depth=6,
        heads=16,
        mlp_dim=2048,
        num_frames=BL - 1,
        dropout=0.1,
    ).to(device)
    # load_state_dict_flexible(transition, "../checkpoints_pa/encoder_0.1.pth")
    load_state_dict_flexible(
        transition, "../checkpoints_pa/encoder_mrg_0.1_alpha_32_num_ex_all_ul_F.pth"
    )

    # transition.load_state_dict(torch.load("../checkpoints/best_classifier.pth"))
    transition.eval()

    for key, data in database.items():
        inputs1 = (  # [1, T, 256, 384]
            data["cam_zed_embd"][:, :].to(device).unsqueeze(0)
        )
        # acs = data["action"][t, :].to(device).unsqueeze(0)
        # acs = normalize_acs(acs, device=device)
        states = data["state"][:, :].to(device).unsqueeze(0)  # [1, T, 8]

        semantic_feature = transition.semantic_embed(  # [1, T, embedding_dim]
            inp1=inputs1, state=states
        ).squeeze()

        # Update the database with semantic features
        database[key].update({"semantic_feat": semantic_feature})

    constraint1 = {
        "front": database[5]["agentview_image"][300],
        "semantic_feat": database[5]["semantic_feat"][300],
        "database_id": 5,
        "timestep": 300,
    }  # weak unsafe frame
    constraint2 = {
        "front": database[0]["agentview_image"][200],
        "semantic_feat": database[0]["semantic_feat"][200],
        "database_id": 0,
        "timestep": 200,
    }  # unsafe frame
    constraint3 = {
        "front": database[1]["agentview_image"][300],
        "semantic_feat": database[1]["semantic_feat"][300],
        "database_id": 1,
        "timestep": 300,
    }

    scale = 1.0

    for const_num, constraint in enumerate([constraint1, constraint2, constraint3]):
        range_bins = np.linspace(-1.0, 1.0, 11)
        bins_high = range_bins[1:]
        bins_low = range_bins[:-1]
        bin_widths = bins_high - bins_low
        binned_values = {i: [] for i in range(len(bins_low))}

        feat_c = constraint["semantic_feat"]
        for key, data in database.items():
            if key == constraint["database_id"]:
                continue  # Skip the constraint frame itself
            feat = data["semantic_feat"]
            cos_sim = F.cosine_similarity(feat, feat_c, dim=-1)
            for i in range(len(bins_low)):
                mask = (
                    (cos_sim >= bins_low[i])
                    & (cos_sim < bins_high[i])
                    & (data["failure"] != -1)
                )
                if mask.any():
                    binned_values[i].append(
                        {
                            "image": data["agentview_image"][mask].cpu().numpy(),
                            "label": data["failure"][mask].cpu().numpy(),
                        }
                    )

        # Convert lists to numpy arrays
        for i in range(len(bins_low)):
            if binned_values[i]:
                binned_values[i] = {
                    "image": np.concatenate(
                        [item["image"] for item in binned_values[i]], axis=0
                    ),
                    "label": np.concatenate(
                        [item["label"] for item in binned_values[i]], axis=0
                    ),
                }
            else:
                binned_values[i] = {"image": np.array([]), "label": np.array([])}

        label_to_color = {
            1: "red",
            2: "green",
            3: "blue",
        }
        # Plot the images
        num_cols = np.sum(
            [binned_values[i]["image"].size > 0 for i in range(len(bins_low))]
        )
        fig, axes = plt.subplots(
            nrows=4 + 1,
            ncols=num_cols,
            figsize=(1.5 * num_cols, 6),
            constrained_layout=True,
        )

        cols = [i for i in range(len(bins_low)) if binned_values[i]["image"].size > 0]
        middle_col = len(cols) // 2

        axes[0, middle_col].imshow(constraint["front"].cpu().numpy())
        for ax in axes[0, :]:
            ax.axis("off")
        rect = patches.Rectangle(
            (0, 0),  # (x,y) bottom left corner
            constraint["front"].cpu().numpy().shape[1],  # width
            constraint["front"].cpu().numpy().shape[0],  # height
            linewidth=2,  # thickness of the outline
            edgecolor=label_to_color[const_num + 1],  # color of the outline
            facecolor="none",  # no fill
        )
        axes[0, middle_col].add_patch(rect)
        axes[0, middle_col].set_title("Constraint Frame")

        for i, ax_row in enumerate(axes[1:]):
            for j, ax in zip(cols, ax_row):
                if binned_values[j]["image"].size > 0:
                    # select random image from the bin
                    idx = random.randint(0, len(binned_values[j]["image"]) - 1)

                    img = binned_values[j]["image"][idx]
                    label = binned_values[j]["label"][idx]

                    # img = img.transpose(1, 2, 0)
                    ax.imshow(img)

                    # Add outline based on label
                    rect = patches.Rectangle(
                        (0, 0),  # (x,y) bottom left corner
                        img.shape[1],  # width
                        img.shape[0],  # height
                        linewidth=2,  # thickness of the outline
                        edgecolor=label_to_color[label],  # color of the outline
                        facecolor="none",  # no fill
                    )
                    ax.add_patch(rect)

                else:
                    ax.axis("off")
                ax.axis("off")
                if i == 0:
                    ax.set_title(
                        rf"$\epsilon$ = {bins_low[j]:.2f} - {bins_high[j]:.2f}"
                    )

        fig.text(
            0.01,
            0.95,
            "Left Region = Red",
            ha="left",
            va="top",
            fontsize=15,
            color="red",
        )
        fig.text(
            0.01,
            0.91,
            "Middle Region = Green",
            ha="left",
            va="top",
            fontsize=15,
            color="green",
        )
        fig.text(
            0.01,
            0.87,
            "Right Region = Blue",
            ha="left",
            va="top",
            fontsize=15,
            color="blue",
        )

        plt.subplots_adjust(
            bottom=0.2
        )  # Increase space at the bottom (default is ~0.11)

        plt.suptitle(
            f"Cosine Similarity Images for Region {const_num + 1}",
            fontsize=16,
        )
        plt.savefig(f"cosine_similarity_images_region_{const_num + 1}.png")
