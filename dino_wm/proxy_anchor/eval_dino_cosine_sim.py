import copy
import os
import random
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader
from torchvision import transforms
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

# Import custom modules
from dino_wm.dino_decoder import VQVAE
from dino_wm.dino_models import VideoTransformer, normalize_acs
from dino_wm.test_loader import SplitTrajectoryDataset


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

    BS, BL = 100, 4
    hdf5_file = "/home/sunny/data/skittles/vlog-test-labeled/consolidated.h5"
    expert_data = SplitTrajectoryDataset(hdf5_file, BL, split="train", num_test=0)

    # hdf5_file = "/home/sunny/data/skittles/consolidated.h5"
    # expert_data = SplitTrajectoryDataset(hdf5_file, BL, split="test", num_test=10)
    expert_loader = iter(
        DataLoader(expert_data, batch_size=BS, shuffle=True, num_workers=4)
    )

    device = "cuda:0"

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
    load_state_dict_flexible(
        transition, "../checkpoints_pa/encoder_mrg_0.1_num_ex_20.pth"
    )
    transition.eval()

    decoder = VQVAE().to(device)
    decoder.load_state_dict(torch.load("../checkpoints/testing_decoder.pth"))
    decoder.eval()

    # data = next(expert_loader)
    # pred1, pred2, pred_state, pred_fail, semantic_feat = transition_from_data(
    #     data, transition, device
    # )

    def init_nested_dict():
        # return {
        #     "safe": {"safe": [], "unsafe": [], "weak_unsafe": []},
        #     "unsafe": {"unsafe": [], "weak_unsafe": []},
        #     "weak_unsafe": {"weak_unsafe": []},
        # }
        return {
            "safe": {"safe": [], "unsafe": []},
            "unsafe": {"unsafe": []},
        }

    cosine_similarities_cam_1 = init_nested_dict()
    cosine_similarities_cam_2 = copy.deepcopy(cosine_similarities_cam_1)
    euc_distances_cam_1 = copy.deepcopy(cosine_similarities_cam_1)
    euc_distances_cam_2 = copy.deepcopy(cosine_similarities_cam_1)

    cosine_similarities_sem = init_nested_dict()
    cosine_similarities_sem_max_difference = {"safe": {"unsafe": []}}
    closest_diff_class = {
        "cos_sim": -float("inf"),
        "wrist_image1": None,
        "front_image1": None,
        "wrist_image2": None,
        "front_image2": None,
    }
    furthest_same_class = copy.deepcopy(closest_diff_class)
    furthest_same_class["cos_sim"] = float("inf")

    failure_modes = {
        "safe_safe": copy.deepcopy(furthest_same_class),
        "unsafe_unsafe": copy.deepcopy(furthest_same_class),
        "safe_unsafe": copy.deepcopy(closest_diff_class),
    }

    dataframe = {
        "embed_cam_1": [],
        "embed_cam_2": [],
        "semantic_feat": [],
        "failure": [],
        "robot0_eye_in_hand_image": [],
        "agentview_image": [],
    }

    for _ in tqdm(range(len(expert_loader) - 1), desc="Creating dataframe"):
        data = next(expert_loader)
        __, __, ___, __, semantic_feat = transition_from_data(data, transition, device)
        embed_cam_1 = data["cam_zed_embd"].to(device)
        embed_cam_2 = data["cam_rs_embd"].to(device)

        semantic_data = semantic_feat.to(device)

        dataframe["embed_cam_1"].append(embed_cam_1[:, -1].cpu())
        dataframe["embed_cam_2"].append(embed_cam_2[:, -1].cpu())
        dataframe["semantic_feat"].append(semantic_data[:, -1].cpu())
        dataframe["failure"].append(data["failure"][:, -1].cpu())
        dataframe["robot0_eye_in_hand_image"].append(
            data["robot0_eye_in_hand_image"][:, -1].cpu()
        )
        dataframe["agentview_image"].append(data["agentview_image"][:, -1].cpu())

    for k, v in dataframe.items():
        dataframe[k] = torch.cat(v, dim=0)

    masks = {
        "safe": dataframe["failure"][:] == 0.0,
        "unsafe": (dataframe["failure"][:] == 1.0) | (dataframe["failure"][:] == 2.0),
        # "weak_unsafe": data["failure"][:, 0] == 2.0,
    }
    num_safe = masks["safe"].sum().item()
    num_unsafe = masks["unsafe"].sum().item()
    total = num_safe + num_unsafe
    expected_total_pairs = (total * (total - 1)) // 2
    print("Safe:", num_safe, "Unsafe:", num_unsafe, "Total:", num_safe + num_unsafe)

    embed_cam_1_class = {k: dataframe["embed_cam_1"][m] for k, m in masks.items()}
    embed_cam_2_class = {k: dataframe["embed_cam_2"][m] for k, m in masks.items()}
    semantic_feat_class = {k: dataframe["semantic_feat"][m] for k, m in masks.items()}

    for k1 in cosine_similarities_cam_1:
        for k2 in cosine_similarities_cam_1[k1]:
            if len(embed_cam_1_class[k1]) > 0 and len(embed_cam_1_class[k2]) > 0:
                for cam_data, cos_dict, euc_dict in [
                    (
                        embed_cam_1_class,
                        cosine_similarities_cam_1,
                        euc_distances_cam_1,
                    ),
                    (
                        embed_cam_2_class,
                        cosine_similarities_cam_2,
                        euc_distances_cam_2,
                    ),
                ]:
                    anchors = torch.norm(cam_data[k1], dim=-2)  # [N, Z]
                    queries = torch.norm(cam_data[k2], dim=-2)  # [M, Z]

                    anchors_norm = F.normalize(anchors, p=2, dim=1)  # [N, Z]
                    queries_norm = F.normalize(queries, p=2, dim=1)  # [M, Z]

                    # Compute cosine similarity
                    cos_sim_matrix = anchors_norm @ queries_norm.T  # # [N, M]
                    if k1 == k2:  # Avoid self-comparison and double counting
                        mask = torch.triu(
                            torch.ones_like(cos_sim_matrix), diagonal=1
                        ).bool()
                        cos_sim_matrix[~mask] = -2.0  # Set masked values to -2.0

                    cos_sim = cos_sim_matrix[cos_sim_matrix != -2.0].flatten()
                    cos_dict[k1][k2].append(cos_sim.cpu().numpy())

                    # Compute Euclidean distances
                    euc_dist = torch.norm(
                        anchors.unsqueeze(1) - queries.unsqueeze(0), dim=2
                    )  # [N, M]

                    if k1 == k2:  # Avoid self-comparison
                        mask = torch.triu(torch.ones_like(euc_dist), diagonal=1).bool()
                        euc_dist[~mask] = -1.0  # Set masked values to -1.0

                    euc_dist = euc_dist[euc_dist != -1.0]
                    euc_dict[k1][k2].append(euc_dist.cpu().numpy())

                # Compute cosine similarities for semantic embeddings
                anchors_norm = F.normalize(
                    semantic_feat_class[k1], p=2, dim=1
                )  # [N, Z]
                queries_norm = F.normalize(
                    semantic_feat_class[k2], p=2, dim=1
                )  # [M, Z]

                # Compute cosine similarity as dot product of normalized vectors
                cos_sim_matrix = anchors_norm @ queries_norm.T  # [N, M]

                if k1 == k2:
                    mask = cos_sim_matrix < failure_modes[f"{k1}_{k2}"]["cos_sim"]
                    if mask.any():
                        failure_modes[f"{k1}_{k2}"]["cos_sim"] = cos_sim_matrix[
                            mask
                        ].min()
                        mask = cos_sim_matrix == failure_modes[f"{k1}_{k2}"]["cos_sim"]
                        index_1, index_2 = (mask).nonzero(as_tuple=False)[0]
                        failure_modes[f"{k1}_{k2}"]["wrist_image1"] = dataframe[
                            "robot0_eye_in_hand_image"
                        ][masks[k1]][index_1]
                        failure_modes[f"{k1}_{k2}"]["front_image1"] = dataframe[
                            "agentview_image"
                        ][masks[k1]][index_1]
                        failure_modes[f"{k1}_{k2}"]["class1"] = k1
                        failure_modes[f"{k1}_{k2}"]["wrist_image2"] = dataframe[
                            "robot0_eye_in_hand_image"
                        ][masks[k2]][index_2]
                        failure_modes[f"{k1}_{k2}"]["front_image2"] = dataframe[
                            "agentview_image"
                        ][masks[k2]][index_2]
                        failure_modes[f"{k1}_{k2}"]["class2"] = k2

                if k1 != k2:
                    if f"{k1}_{k2}" not in failure_modes:
                        continue
                    mask = cos_sim_matrix > failure_modes[f"{k1}_{k2}"]["cos_sim"]
                    if mask.any():
                        failure_modes[f"{k1}_{k2}"]["cos_sim"] = cos_sim_matrix[
                            mask
                        ].max()
                        mask = cos_sim_matrix == failure_modes[f"{k1}_{k2}"]["cos_sim"]
                        index_1, index_2 = (mask).nonzero(as_tuple=False)[0]
                        failure_modes[f"{k1}_{k2}"]["wrist_image1"] = dataframe[
                            "robot0_eye_in_hand_image"
                        ][masks[k1]][index_1]
                        failure_modes[f"{k1}_{k2}"]["front_image1"] = dataframe[
                            "agentview_image"
                        ][masks[k1]][index_1]
                        failure_modes[f"{k1}_{k2}"]["class1"] = k1
                        failure_modes[f"{k1}_{k2}"]["wrist_image2"] = dataframe[
                            "robot0_eye_in_hand_image"
                        ][masks[k2]][index_2]
                        failure_modes[f"{k1}_{k2}"]["front_image2"] = dataframe[
                            "agentview_image"
                        ][masks[k2]][index_2]
                        failure_modes[f"{k1}_{k2}"]["class2"] = k2

                if k1 == k2:  # avoid double counting and self comparison
                    mask = torch.triu(
                        torch.ones_like(cos_sim_matrix), diagonal=1
                    ).bool()
                    cos_sim_matrix[~mask] = -2.0  # Set masked values to -2.0
                cos_sim_sem = cos_sim_matrix[cos_sim_matrix != -2.0]

                if k1 != k2:
                    cos_sim_sem_max, __ = torch.max(cos_sim_matrix, dim=0)
                    cosine_similarities_sem_max_difference[k1][k2].append(
                        cos_sim_sem_max.cpu().numpy()
                    )
                cosine_similarities_sem[k1][k2].append(cos_sim_sem.cpu().numpy())

    # Plotting
    fig, axes = plt.subplots(3, 2, figsize=(10, 6))
    fig.suptitle("DINO WM")
    titles = [
        "Cos Sim - Front Cam",
        "Cos Sim - Wrist Cam",
        "Euc Dist - Front Cam",
        "Euc Dist - Wrist Cam",
        "Cos Sim - Sem Feat",
        "Cos Sim Closest Pair - Sem Feat",
    ]

    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_ylabel("Probability Density")

    total_pairs_found = 0
    expected_pairs = {
        "safe_safe": num_safe * (num_safe - 1) // 2,
        "unsafe_unsafe": num_unsafe
        * (num_unsafe - 1)
        // 2,  # Assuming weak_unsafe is not counted separately
        "safe_unsafe": num_safe * num_unsafe,
        "total": expected_total_pairs,
    }

    colors = {
        "safe-safe": "blue",
        "unsafe-unsafe": "green",
        "safe-unsafe": "orange",
    }

    for k1 in cosine_similarities_cam_1:
        for k2 in cosine_similarities_cam_1[k1]:
            if cosine_similarities_cam_1[k1][k2]:
                cs1 = np.concatenate(cosine_similarities_cam_1[k1][k2])
                cs2 = np.concatenate(cosine_similarities_cam_2[k1][k2])
                ed1 = np.concatenate(euc_distances_cam_1[k1][k2])
                ed2 = np.concatenate(euc_distances_cam_2[k1][k2])
                sem_cs = np.concatenate(cosine_similarities_sem[k1][k2])
                if k1 != k2:
                    sem_cs_max = np.concatenate(
                        cosine_similarities_sem_max_difference[k1][k2]
                    )

                total_pairs_found += len(cs1)

                def limit_kde(arr):
                    return gaussian_kde(
                        arr
                        if len(arr) <= 1000
                        else np.random.choice(arr, 1000, replace=False)
                    )

                kde_cs1, kde_cs2 = limit_kde(cs1), limit_kde(cs2)
                kde_ed1, kde_ed2 = limit_kde(ed1), limit_kde(ed2)
                kde_sem = limit_kde(sem_cs)
                if k1 != k2:
                    kde_sem_max = limit_kde(sem_cs_max)

                x_cs = np.linspace(cs1.min() - 1e-3, 1, 1000)
                x_ed = np.linspace(ed1.min() - 1, ed1.max() + 1, 1000)
                x_sem = np.linspace(sem_cs.min() - 1e-3, 1, 1000)

                label = f"{k1}-{k2}"
                axes[0, 0].plot(x_cs, kde_cs1(x_cs), label=label, color=colors[label])
                axes[0, 1].plot(x_cs, kde_cs2(x_cs), label=label, color=colors[label])
                axes[1, 0].plot(x_ed, kde_ed1(x_ed), label=label, color=colors[label])
                axes[1, 1].plot(x_ed, kde_ed2(x_ed), label=label, color=colors[label])
                axes[2, 0].plot(
                    x_sem,
                    kde_sem(x_sem),
                    label=label,
                    color=colors[label],
                )
                if k1 != k2:
                    axes[2, 1].plot(
                        x_sem,
                        kde_sem_max(x_sem),
                        label=label,
                        color=colors[label],
                    )

    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    unique = dict(zip(labels, handles))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.98),
        frameon=False,
    )
    plt.savefig("latent_analysis.png")

    fig = plt.figure(figsize=(12, 6))
    outer = gridspec.GridSpec(
        len(failure_modes.keys()),
        1,
        height_ratios=[
            1,
        ]
        * len(failure_modes),
        hspace=0.5,
    )

    group_titles = []

    for k in failure_modes:
        group_titles.append(
            f"Failure Mode: {k}, Cos Sim: {failure_modes[k]['cos_sim']:.2f}"
        )

    # Y positions for group titles in figure coordinates
    group_title_positions = [
        0.93,
        0.64,
        0.34,
    ]  # increase second one slightly to avoid overlap

    for group_idx, (data_dict, title) in enumerate(
        zip(failure_modes.values(), group_titles)
    ):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=outer[group_idx], wspace=0.3
        )

        for j, key in enumerate(
            ["wrist_image1", "front_image1", "wrist_image2", "front_image2"]
        ):
            if data_dict[key] is None:
                continue
            ax = plt.Subplot(fig, inner[j])
            image = np.array(data_dict[key], dtype=np.float32)
            if np.max(image) > 1.0:
                image = image / 255.0
            ax.imshow(image)
            ax.set_title(f"{key}, Class: {data_dict[f'class{j // 2 + 1}']}", fontsize=9)
            ax.axis("off")
            fig.add_subplot(ax)

        # Group title (above row)
        fig.text(
            0.5,
            group_title_positions[group_idx],
            title,
            ha="center",
            va="bottom",  # ensures it aligns *above* the row
            fontsize=14,
            weight="bold",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Give top text breathing room
    plt.savefig("failure_modes.png")
