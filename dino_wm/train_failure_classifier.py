import argparse
import copy
import os
import random
import sys

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import umap.umap_ as umap
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader, Subset
from tqdm import *
from utils.utils import compare_kdes, load_state_dict_flexible, PrivilegedTeacherForcingLoss

import wandb
from models.dino_models import VideoTransformer, normalize_acs, select_xyyaw_from_state
from utils.test_loader import SplitTrajectoryDataset

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.extend(
    [
        base_dir,
    ]
)

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # set random seed for all gpus


def make_parser():
    parser = argparse.ArgumentParser()
    # export directory, training and val datasets, test datasets
    parser.add_argument(
        "--embedding-size",
        default=512,
        type=int,
        dest="sz_embedding",
        help="Size of embedding that is appended to backbone model.",
    )
    parser.add_argument(
        "--batch-size",
        default=150,
        type=int,
        dest="sz_batch",
        help="Number of samples per batch.",
    )
    parser.add_argument(
        "--epochs",
        default=60,
        type=int,
        dest="nb_epochs",
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--gpu-id", default=0, type=int, help="ID of GPU that is used for training."
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        dest="nb_workers",
        help="Number of workers for dataloader.",
    )
    parser.add_argument("--loss", default="priv", help="Criterion for training")
    parser.add_argument("--optimizer", default="adamw", help="Optimizer setting")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate setting")
    parser.add_argument(
        "--weight-decay", default=1e-4, type=float, help="Weight decay setting"
    )
    parser.add_argument(
        "--lr-decay-step", default=10, type=int, help="Learning decay step setting"
    )
    parser.add_argument(
        "--lr-decay-gamma", default=0.5, type=float, help="Learning decay gamma setting"
    )
    parser.add_argument(
        "--alpha", default=32, type=float, help="Scaling Parameter setting"
    )
    parser.add_argument(
        "--dont-save-model",
        dest="save_model",
        action="store_false",
        help="Don't save model",
    )
    return parser


parser = make_parser()
args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Dataset Loader and Sampler
BS = args.sz_batch  # batch size
BL = 1

hdf5_file = "/data/sunny/sweeper/train/consolidated.h5"
hdf5_file_test = "/data/sunny/sweeper/test/consolidated.h5"

train_data_labeled = SplitTrajectoryDataset(
    hdf5_file,
    BL,
    split="train",
    num_test=0,
)

test_data = SplitTrajectoryDataset(
    hdf5_file_test,
    BL,
    split="train",
    num_test=0,
    provide_labels=True,
    num_examples_per_class=None,  # Don't limit number of examples per class for evaluation
)

train_loader_labeled = DataLoader(
    train_data_labeled, batch_size=BS, shuffle=True, num_workers=args.nb_workers
)
test_loader = DataLoader(
    test_data, batch_size=BS, shuffle=True, num_workers=args.nb_workers
)

device = "cuda:0"


model = VideoTransformer(
    image_size=(224, 224),
    dim=384,  # DINO feature dimension
    ac_dim=10,  # Action embedding dimension
    state_dim=3,  # State dimension
    depth=6,
    heads=16,
    mlp_dim=2048,
    num_frames=3,
    dropout=0.1,
).to(device)

load_state_dict_flexible(model, "dino_wm/checkpoints/best_testing.pth")

for name, param in model.named_parameters():
    param.requires_grad = name.startswith("semantic_encoder")

params = {}

def mapping_fn(X):
    # Maps a distance to a cosine similarity
    # Distance of 1.0 -> cosine sim of -1.0
    # Distance of 0.0 -> cosine sim of 1.0
    return -2 * (X / 180) + 1

criterion = PrivilegedTeacherForcingLoss(mapping_fn=mapping_fn)


# Wandb Initialization
wandb_name_kwargs = params
wandb_name = f"{args.loss}" + "".join(
    f"_{key}_{value}" for key, value in wandb_name_kwargs.items() if value is not None
)
wandb.init(name=wandb_name, project="ProxyAnchor")
wandb.config.update(args)

wandb.define_metric("num_updates", step_metric="num_updates")
wandb.define_metric("*", step_metric="num_updates")

# Train Parameters
param_groups = [
    {
        "params": model.semantic_encoder.parameters(),  # Semantic encoder parameters
        "lr": float(args.lr) * 1,
    },
]
# Optimizer Setting
opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.StepLR(
    opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma
)

print("Training parameters: {}".format(vars(args)))
print("Training for {} epochs.".format(args.nb_epochs))
losses_list = []
best_epoch = 0
best_eval = -float("inf")

num_updates = 0

for epoch in tqdm(range(0, args.nb_epochs), desc="Training Epochs", position=0):
    model.train()

    losses_per_epoch = []
    auc_per_epoch = []
    y_pred_tot = []
    y_true_tot = []

    # Warmup: Train only new params, helps stabilize learning.
    # TODO: implement warmup training if needed

    # pbar = tqdm(enumerate(expert_loader))
    max_timesteps = 10_000  # Maximum number of timesteps to sample
    min_timesteps = 1_000  # Minimum number of timesteps to sample
    total_timesteps = len(train_data_labeled)
    if total_timesteps > max_timesteps:
        subset_indices = random.sample(range(total_timesteps), max_timesteps)
        subset = Subset(train_data_labeled, subset_indices)
        train_loader_labeled = DataLoader(
            subset, batch_size=BS, shuffle=True, num_workers=args.nb_workers
        )
    elif (  # If small dataset
        total_timesteps < min_timesteps
    ):
        # Step 1: Include all original samples once
        all_indices = list(range(total_timesteps))
        extra_needed = min_timesteps - total_timesteps
        extra_indices = random.choices(all_indices, k=extra_needed)
        combined_indices = all_indices + extra_indices
        bootstrapped_subset = Subset(train_data_labeled, combined_indices)

        train_loader_labeled = DataLoader(
            bootstrapped_subset,
            batch_size=BS,
            shuffle=True,
            num_workers=args.nb_workers,
        )

    pbar = tqdm(
        enumerate(train_loader_labeled),
        total=len(train_loader_labeled),
        position=1,
        leave=False,
    )

    # args.beta = np.linspace(0.2, 1.0, args.nb_epochs)[epoch]  # Linear increase of beta

    def create_coverage_plot():
        fig_coverage, ax_coverage = plt.subplots(figsize=(8, 8))
        ax_coverage.set_title("Coverage of Data on Table")
        ax_coverage.set_xlabel("X Coordinate")
        ax_coverage.set_ylabel("Y Coordinate")
        ax_coverage.set_xlim(0, 224)
        ax_coverage.set_ylim(0, 224)
        ax_coverage.set_aspect("equal")
        ax_coverage.invert_yaxis()
        return fig_coverage, ax_coverage

    fig_coverage, ax_coverage = create_coverage_plot()
    fig_coverage_eval, ax_coverage_eval = create_coverage_plot()

    for batch_idx, data in pbar:
        # labels_gt: [150 2]
        labels_gt = data["failure"][:].to(device, dtype=torch.float32).squeeze()
        labels_gt_xy = get_class_from_xy(labels_gt)

        data1 = data["cam_zed_embd"].to(device)  # [B 1, 256, 384]
        # data2 = data["cam_rs_embd"].to(device)  # [B 1, 256, 384]
        inputs1 = data1[:, -1:]  # [B 1, 256, 384]

        data_state = select_xyyaw_from_state(data["state"]).to(device)
        states = data_state[:, -1:]  # [B 1, 3]

        data_acs = data["action"].to(device)
        acs = data_acs[:, -1:]  # [B 1, 10]
        acs = normalize_acs(acs, device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            semantic_features = model.semantic_embed(inp1=inputs1, state=states)

        labels_gt_xy_masked = copy.deepcopy(labels_gt_xy)
        mask = (labels_gt_xy_masked != -1.0).squeeze()
        labels_gt_xy_masked = labels_gt_xy_masked[mask]
        semantic_features = semantic_features[mask]  # Remove -1 labels

        ax_coverage.scatter(
            data["failure"][:, -1][mask.cpu(), 0],
            data["failure"][:, -1][mask.cpu(), 1],
            color=[
                class_to_colors[label.item()]
                for label in labels_gt_xy[mask.cpu()].cpu().numpy()
            ],
        )

        loss = criterion(
            X=einops.rearrange(
                semantic_features.float(), "B T Z -> (B T) Z"
            ).cuda(),
            T=labels_gt[mask],
        )

        opt.zero_grad()
        loss.backward()
        num_updates += 1

        torch.nn.utils.clip_grad_value_(model.semantic_encoder.parameters(), 10)

        losses_per_epoch.append(loss.data.cpu().numpy())
        opt.step()

        pbar.set_description(
            "Train Epoch: {} Loss: {:.6f}".format(
                epoch,
                loss.item(),
            )
        )

    fig_coverage.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=label_to_str[label],
                markersize=10,
                markerfacecolor=color,
            )
            for label, color in class_to_colors.items()
        ],
        title="Classes",
    )
    wandb.log({"train/coverage": wandb.Image(fig_coverage)}, step=num_updates)

    losses_list.append(np.mean(losses_per_epoch))
    wandb.log(
        {"train/Loss": losses_list[-1], "num_updates": num_updates},
        step=num_updates,
    )

    scheduler.step()

    if epoch >= 0:
        model.eval()
        loses_per_epoch = []
        X = []
        y = []
        y_priv = []
        with torch.no_grad():
            pbar = tqdm(
                enumerate(test_loader),
                total=len(test_loader),
                desc="Evaluation",
                position=2,
                leave=False,
            )
            for batch_idx, data in pbar:
                labels_gt = data["failure"][:, -1:].to(
                    device, dtype=torch.float32
                )  # [B, 2]
                labels_gt_xy = get_class_from_xy(labels_gt)
                mask = (labels_gt_xy != -1.0).squeeze()

                ax_coverage_eval.scatter(
                    data["failure"][:, -1][mask.cpu(), 0],
                    data["failure"][:, -1][mask.cpu(), 1],
                    color=[
                        class_to_colors[label.item()]
                        for label in labels_gt_xy[mask.cpu()].cpu().numpy()
                    ],
                )

                inputs1 = (  # [B, 1, 256, 384]
                    data["cam_zed_embd"][:, -1:].to(device)[mask]
                )
                states = select_xyyaw_from_state(data["state"][:, -1:]).to(device)[
                    mask
                ]  # [B, 1, 3]

                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=True
                ):
                    semantic_features = model.semantic_embed(  # [embedding_dim]
                        inp1=inputs1, state=states
                    )

                labels_gt_xy_masked = copy.deepcopy(labels_gt_xy)[mask]

                # Normalize all vectors for cosine similarity
                semantic_features_norm = F.normalize(
                    semantic_features, dim=-1
                )  # (10, 1, 512)

                # Broadcastable shapes: (10, 1, 1, 512) and (1, 1, 2, 512)
                semantic_features_exp = einops.rearrange(
                    semantic_features_norm, "B T Z -> B T 1 Z"
                )  # (BS, T, 1, 512)

                loss = criterion(
                    X=einops.rearrange(
                        semantic_features.float(), "B T Z -> (B T) Z"
                    ).cuda(),
                    T=labels_gt[mask].squeeze().cuda(),
                )

                losses_per_epoch.append(loss.detach().cpu().numpy())

                X.append(semantic_features.cpu().numpy())
                y.append(labels_gt_xy_masked.cpu().numpy())
                y_priv.append(labels_gt[mask].cpu().numpy())

            X = einops.rearrange(np.concatenate(X, axis=0), "B T Z -> (B T) Z")
            y = np.concatenate(y, axis=0).squeeze()
            y_priv = np.concatenate(y_priv, axis=0).squeeze()
            classes_eval = np.unique(y)
            num_classes_eval = len(np.unique(y))

            y_gt_masked = copy.deepcopy(y)

            # Calculate metrics
            fig_coverage_eval.legend(
                handles=[
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=label_to_str[label],
                        markersize=10,
                        markerfacecolor=color,
                    )
                    for label, color in class_to_colors.items()
                ],
                title="Classes",
            )
            wandb.log(
                {"eval/coverage": wandb.Image(fig_coverage_eval)}, step=num_updates
            )

            wandb.log({"eval/Loss": np.mean(losses_per_epoch)}, step=num_updates)
            # metrics["Cross Entropy Loss"].append(
            #     F.cross_entropy(cos_sim, gt_labels, reduction="mean").item()
            # )

        cosine_sims = {i: {j: [] for j in classes_eval} for i in classes_eval}

        def cosine_sim_plot_eval(X, y):
            y_masked = copy.deepcopy(y)

            X_class = {
                k: X[y_masked == k]
                / (np.linalg.norm(X[y_masked == k], axis=1, keepdims=True) + 1e-8)
                for k in np.unique(y_masked)
            }

            fig, ax = plt.subplots(figsize=(5 * num_classes_eval, 8))

            plt.title("Cosine Similarity Distribution per Class")
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Normalized Density")

            class_pairs = []
            for i in classes_eval:
                for j in classes_eval:
                    if i <= j:
                        class_pairs.append((i, j))

            cmap = plt.cm.rainbow
            colors = [cmap(i / len(class_pairs)) for i in range(len(class_pairs))]

            kde_dict = {}

            if len(class_pairs) > 10:  # If too many pairs, do one vs. rest
                ovr = True
                for i in classes_eval:
                    # Same to same comparison
                    cos_sim = X_class[i] @ X_class[i].T
                    mask = np.triu(np.ones_like(cos_sim, dtype=bool), k=1)
                    cos_sim = np.where(mask, cos_sim, -2.0)
                    cos_sim = cos_sim[cos_sim != -2.0].flatten()

                    if len(cos_sim) > 1000:
                        cos_sim_sampled = np.random.choice(cos_sim, 1000, replace=False)
                    else:
                        cos_sim_sampled = cos_sim

                    kde_cs = gaussian_kde(cos_sim_sampled)
                    kde_dict[(i, i)] = kde_cs

                    # One vs. rest comparison
                    cos_sim_list = []
                    for j in classes_eval:
                        if i != j:
                            cos_sim = X_class[i] @ X_class[j].T

                            cos_sim_list.append(cos_sim.flatten())

                    cos_sim = np.concatenate(cos_sim_list, axis=0)
                    if len(cos_sim) > 1000:
                        cos_sim_sampled = np.random.choice(cos_sim, 1000, replace=False)
                    else:
                        cos_sim_sampled = cos_sim
                    kde_cs = gaussian_kde(cos_sim_sampled)
                    kde_dict[(i, "rest")] = kde_cs

            else:
                ovr = False
                for idx, (i, j) in enumerate(class_pairs):
                    cos_sim = X_class[i] @ X_class[j].T

                    if i == j:  # Avoid self-comparison and double counting
                        mask = np.triu(np.ones_like(cos_sim, dtype=bool), k=1)
                        cos_sim = np.where(mask, cos_sim, -2.0)

                    cos_sim = cos_sim[cos_sim != -2.0].flatten()

                    if len(cos_sim) > 1000:
                        cos_sim_sampled = np.random.choice(cos_sim, 1000, replace=False)
                    else:
                        cos_sim_sampled = cos_sim

                    kde_cs = gaussian_kde(cos_sim_sampled)
                    kde_dict[(i, j)] = kde_cs

            for idx, ((i, j), kde_cs) in enumerate(kde_dict.items()):
                x_cs = np.linspace(-1 - 1e-3, 1 + 1e-3, 1000)
                y_pdf = kde_cs(x_cs)
                dx = x_cs[1] - x_cs[0]
                y_pdf_normalized = y_pdf / (np.sum(y_pdf) * dx)

                if j == "rest":
                    color = colors[idx]
                    label = f"{label_to_str[i]}-rest"
                else:
                    color = "black" if ovr else colors[idx]
                    label = f"{label_to_str[i]}-{label_to_str[j]}"
                ax.plot(x_cs, y_pdf_normalized, label=label, color=color)

                # Statistics
                median_val = np.median(cos_sim_sampled)
                lower, upper = np.percentile(cos_sim_sampled, [2.5, 97.5])

                # Get KDE values at stat locations
                median_y = kde_cs(median_val) / (np.sum(y_pdf) * dx)
                lower_y = kde_cs(lower) / (np.sum(y_pdf) * dx)
                upper_y = kde_cs(upper) / (np.sum(y_pdf) * dx)

                # Plot short vertical lines
                ax.vlines(
                    median_val, 0, median_y, color=color, linestyle="dashed", alpha=0.8
                )
                ax.vlines(lower, 0, lower_y, color=color, linestyle="dotted", alpha=0.5)
                ax.vlines(upper, 0, upper_y, color=color, linestyle="dotted", alpha=0.5)

                # Text labels with white background
                label_kwargs = dict(
                    ha="center",
                    fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=1.0),
                )
                ax.text(
                    median_val,
                    median_y + 0.01,
                    f"m={median_val:.2f}",
                    color=color,
                    **label_kwargs,
                )
                ax.text(
                    lower, lower_y + 0.01, f"↓{lower:.2f}", color=color, **label_kwargs
                )
                ax.text(
                    upper, upper_y + 0.01, f"↑{upper:.2f}", color=color, **label_kwargs
                )

            # Add dummy lines for legend explanation
            ax.plot([], [], linestyle="dashed", color="black", label="m = Median")
            ax.plot(
                [], [], linestyle="dotted", color="black", label="↓ ↑ = 95% Interval"
            )

            ax.legend()
            plt.tight_layout()
            wandb.log(
                {"eval/cosine_sim_plot": wandb.Image(fig), "num_updates": num_updates},
                step=num_updates,
            )

            # Generate comparison statistics
            js_div, ws_dist = [], []
            for _class in np.unique(y_masked):
                if ovr:
                    js_div_, ws_dist_ = compare_kdes(
                        kde1=kde_dict[(_class, _class)],
                        kde2=kde_dict[(_class, "rest")],
                    )
                    js_div.append(js_div_)
                    ws_dist.append(ws_dist_)
                else:
                    for _class2 in np.unique(y_masked):
                        if _class < _class2:
                            js_div_, ws_dist_ = compare_kdes(
                                kde1=kde_dict[(_class, _class)],
                                kde2=kde_dict[(_class, _class2)],
                            )
                            js_div.append(js_div_)
                            ws_dist.append(ws_dist_)

            wandb.log(
                {
                    "eval/avg_JS": np.mean(js_div),
                    "eval/avg_wass_dist": np.mean(ws_dist),
                    "num_updates": num_updates,
                },
                step=num_updates,
            )
            plt.close()

        cosine_sim_plot_eval(X, y)

        def cos_sim_semantic(X, y_priv):
            mask = random.sample(range(len(y_priv)), min(1000, len(y_priv)))
            y_priv_masked = copy.deepcopy(y_priv[mask])
            X_masked = X[mask]

            X_masked_norm = F.normalize(torch.tensor(X_masked, device=device), dim=-1)
            cos_sim = (X_masked_norm @ X_masked_norm.T).flatten().cpu().numpy()

            diff = y_priv_masked[:, None] - y_priv_masked[None, :]
            dist = np.linalg.norm(diff, axis=-1).flatten()

            fig, ax = plt.subplots(figsize=(6, 6))

            plt.title("XY Distance vs Cosine Similarity")
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Distance")
            plt.xlim(max(-1, cos_sim.min()), min(1, cos_sim.max()))

            # Scatter plot
            ax.scatter(cos_sim, dist, s=5, alpha=0.5)

            y = np.linspace(0, dist.max(), 1000)
            x = mapping_fn(y)

            # Plot the mapping function
            ax.plot(x, y, color="red", linestyle="--", label="Mapping Function")
            plt.tight_layout()
            wandb.log(
                {
                    "eval/cosine_sim_vs_dist": wandb.Image(fig),
                    "num_updates": num_updates,
                },
                step=num_updates,
            )
            plt.close()

        cos_sim_semantic(X, y_priv)

        wandb.log({"num_updates": num_updates}, step=num_updates)

        # ---- Flatten and Prepare Data ----
        if len(X) == 0 or len(y) == 0:
            print("No data to visualize.")
            continue

        # print("Visualizing embeddings with UMAP...")

        # ---- UMAP Setup ----
        umap_input = np.concatenate([X], axis=0)

        reducer = umap.UMAP(n_components=2, metric="cosine")
        umap_output = reducer.fit_transform(umap_input)

        X_umap = umap_output

        # ---- Plot ----
        plt.figure(figsize=(8, 6))

        # Plot data points
        for class_idx in range(num_classes_eval):
            idxs = y == classes_eval[class_idx]
            plt.scatter(
                X_umap[idxs, 0],
                X_umap[idxs, 1],
                s=15,
                color=class_to_colors[class_idx],
                label=f"Class {class_idx} (data)",
                alpha=0.7,
            )

        # ---- Final Formatting ----
        plt.title("UMAP visualization of embeddings (cosine distance)")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.legend(loc="best", fontsize=8, frameon=True)
        plt.tight_layout()

        wandb.log(
            {"eval/umap_plot": wandb.Image(plt), "num_updates": num_updates},
            step=num_updates,
        )
        plt.close()

        if args.save_model:
            model_name = wandb_name
            save_name = f"../checkpoints_sem/encoder_{model_name}.pth"
            # best_save_name = f"../checkpoints_sem/best_encoder_{model_name}.pth"

            torch.save(
                model.state_dict(),
                save_name,
            )
            tqdm.write(f"Model saved to {save_name}")
