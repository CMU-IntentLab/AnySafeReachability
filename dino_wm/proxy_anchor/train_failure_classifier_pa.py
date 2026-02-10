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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset
from tqdm import *
from utils import compare_kdes, load_state_dict_flexible

import wandb
from dino_wm.dino_models import VideoTransformer, normalize_acs, select_xyyaw_from_state
from dino_wm.test_loader import SplitTrajectoryDataset
from proxy_anchor.code import losses

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
    parser = argparse.ArgumentParser(
        description="Official implementation of `Proxy Anchor Loss for Deep Metric Learning`"
        + "Our code is modified from `https://github.com/dichotomies/proxy-nca`"
    )
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
    parser.add_argument("--model", default="bn_inception", help="Model for training")
    parser.add_argument("--loss", default="Proxy_Anchor", help="Criterion for training")
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
        "--mrg", default=0.1, type=float, help="Margin parameter setting"
    )
    parser.add_argument(
        "--temp",
        default=0.05,
        type=float,
        help="Temperature for softmax in Proxy Anchor",
    )
    parser.add_argument(
        "--beta",
        default=0.1,
        type=float,
        help="Beta parameter for Proxy Anchor loss, controls the influence of unlabeled data",
    )
    parser.add_argument("--IPC", type=int, help="Balanced sampling, images per class")
    parser.add_argument("--warm", default=1, type=int, help="Warmup training epochs")
    parser.add_argument(
        "--bn-freeze", default=1, type=int, help="Batch normalization parameter freeze"
    )
    parser.add_argument("--l2-norm", default=1, type=int, help="L2 normlization")
    parser.add_argument("--remark", default="", help="Any remark")
    parser.add_argument(
        "--dont-save-model",
        dest="save_model",
        action="store_false",
        help="Don't save model",
    )
    parser.add_argument(
        "--num-examples-per-class",
        type=int,
        default=None,  # None means all examples
        help="Number of examples per class for training",
    )
    parser.add_argument(
        "--use-unlabeled-data",
        action="store_true",
        default=False,
        help="Use unlabeled data for training",
    )
    parser.add_argument(
        "--unlabeled-ratio",
        type=float,
        default=1.5,
        help="Ratio of unlabeled data to labeled data for training",
    )
    parser.add_argument(
        "--ratio-schedule",
        type=str,
        default="const",
        choices=["const", "lin", "exp"],
        help="Schedule for the ratio of unlabeled data to labeled data",
    )

    parser.add_argument(
        "--boundary-type",
        type=str,
        default="2x3",
        choices=["2x3", "1x3", "3x1", "2x2", "1x4"],
        help="Type of boundary for classification",
    )
    return parser


parser = make_parser()
args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Wandb Initialization
wandb_name_kwargs = {
    "mrg": args.mrg,
    "alpha": int(args.alpha),
    "bound": args.boundary_type,
}
if args.use_unlabeled_data:
    wandb_name_kwargs["ul"] = "T"
    wandb_name_kwargs["ul_ratio"] = (
        args.unlabeled_ratio
        if args.unlabeled_ratio != -1.0
        else f"all_{args.ratio_schedule}"
    )
    wandb_name_kwargs["beta"] = args.beta
    wandb_name_kwargs["temp"] = args.temp

wandb_name = "".join(
    f"{key}_{value}_" for key, value in wandb_name_kwargs.items() if value is not None
).rstrip("_")
wandb.init(name=wandb_name, project="ProxyAnchor")
wandb.config.update(args)

wandb.define_metric("num_updates", step_metric="num_updates")
wandb.define_metric("*", step_metric="num_updates")

# Dataset Loader and Sampler
BS = args.sz_batch  # batch size
BL = 1

device = "cuda:0"

if args.boundary_type == "2x3":
    x_class_boundaries = [0, 224 // 3, 224 * 2 // 3, 224]  # x boundaries for 3 classes
    y_class_boundaries = [224 // 3, 224 * 2 // 3, 224]  # y boundaries for 3 classes

    label_to_str = {
        0: "Left Top",
        1: "Left Bottom",
        2: "Middle Top",
        3: "Middle Bottom",
        4: "Right Top",
        5: "Right Bottom",
    }
elif args.boundary_type == "1x3":
    x_class_boundaries = [0, 224 // 3, 224 * 2 // 3, 224]  # x boundaries for 3 classes
    y_class_boundaries = [0, 224]

    label_to_str = {
        0: "Left",
        1: "Middle",
        2: "Right",
    }

elif args.boundary_type == "3x1":
    y_class_boundaries = [
        224 // 3,
        5 * 224 // 9,
        7 * 224 // 9,
        224,
    ]  # y boundaries for 3 classes
    x_class_boundaries = [0, 224]

    label_to_str = {
        0: "Top",
        1: "Middle",
        2: "Bottom",
    }

elif args.boundary_type == "2x2":
    x_class_boundaries = [0, 224 // 2, 224]  # x boundaries for 2 classes
    y_class_boundaries = [224 // 3, 2 * 224 // 3, 224]  # y boundaries for 2 classes
    label_to_str = {
        0: "Left Top",
        1: "Left Bottom",
        2: "Right Top",
        3: "Right Bottom",
    }
elif args.boundary_type == "1x4":
    x_class_boundaries = [
        0,
        224 // 4,
        224 * 2 // 4,
        224 * 3 // 4,
        224,
    ]  # x boundaries for 4 classes
    y_class_boundaries = [0, 224]
    label_to_str = {
        0: "Left",
        1: "Middle Left",
        2: "Middle Right",
        3: "Right",
    }

else:
    raise ValueError("Invalid boundary type: {}".format(args.boundary_type))

# 3 * 2 = 6 classes in total
nb_classes = (len(x_class_boundaries) - 1) * (len(y_class_boundaries) - 1)

cmap = plt.cm.rainbow
class_to_colors = {i: cmap(i / nb_classes) for i in range(nb_classes)}


def get_class_from_xy(labels):
    device = "cuda:0"
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
    nb_classes=nb_classes,
).to(device)
# model.load_state_dict(torch.load("../checkpoints/best_classifier.pth"), strict=False)
# load_state_dict_flexible(model, "../checkpoints/best_classifier.pth")
load_state_dict_flexible(model, "../checkpoints/best_testing.pth")
# load_state_dict_flexible(model, "../checkpoints/best_testing_xy.pth")
# model.load_state_dict(torch.load("../checkpoints_pa/encoder_0.1.pth"))

for name, param in model.named_parameters():
    param.requires_grad = name.startswith("semantic_encoder")

# DML Losses
criterion = losses.Proxy_Anchor(
    nb_classes=nb_classes,
    sz_embed=args.sz_embedding,
    mrg=args.mrg,
    alpha=args.alpha,
).cuda()

# Train Parameters
param_groups = [
    {
        "params": model.semantic_encoder.parameters(),  # Semantic encoder parameters
        "lr": float(args.lr) * 1,
    },
    {"params": criterion.parameters(), "lr": float(args.lr) * 100},  # Just proxies
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

# Datasets
hdf5_file = "/home/sunny/data/sweeper/train/consolidated.h5"
hdf5_file_test = "/home/sunny/data/sweeper/test/consolidated.h5"

train_data_labeled = SplitTrajectoryDataset(
    hdf5_file,
    BL,
    split="train",
    num_test=0,
    provide_labels=True,  # Labeled data
    num_examples_per_class=-1,  # args.num_examples_per_class,
    xy_to_class_label_fn=get_class_from_xy,
    only_pass_labeled_examples=True,
)

if args.use_unlabeled_data:
    train_data_unlabeled = SplitTrajectoryDataset(
        hdf5_file,
        BL,
        split="train",
        num_test=0,
        provide_labels=False,  # Unlabeled data
        num_examples_per_class=int(args.num_examples_per_class * args.unlabeled_ratio)
        if args.unlabeled_ratio != -1.0
        else None,
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

num_updates = 0

for epoch in tqdm(range(0, args.nb_epochs), desc="Training Epochs", position=0):
    model.train()

    losses_per_epoch = []
    auc_per_epoch = []
    metrics = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-score": [],
    }
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
    elif (  # If small dataset and using unlabeled data
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

    if (  # If using full dataset
        args.use_unlabeled_data and args.unlabeled_ratio == -1.0
    ):
        if args.ratio_schedule == "const":
            num_to_sample = max_timesteps
        elif args.ratio_schedule == "lin":
            num_to_sample = (max_timesteps * epoch / args.nb_epochs).astype(int)
        elif args.ratio_schedule == "exp":
            num_to_sample = (
                max_timesteps * np.exp(-5 * (1 - epoch / args.nb_epochs) ** 2)
            ).astype(int)
        else:
            raise ValueError("Invalid ratio schedule: {}".format(args.ratio_schedule))
        subset_indices_unlabeled = random.sample(
            range(len(train_data_unlabeled)),
            num_to_sample,
        )
        subset_unlabeled = Subset(train_data_unlabeled, subset_indices_unlabeled)
        train_loader_unlabeled = DataLoader(
            subset_unlabeled, batch_size=BS, shuffle=True, num_workers=args.nb_workers
        )

    elif args.use_unlabeled_data and args.unlabeled_ratio != -1.0:
        train_loader_unlabeled = DataLoader(
            dataset=train_data_unlabeled,
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
        # Gaussian noise to labels for robustness
        # labels_gt = labels_gt + torch.randn_like(labels_gt) * 3.0
        labels_gt = get_class_from_xy(labels_gt)

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
            if args.use_unlabeled_data:  # and epoch >= 20:
                semantic_features_unlabeled_tensor = []
                for idx, data_unlabeled in enumerate(train_loader_unlabeled):
                    semantic_features_unlabeled = model.semantic_embed(
                        inp1=data_unlabeled["cam_zed_embd"][:, -1:].to(device),
                        # inp2=data_unlabeled["cam_rs_embd"][:, -1:].to(device),
                        state=select_xyyaw_from_state(
                            data_unlabeled["state"][:, -1:]
                        ).to(device),
                    )
                    semantic_features_unlabeled_tensor.append(
                        semantic_features_unlabeled
                    )
                semantic_features_unlabeled_tensor = torch.cat(
                    semantic_features_unlabeled_tensor, dim=0
                )  # Concatenate all unlabeled features

                # If ratio is specified, sample the correct amount of unlabeled data
                if args.use_unlabeled_data and args.unlabeled_ratio != -1.0:
                    # Ensure that correct amount of unlabeled data is given
                    all_indices = list(range(len(semantic_features_unlabeled_tensor)))
                    needed_datapoints = int(
                        semantic_features.shape[0] * args.unlabeled_ratio
                    )
                    if needed_datapoints > len(all_indices):
                        extra_needed = needed_datapoints - len(all_indices)
                        extra_indices = random.choices(all_indices, k=extra_needed)
                        combined_indices = all_indices + extra_indices
                    else:
                        combined_indices = all_indices[
                            random.sample(range(len(all_indices)), k=needed_datapoints)
                        ]
                    semantic_features_unlabeled_tensor = (
                        semantic_features_unlabeled_tensor[combined_indices]
                    )

        labels_gt_masked = copy.deepcopy(labels_gt)
        mask = (labels_gt_masked != -1.0).squeeze()
        labels_gt_masked = labels_gt_masked[mask]
        semantic_features = semantic_features[mask]  # Remove -1 labels

        ax_coverage.scatter(
            data["failure"][:, -1][mask.cpu(), 0],
            data["failure"][:, -1][mask.cpu(), 1],
            color=[
                class_to_colors[label.item()]
                for label in labels_gt_masked.cpu().numpy()
            ],
        )

        loss, __, __ = criterion(
            X=semantic_features.float(),
            T=labels_gt_masked.squeeze().cuda(),
        )

        if args.use_unlabeled_data:
            wandb.log(
                {
                    "train/data_ratio": semantic_features_unlabeled_tensor.shape[0]
                    / semantic_features.shape[0]
                },
                step=num_updates,
            )

        P = criterion.proxies.detach()  # Ensure P is in the same dtype as X
        assert criterion.proxies.requires_grad
        semantic_features = einops.rearrange(
            semantic_features.float(), "B T Z -> (B T) Z"
        )  # Ensure X is in the correct shape

        cos_sim = F.linear(losses.l2_norm(semantic_features), losses.l2_norm(P))
        cos_sim_logits = F.softmax(cos_sim, dim=-1)  # Softmax over classes

        if nb_classes == 2:
            auc = roc_auc_score(
                y_true=einops.rearrange(labels_gt_masked, "B T -> (B T)").cpu().numpy(),
                y_score=cos_sim_fail.detach().cpu().numpy(),
            )
            cos_sim_proxies = (
                torch.gather(  # Cos sim for data point to corresponding proxy
                    F.linear(losses.l2_norm(semantic_features), losses.l2_norm(P)),
                    dim=1,
                    index=labels_gt_masked.to(torch.int64),
                )
            )
            cos_sim_proxies_incorrect = (
                torch.gather(  # Cos sim for data point to corresponding proxy
                    F.linear(losses.l2_norm(semantic_features), losses.l2_norm(P)),
                    dim=1,
                    index=1 - labels_gt_masked.to(torch.int64),
                )
            )
            wandb.log(
                {
                    "train/cos_sim_to_correct_proxies": cos_sim_proxies.mean().item(),
                    "train/cos_sim_to_incorrect_proxies": cos_sim_proxies_incorrect.mean().item(),
                },
                step=num_updates,
            )

        else:
            auc = roc_auc_score(
                y_true=labels_gt_masked.cpu().numpy().squeeze(),
                y_score=cos_sim_logits.detach().cpu().numpy(),
                multi_class="ovo",
                average="macro",
                labels=np.arange(nb_classes),
            )

        y_pred = cos_sim_logits.argmax(dim=-1)  # Predicted labels

        if batch_idx % 50 == 0:
            # Show train images
            fig, ax = plt.subplots(figsize=(8, 8))

            ax.imshow(data["agentview_image"][mask.cpu()][-1, -1].cpu().numpy())

            ax.scatter(
                data["failure"][mask.cpu()][-1, -1, 0].cpu().numpy(),
                data["failure"][mask.cpu()][-1, -1, 1].cpu().numpy(),
                marker="x",
                color="blue",
            )
            wandb.log({"train/front_image": wandb.Image(fig)})
            plt.close(fig)

        metrics["Accuracy"].append(
            (y_pred == labels_gt_masked.squeeze()).float().mean().item()
        )
        metrics["Precision"].append(
            precision_score(
                labels_gt_masked.squeeze().cpu().numpy(),
                y_pred.cpu().numpy(),
                average="macro",
                zero_division=0,
            )
        )
        metrics["Recall"].append(
            recall_score(
                labels_gt_masked.squeeze().cpu().numpy(),
                y_pred.cpu().numpy(),
                average="macro",
                zero_division=0,
            )
        )
        metrics["F1-score"].append(
            f1_score(
                labels_gt_masked.squeeze().cpu().numpy(),
                y_pred.cpu().numpy(),
                average="macro",
                zero_division=0,
            )
        )
        y_pred_tot.append(y_pred.cpu().numpy())
        y_true_tot.append(labels_gt_masked.cpu().numpy())

        opt.zero_grad()
        loss.backward()
        num_updates += 1

        torch.nn.utils.clip_grad_value_(model.semantic_encoder.parameters(), 10)
        torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)

        losses_per_epoch.append(loss.data.cpu().numpy())
        auc_per_epoch.append(auc)
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

    wandb.log(
        {
            "train/Confusion Matrix": wandb.plot.confusion_matrix(
                preds=np.concatenate(y_pred_tot),
                y_true=np.concatenate(y_true_tot).squeeze(),
                class_names=list(label_to_str.values()),
            )
        },
        step=num_updates,
    )

    losses_list.append(np.mean(losses_per_epoch))
    wandb.log(
        {"train/Proxy Anchor Loss": losses_list[-1], "num_updates": num_updates},
        step=num_updates,
    )
    wandb.log(
        {"train/AUC": np.mean(auc_per_epoch), "num_updates": num_updates},
        step=num_updates,
    )
    for metric, values in metrics.items():
        wandb.log(
            {f"train/{metric}": np.mean(values), "num_updates": num_updates},
            step=num_updates,
        )

    scheduler.step()

    if epoch >= 0:
        model.eval()
        metrics = {}
        X = []
        y = []
        y_pred = []
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
                )  # [B, 1]
                labels_gt = get_class_from_xy(labels_gt)
                mask = (labels_gt != -1.0).squeeze()

                ax_coverage_eval.scatter(
                    data["failure"][:, -1][mask.cpu(), 0],
                    data["failure"][:, -1][mask.cpu(), 1],
                    color=[
                        class_to_colors[label.item()]
                        for label in labels_gt[mask.cpu()].cpu().numpy()
                    ],
                )

                inputs1 = (  # [B, 1, 256, 384]
                    data["cam_zed_embd"][:, -1:].to(device)[mask]
                )
                states = select_xyyaw_from_state(data["state"][:, -1:]).to(device)[
                    mask
                ]  # [B, 1, 3]

                semantic_features = model.semantic_embed(  # [embedding_dim]
                    inp1=inputs1, state=states
                )

                labels_gt_masked = copy.deepcopy(labels_gt)[mask]

                # Normalize all vectors for cosine similarity
                semantic_features_norm = F.normalize(
                    semantic_features, dim=-1
                )  # (10, 1, 512)
                proxies_norm = F.normalize(criterion.proxies, dim=-1)  # (2, 512)

                # Broadcastable shapes: (10, 1, 1, 512) and (1, 1, 2, 512)
                semantic_features_exp = einops.rearrange(
                    semantic_features_norm, "B T Z -> B T 1 Z"
                )  # (BS, T, 1, 512)
                proxies_exp = einops.rearrange(
                    proxies_norm, "L Z -> 1 1 L Z"
                )  # (1, 1, 2, 512)

                # Compute cosine similarity
                cos_sim = (semantic_features_exp * proxies_exp).sum(
                    dim=-1
                )  # (10, 1, 2)

                # Choose the index (0 or 1) of the most similar vector
                logits = F.softmax(cos_sim, dim=-1)  # (10, 1, 2)
                pred_labels = logits.argmax(dim=-1)  # (10, 1)

                X.append(semantic_features.cpu().numpy())
                y.append(labels_gt_masked.cpu().numpy())
                y_pred.append(pred_labels.cpu().numpy())

            X = einops.rearrange(np.concatenate(X, axis=0), "B T Z -> (B T) Z")
            y = np.concatenate(y, axis=0).squeeze()
            y_pred = np.concatenate(y_pred, axis=0).squeeze()
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
            metrics["Accuracy"] = balanced_accuracy = accuracy_score(
                y_gt_masked, y_pred
            )
            metrics["Precision"] = precision_score(
                y_gt_masked, y_pred, average="macro", zero_division=0
            )
            metrics["Recall"] = recall_score(
                y_gt_masked, y_pred, average="macro", zero_division=0
            )
            metrics["F1-score"] = f1_score(
                y_gt_masked, y_pred, average="macro", zero_division=0
            )
            metrics["Balanced Accuracy"] = accuracy_score(y_gt_masked, y_pred)
            loss, __, __ = criterion(
                X=torch.tensor(X, device=device),
                T=torch.tensor(y_gt_masked, device=device),
            )
            metrics["Proxy Anchor Loss"] = loss.detach().cpu().numpy()
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

            if len(class_pairs) >= 7:  # If too many pairs, do one vs. rest
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

        def const_conditioned_plots(X, y):
            y_masked = copy.deepcopy(y)

            P = criterion.proxies.detach()  # Ensure P is in the same dtype as X

            cos_sim_proxies = -F.linear(
                losses.l2_norm(torch.tensor(X, device=P.device).float()),
                losses.l2_norm(P),
            ).cpu()

            thresholds = np.linspace(-1, 1, 100)
            proxy_data = {
                k: {
                    "cos_sim": cos_sim_proxies[:, k],
                    "tp_rates": [],
                    "tn_rates": [],
                    "fp_rates": [],
                    "fn_rates": [],
                }
                for k in classes_eval
            }
            for t in thresholds:
                for prox, data in proxy_data.items():
                    cos_sim = data["cos_sim"]

                    tp = ((cos_sim > t) & (y_masked != prox)).sum()
                    fp = ((cos_sim > t) & (y_masked == prox)).sum()
                    tn = ((cos_sim <= t) & (y_masked == prox)).sum()
                    fn = ((cos_sim <= t) & (y_masked != prox)).sum()

                    data["tp_rates"].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                    data["tn_rates"].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
                    data["fp_rates"].append(fp / (fp + tn) if (fp + tn) > 0 else 0)
                    data["fn_rates"].append(fn / (fn + tp) if (fn + tp) > 0 else 0)

            intersect_thresholds = []
            intersect_values = []
            for prox, data in proxy_data.items():
                data["tp_rates"] = np.array(data["tp_rates"])
                data["tn_rates"] = np.array(data["tn_rates"])
                thresholds = np.array(thresholds)
                diff = np.abs(data["tp_rates"] - data["tn_rates"])
                intersect_idx = np.argmin(diff)
                intersect_threshold = thresholds[intersect_idx]
                intersect_value = data["tp_rates"][
                    intersect_idx
                ]  # or tn_rates[intersect_idx]
                data["intersect_threshold"] = intersect_threshold
                intersect_thresholds.append(intersect_threshold)
                data["intersect_value"] = intersect_value
                intersect_values.append(intersect_value)

            # Plot all the metrics
            fig, axes = plt.subplots(
                1, num_classes_eval, figsize=(10 * 8, num_classes_eval)
            )
            for ax, (prox, data) in zip(axes, proxy_data.items()):
                ax.set_aspect("equal")
                thresholds = np.array(thresholds)

                ax.set_title(f"Conditioned on {label_to_str[prox]} Proxy")

                ax.plot(
                    thresholds,
                    data["tp_rates"],
                    label="True Positive Rate",
                    color="blue",
                )
                ax.plot(
                    thresholds,
                    data["tn_rates"],
                    label="True Negative Rate",
                    color="orange",
                )

                # Add vertical line and label at intersection
                ax.axvline(
                    data["intersect_threshold"],
                    color="black",
                    linestyle="--",
                    linewidth=1,
                )
                ax.text(
                    data["intersect_threshold"],
                    0.05,  # slightly above bottom
                    f"Threshold = {data['intersect_threshold']:.2f}, TPR = {data['intersect_value']:.2f}",
                    rotation=90,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    backgroundcolor="white",
                    fontsize=9,
                )

                ax.set_xlabel("Cosine Similarity Threshold")
                ax.set_ylabel("Rate")
                ax.legend()

                # save thresholds to wm
                model.thresholds[prox] = data["intersect_threshold"]

            plt.tight_layout()

            wandb.log(
                {"eval/metric_plot": wandb.Image(fig), "num_updates": num_updates},
                step=num_updates,
            )
            wandb.log(
                {
                    "eval/intersect_threshold_variance": np.var(intersect_thresholds),
                    "eval/TPR_avg": np.mean(intersect_values),
                    "num_updates": num_updates,
                },
                step=num_updates,
            )
            plt.close()

            # Plot AUC curve
            fig, axes = plt.subplots(
                1, num_classes_eval, figsize=(10 * num_classes_eval, 8)
            )
            for ax, (prox, data) in zip(axes, proxy_data.items()):
                ax.set_aspect("equal")
                fp_rates = np.array(data["fp_rates"])
                tp_rates = np.array(data["tp_rates"])
                ax.plot(fp_rates, tp_rates, label="ROC Curve", color="blue")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title(f"Conditioned on {label_to_str[prox]} Proxy")
                ax.legend()
            plt.tight_layout()
            wandb.log(
                {"eval/roc_curve": wandb.Image(fig), "num_updates": num_updates},
                step=num_updates,
            )
            plt.close()

            # Plot cosine similarity distribution
            fig, axes = plt.subplots(
                1, num_classes_eval, figsize=(10 * num_classes_eval, 8)
            )

            used_labels = set()
            global_handles = []
            global_labels = []

            for ax, (prox, data) in zip(axes, proxy_data.items()):
                # ax.set_aspect("equal")
                cos_sim = -data[
                    "cos_sim"
                ]  # It's already negative cosine similarity, so we make it positive for plotting

                for idx, label in enumerate(classes_eval):
                    class_data = cos_sim[y == label]
                    if len(class_data) < 2:
                        continue
                    kde = gaussian_kde(class_data)
                    x_vals = np.linspace(-1, 1, 200)
                    y_vals = kde(x_vals)
                    color = class_to_colors[label]

                    plot_label = f"{label_to_str[label]}"
                    (line,) = ax.plot(
                        x_vals, y_vals, label=plot_label, color=color, alpha=0.7
                    )

                    if plot_label not in used_labels:
                        used_labels.add(plot_label)
                        global_handles.append(line)
                        global_labels.append(plot_label)

                ax.set_title(f"Conditioned on {label_to_str[prox]} Proxy")
                ax.set_xlabel("Cosine Similarity")
                ax.set_ylabel("Density")

            # Global legend above all subplots
            fig.legend(
                global_handles,
                global_labels,
                loc="upper center",
                ncol=len(global_labels),
                fontsize="x-large",
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            wandb.log(
                {
                    "eval/const_conditioned_cosine_sim": wandb.Image(fig),
                    "num_updates": num_updates,
                },
                step=num_updates,
            )
            plt.close()

            return cos_sim_proxies

        cos_sim_proxies = const_conditioned_plots(X, y)

        if nb_classes == 2:
            y_masked = copy.deepcopy(y)
            auc = roc_auc_score(
                y_true=y_masked,
                y_score=-cos_sim_fail.cpu().numpy(),
            )
            wandb.log({"eval/AUC": auc, "num_updates": num_updates}, step=num_updates)

        else:
            auc = roc_auc_score(
                y_true=losses.binarize(
                    labels_gt_masked.flatten(),
                    nb_classes=nb_classes,
                )
                .cpu()
                .numpy(),
                y_score=einops.rearrange(logits, "B T L -> (B T) L").cpu().numpy(),
                multi_class="ovo",
                average="macro",
            )
            wandb.log({"eval/AUC": auc, "num_updates": num_updates}, step=num_updates)

        for key, value in metrics.items():
            metrics[key] = np.mean(value)
        wandb.log(
            {
                "eval/Confusion matrix": wandb.plot.confusion_matrix(
                    preds=y_pred,
                    y_true=y_gt_masked,
                    class_names=list(label_to_str.values()),
                )
            },
            step=num_updates,
        )
        wandb_log = {f"eval/{k}": v for k, v in metrics.items()}
        wandb_log["num_updates"] = num_updates
        wandb.log(wandb_log, step=num_updates)

        # ---- Flatten and Prepare Data ----
        if len(X) == 0 or len(y) == 0:
            print("No data to visualize.")
            continue

        # print("Visualizing embeddings with UMAP...")

        # ---- UMAP Setup ----
        umap_input = np.concatenate(
            [X, criterion.proxies.detach().cpu().numpy()], axis=0
        )

        reducer = umap.UMAP(n_components=2, metric="cosine")
        umap_output = reducer.fit_transform(umap_input)

        X_umap = umap_output[:-nb_classes]
        proxies_umap = umap_output[-nb_classes:]

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

        # Plot proxies
        for i, proxy in enumerate(proxies_umap):
            plt.scatter(
                proxy[0],
                proxy[1],
                color=class_to_colors[i],
                marker="X",
                s=100,
                edgecolor="black",
                linewidth=1.2,
                label=f"Class {i} (proxy)",
                alpha=1.0,
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

        with torch.no_grad():
            model.proxies.copy_(criterion.proxies)

        if args.save_model:
            model_name = wandb_name
            save_name = f"../checkpoints_pa/encoder_{model_name}_strat.pth"
            best_save_name = f"../checkpoints_pa/best_encoder_{model_name}_strat.pth"

            torch.save(
                model.state_dict(),
                save_name,
            )
            tqdm.write(f"Model saved to {save_name}")

            if balanced_accuracy > best_eval:
                best_eval = balanced_accuracy
                print(f"New best at iter {i}, saving model to {best_save_name}.")
                torch.save(
                    model.state_dict(),
                    best_save_name,
                )
