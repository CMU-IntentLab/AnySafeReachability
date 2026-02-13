'''
Used for training latent safe classifiers.
'''

import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from models.dino_decoder import VQVAE
from models.dino_models import VideoTransformer, normalize_acs, select_xyyaw_from_state
from einops import rearrange
from sklearn.metrics import balanced_accuracy_score
from utils.test_loader import SplitTrajectoryDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import wandb
from utils import load_state_dict_flexible

dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")

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
        transforms.CenterCrop(518),  # should be multiple of model patch_size
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.2),
    ]
)

DINO_transform = transforms.Compose(
    [
        transforms.Resize(224),
        # transforms.CenterCrop(224), #should be multiple of model patch_size
        transforms.ToTensor(),
    ]
)
norm_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

# Class is defined as:
# If you are in circle centered on left third of screen, class 0
# If you are in circle centered on middle third of screen, class 1
# If you are in circle centered on right third of screen, class 2
# If you are outside all circles, class 3

circle_centers = torch.tensor(
    [
        [224 // 6, 2 * 224 // 3],
        [224 // 2, 2 * 224 // 3],
        [224 * 5 // 6, 2 * 224 // 3],
    ],
    device="cuda:0",
)
radius = 30
nb_classes = 4  # including the "safe" class


def get_class_from_xy(labels):
    assert labels.shape[-1] == 2, "Labels should have shape (..., 2)"

    # Compute pairwise distances: (..., num_centers)
    distances = torch.norm(labels[..., None, :] - circle_centers[None, ...], dim=-1)

    # Get nearest center index
    class_labels = torch.argmin(distances, dim=-1)

    # Mark "outside circle" as class 3
    min_distances = torch.min(distances, dim=-1).values
    class_labels = class_labels.masked_fill(min_distances > radius, 3)

    return class_labels


def fail_loss(pred, fail_data):
    safe_data = torch.where(fail_data != 1.0)
    unsafe_data = torch.where(fail_data == 1.0)

    pos = pred[safe_data]
    neg = pred[unsafe_data]

    gamma = 0.75
    lx_loss = (
        (1 / pos.size(0)) * torch.sum(torch.relu(gamma - pos))
        if pos.size(0) > 0
        else 0.0
    )  # penalizes safe for being negative
    lx_loss += (
        (1 / neg.size(0)) * torch.sum(torch.relu(gamma + neg))
        if neg.size(0) > 0
        else 0.0
    )  # penalizes unsafe for being positive

    return lx_loss


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Please provide the class number as an argument."
    _class = sys.argv[1]
    assert _class.isdigit(), "Class number must be an integer."
    _class = int(_class)
    assert 0 <= _class < nb_classes, (
        f"Class number must be between 0 and {nb_classes - 1}."
    )
    print(f"Training for class: {_class}")
    wandb.init(project="Latent Safe", name=f"Cl`assifier Class {_class} DINO")

    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    BS = 16
    BL = 4
    EVAL_H = 16
    H = 3

    hdf5_file = "path/to/train/dataset/consolidated.h5"
    hdf5_file_test = "/data/sweeper/test/consolidated.h5"

    expert_data = SplitTrajectoryDataset(
        hdf5_file, BL, split="train", num_test=0, only_pass_labeled_examples=True
    )
    expert_data_eval = SplitTrajectoryDataset(
        hdf5_file, BL, split="test", num_test=5, only_pass_labeled_examples=True
    )
    expert_data_imagine = SplitTrajectoryDataset(
        hdf5_file_test, 32, split="test", num_test=5, only_pass_labeled_examples=True
    )

    expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
    expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
    expert_loader_imagine = iter(
        DataLoader(expert_data_imagine, batch_size=1, shuffle=True)
    )

    device = "cuda:0"

    decoder = VQVAE().to(device)
    decoder.load_state_dict(torch.load("checkpoints/testing_decoder.pth"))
    decoder.eval()

    transition = VideoTransformer(
        image_size=(224, 224),
        dim=384,  # DINO feature dimension
        ac_dim=10,  # Action embedding dimension
        state_dim=3,  # State dimension
        depth=6,
        heads=16,
        mlp_dim=2048,
        num_frames=BL - 1,
        dropout=0.1,
    ).to(device)
    load_state_dict_flexible(
        transition,
        "dino_wm/checkpoints/best_testing.pth",
    )
    # transition.load_state_dict(torch.load("checkpoints/best_testing.pth"), strict=False)

    for name, param in transition.named_parameters():
        param.requires_grad = name.startswith("margin_head")

    data = next(expert_loader)

    data1 = data["cam_zed_embd"].to(device)
    # data2 = data["cam_rs_embd"].to(device)
    inputs1 = data1[:, :-1]
    output1 = data1[:, 1:]

    # inputs2 = data2[:, :-1]
    # output2 = data2[:, 1:]

    data_state = select_xyyaw_from_state(data["state"].to(device))
    states = data_state[:, :-1]
    output_state = data_state[:, 1:]

    data_acs = data["action"].to(device)
    acs = data_acs[:, :-1]
    acs = normalize_acs(acs, device)

    # Forward pass
    optimizer = AdamW(
        [
            {"params": transition.margin_head.parameters(), "lr": 5e-5},
        ]
    )

    best_eval = float("inf")
    best_fail = float("inf")
    iters = []
    train_iter = 10000

    for i in tqdm(range(train_iter), desc="Training", unit="iter"):
        if i % len(expert_loader) == 0:
            expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
        if i % len(expert_loader_eval) == 0:
            expert_loader_eval = iter(
                DataLoader(expert_data_eval, batch_size=BS, shuffle=True)
            )
        if i % len(expert_loader_imagine) == 0:
            expert_loader_imagine = iter(
                DataLoader(expert_data_imagine, batch_size=1, shuffle=True)
            )

        data = next(expert_loader)
        data1 = data["cam_zed_embd"].to(device)
        inputs1 = data1[:, :-1]
        output1 = data1[:, 1:]

        data_state = select_xyyaw_from_state(data["state"].to(device))
        states = data_state[:, :-1]
        output_state = data_state[:, 1:]

        data_acs = data["action"].to(device)
        acs = data_acs[:, :-1]
        acs = normalize_acs(acs, device)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred1, pred_state, __, __, latent = transition(
                inputs1, states, acs, return_latent=True
            )
            pred_fail = transition.fail_pred(inp1=output1, state=output_state)
            # gt_labels: [BS T]
            gt_labels = get_class_from_xy(data["failure"][:, 1:].to(device))
            # mask: [BS]
            mask = (gt_labels[:, -1] != -1.0).squeeze()
            gt_labels = (gt_labels[mask] == _class).float()  # [BS]
            pred_fail = pred_fail[mask]
            # Unsafe = 1.0, Safe = 0.0
            loss = fail_loss(pred_fail.squeeze(), gt_labels)

        pred_labels = (pred_fail < 0).float()
        true_labels = gt_labels
        # correct = (pred_labels.to(true_labels.device) == true_labels).float().sum()
        # accuracy = correct / (true_labels.shape[0] * true_labels.shape[1])
        balanced_accuracy = balanced_accuracy_score(
            true_labels.flatten().cpu().numpy().astype(int),
            pred_labels.flatten().cpu().numpy().astype(int),
        )
        wandb.log({"train_balanced_accuracy": balanced_accuracy})

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss = loss.item()
        wandb.log({"train_loss": train_loss})
        print(
            f"\rIter {i}, Train Loss: {train_loss:.4f}, failure Loss: {loss.item():.4f}",
            end="",
            flush=True,
        )

        if (i) % 500 == 0:
            iters.append(i)
            eval_data = next(expert_loader_imagine)
            transition.eval()
            with torch.no_grad():
                eval_data1 = eval_data["cam_zed_embd"].to(device)
                # eval_data2 = eval_data["cam_rs_embd"].to(device)

                inputs1 = eval_data1[[0], :H].to(device)
                # inputs2 = eval_data2[[0], :H].to(device)
                all_acs = eval_data["action"][[0]].to(device)
                all_acs = normalize_acs(all_acs, device)
                acs = eval_data["action"][[0], :H].to(device)
                acs = normalize_acs(acs, device)
                states = select_xyyaw_from_state(eval_data["state"][[0], :H].to(device))
                im1s = (
                    eval_data["agentview_image"][[0], :H].squeeze().to(device) / 255.0
                )
                # im2s = (
                #     eval_data["robot0_eye_in_hand_image"][[0], :H].squeeze().to(device)
                #     / 255.0
                # )
                for k in range(EVAL_H - H):
                    pred1, pred_state, __, __ = transition(inputs1, states, acs)
                    pred_fail = transition.fail_pred(inp1=inputs1, state=states)
                    pred_latent = pred1[:, [-1]]  # .squeeze()
                    pred_ims, _ = decoder(pred_latent)

                    pred_ims = rearrange(pred_ims, "(b t) c h w -> b t c h w", t=1)
                    pred_im1 = pred_ims

                    pred_im1 = pred_im1[0].permute(0, 2, 3, 1).detach()
                    pred_fail = pred_fail[:, -1]

                    # if pred_fail < 0:
                    #     pred_im1[:, :, :, 0] *= 2

                    im1s = torch.cat([im1s, pred_im1], dim=0)

                    # getting next inputs
                    acs = torch.cat(
                        [acs[[0], 1:], all_acs[0, H + k].unsqueeze(0).unsqueeze(0)],
                        dim=1,
                    )
                    inputs1 = torch.cat(
                        [inputs1[[0], 1:], pred1[:, -1].unsqueeze(1)], dim=1
                    )
                    states = torch.cat(
                        [states[[0], 1:], pred_state[:, -1].unsqueeze(1)], dim=1
                    )

                gt_im1 = eval_data["agentview_image"][[0], :EVAL_H].squeeze().to(device)
                # for j in range(EVAL_H):
                #     if gt_fail[j] > 0:
                #         gt_im1[j, :, :, 0] *= 2

                gt_imgs = torch.cat([gt_im1], dim=-3) / 255.0
                pred_imgs = torch.cat([im1s], dim=-3)

                vid = torch.cat([gt_imgs, pred_imgs], dim=-2)
                vid = vid[H:]

                vid = rearrange(vid, "t h w c -> t c h w")
                vid = vid.detach().cpu().numpy()
                vid = (vid * 255).clip(0, 255).astype(np.uint8)

                wandb.log({"video": wandb.Video(vid, fps=20)})

                # done logging video

                eval_data = next(expert_loader_eval)

                data1 = eval_data["cam_zed_embd"].to(device)

                inputs1 = data1[:, :-1]
                output1 = data1[:, 1:]

                data_state = select_xyyaw_from_state(eval_data["state"].to(device))
                states = data_state[:, :-1]
                output_state = data_state[:, 1:]

                data_acs = eval_data["action"].to(device)
                acs = data_acs[:, :-1]
                acs = normalize_acs(acs, device)

                pred1, pred_state, __, __ = transition(inputs1, states, acs)
                pred_fail = transition.fail_pred(inp1=output1, state=output_state)

                gt_labels = get_class_from_xy(eval_data["failure"][:, 1:].to(device))
                mask = (gt_labels[:, -1] != -1.0).squeeze()

                gt_labels = (gt_labels[mask] == _class).float()  # [BS]
                pred_fail = pred_fail[mask]

                # 1 for unsafe, 0 for safe
                loss = fail_loss(pred_fail, gt_labels)

            pred_labels = (pred_fail < 0).float()
            true_labels = gt_labels
            # correct = (pred_labels.to(true_labels.device) == true_labels).float().sum()
            # accuracy = correct / (true_labels.shape[0] * true_labels.shape[1])
            balanced_accuracy = balanced_accuracy_score(
                true_labels.flatten().cpu().numpy().astype(int),
                pred_labels.flatten().cpu().numpy().astype(int),
            )
            wandb.log({"eval_balanced_accuracy": balanced_accuracy})

            print(f"\rIter {i}, Eval Loss: {loss.item():.4f},")

            torch.save(
                transition.state_dict(),
                f"checkpoints_latent_safe/class_{_class}_classifier_dino.pth",
            )

            if loss < best_eval:
                best_eval = loss
                print(f"New best at iter {i}, saving model.")
                torch.save(
                    transition.state_dict(),
                    f"checkpoints_latent_safe/class_{_class}_best_classifier_dino.pth",
                )

            transition.train()
            wandb.log({"eval_loss": loss.item(), "failure_loss": loss.item()})

    plt.legend()
    plt.savefig("training curve.png")
