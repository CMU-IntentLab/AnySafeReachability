import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from dino_decoder import VQVAE
from dino_models import VideoTransformer, normalize_acs, select_xyyaw_from_state
from einops import rearrange
from test_loader import SplitTrajectoryDataset
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import wandb

dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    wandb.init(project="dino-WM", name="WM")

    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    BS = 16  # Batch Size
    BL = 4  # Batch Length
    EVAL_H = 16
    H = 3

    hdf5_file = "/home/sunny/data/sweeper/train/consolidated.h5"
    hdf5_file_test = "/home/sunny/data/sweeper/test/consolidated.h5"

    expert_data = SplitTrajectoryDataset(
        hdf5_file, BL, split="train", num_test=0, provide_labels=True
    )
    expert_data_eval = SplitTrajectoryDataset(
        hdf5_file_test, BL, split="test", num_test=467, provide_labels=True
    )
    expert_data_imagine = SplitTrajectoryDataset(
        hdf5_file_test, 32, split="test", num_test=467
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
    transition.train()
    # Forward pass
    optimizer = AdamW(
        [
            {
                "params": transition.transformer.parameters(),
                "lr": 5e-5,
            },  # Dynamics Model
            {"params": transition.state_head.parameters(), "lr": 5e-5},  # State decoder
            {
                "params": transition.front_head.parameters(),
                "lr": 5e-5,
            },  # Front camera decoder
            # {
            #     "params": transition.wrist_head.parameters(),
            #     "lr": 5e-5,
            # },  # Wrist camera decoder
            {
                "params": transition.action_encoder.parameters(),
                "lr": 5e-4,
            },  # Action encoder
            {"params": [transition.pos_embedding], "lr": 5e-4},
            {"params": [transition.temp_embedding], "lr": 5e-4},
        ]
    )

    best_eval = float("inf")
    iters = []
    train_iter = 100000

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

        # data1: [B T N P] N - number of patches, P - patch size
        data1 = data["cam_zed_embd"].to(device)  # Front camera
        inputs1 = data1[:, :-1]  # Inputs are all but last frame
        output1 = data1[:, 1:]  # Outputs are all but first frame

        # data2 = data["cam_rs_embd"].to(device)  # Wrist camera
        # inputs2 = data2[:, :-1]
        # output2 = data2[:, 1:]

        # data_state: [B T S] S - state dimension
        data_state = select_xyyaw_from_state(
            data["state"].to(device)
        )  # Robot Joint States
        inputs_states = data_state[:, :-1]  # Inputs are all but last state
        output_state = data_state[:, 1:]  # Outputs are all but first state

        # data_acs: [B T A] A - action dimension
        data_acs = data["action"].to(device)  # Actions
        norm_acs = normalize_acs(data_acs, device)
        acs = norm_acs[:, :-1]  # Inputs are all but last action

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred1, pred_state, _, __, latent = transition(  # Forward pass
                inputs1, inputs_states, acs, return_latent=True
            )
            # pred1: [B (T-1) N P] - Predicted front camera embeddings
            # pred_state: [B (T-1) S] - Predicted robot joint states
            im1_loss_tf = (  # How different the predictions are from the actual data
                nn.MSELoss()(pred1, output1)
            )
            # im2_loss_tf = nn.MSELoss()(pred2, output2)
            state_loss_tf = nn.MSELoss()(pred_state, output_state)

            loss_tf = im1_loss_tf + state_loss_tf

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            # Imagine step
            detach_pred1 = pred1
            detach_pred_state = pred_state.detach()
            # Concatenate GT data point for t and predictions for t+1
            # inputs1_ar: [B 2 N P], states_ar: [B 2 S], acs_ar: [B 2 A]
            inputs1_ar = torch.cat([data1[:, [0]], detach_pred1[:, [0]]], dim=1)
            states_ar = torch.cat(
                [data_state[:, [0]], detach_pred_state[:, [0]]], dim=1
            )
            acs_ar = norm_acs[:, [0, 1]]

            # Forward pass through transition model
            # pred1_ar: [B 2 N P], pred_state_ar: [B 2 S]
            # Predictions represent t+1, and t+2
            pred1_ar, pred_state_ar, _, __ = transition(inputs1_ar, states_ar, acs_ar)
            output1_ar = data1[:, 2]  # GT for t+2
            output_state_ar = data_state[:, 2]  # GT for t+2
            im1_loss_ar = nn.MSELoss()(pred1_ar[:, 1], output1_ar)
            state_loss_ar = nn.MSELoss()(pred_state_ar[:, 1], output_state_ar)
            loss_ar = im1_loss_ar + state_loss_ar

        loss = loss_tf + loss_ar * 0.5
        # loss = loss_tf

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss = loss.item()
        print(
            f"\rIter {i}, TF Loss: {loss_tf:.4f}, AR loss:{loss_ar} front Loss: {im1_loss_tf.item():.4f}, state Loss: {state_loss_tf.item():.4f}",
            end="",
            flush=True,
        )
        print(
            f"\rIter {i}, TF Loss: {loss_tf:.4f}, front Loss: {im1_loss_tf.item():.4f}, state Loss: {state_loss_tf.item():.4f}",
            end="",
            flush=True,
        )
        wandb.log({"train_loss": loss_tf, "train_loss_ar": loss_ar})
        # eval
        if (i) % 1000 == 0:
            iters.append(i)
            eval_data = next(expert_loader_imagine)
            transition.eval()
            with torch.no_grad():
                # H: 3, EVAL_H: 16
                # eval_data1: [1 32 N P]
                eval_data1 = eval_data["cam_zed_embd"].to(device)
                # inputs1: [1 H N P]
                inputs1 = eval_data1[[0], :H].to(device)

                # all_acs: [1 32 A]
                all_acs = eval_data["action"][[0]].to(device)
                all_acs = normalize_acs(all_acs, device)

                # acs: [1 H A]
                acs = eval_data["action"][[0], :H].to(device)
                acs = normalize_acs(acs, device)

                # inputs_states: [1 H S]
                inputs_states = select_xyyaw_from_state(
                    eval_data["state"][[0], :H].to(device)
                )
                im1s = (
                    eval_data["agentview_image"][[0], :H].squeeze().to(device) / 255.0
                )
                for k in range(EVAL_H - H):
                    pred1, pred_state, _, ___ = transition(inputs1, inputs_states, acs)

                    # pred_latent = pred1[:, [-1]]
                    pred_ims, _ = decoder(pred1[:, [-1]])

                    pred_ims = rearrange(pred_ims, "t c h w -> t h w c", t=1)
                    pred_im1 = pred_ims

                    im1s = torch.cat([im1s, pred_im1], dim=0)

                    # getting next inputs
                    acs = torch.cat(
                        [acs[[0], 1:], all_acs[0, H + k].unsqueeze(0).unsqueeze(0)],
                        dim=1,
                    )
                    inputs1 = torch.cat(
                        [inputs1[[0], 1:], pred1[:, -1].unsqueeze(1)], dim=1
                    )
                    inputs_states = states = torch.cat(
                        [inputs_states[[0], 1:], pred_state[:, -1].unsqueeze(1)], dim=1
                    )

                gt_im1 = eval_data["agentview_image"][[0], :EVAL_H].squeeze().to(device)

                gt_imgs = torch.cat([gt_im1], dim=-2) / 255.0  # [T H W C]
                pred_imgs = torch.cat([im1s], dim=-2)

                vid = torch.cat([gt_imgs, pred_imgs], dim=-2)
                vid = vid.detach().cpu().numpy()
                vid = (vid * 255).clip(0, 255).astype(np.uint8)
                vid = rearrange(vid, "t h w c -> t c h w")
                wandb.log({"video": wandb.Video(vid, fps=20, format="mp4")})

                # done logging video

                eval_data = next(expert_loader_eval)
                data1 = eval_data["cam_zed_embd"].to(device)

                inputs1 = data1[:, :-1]
                output1 = data1[:, 1:]

                data_state = select_xyyaw_from_state(eval_data["state"].to(device))
                states = data_state[:, :-1]
                output_state = data_state[:, 1:]

                data_acs = eval_data["action"].to(device)
                # import ipdb

                # ipdb.set_trace()
                data_acs = normalize_acs(data_acs, device)
                acs = data_acs[:, :-1]
                pred1, pred_state, _, __, latent = transition(
                    inputs1, states, acs, return_latent=True
                )

                pred_latent = pred1[:, [H - 1]]
                pred_ims, _ = decoder(pred_latent)
                pred_im1 = pred_ims
                pred_im1 = pred_im1[0].permute(1, 2, 0).detach().cpu().numpy()
                im1 = eval_data["agentview_image"][0, H].numpy()
                im1_loss = nn.MSELoss()(pred1, output1)
                state_loss = nn.MSELoss()(pred_state, output_state)
                loss = im1_loss + state_loss
            print()
            print(
                f"\rIter {i}, Eval Loss: {loss.item():.4f}, front Loss: {im1_loss.item():.4f}, state Loss: {state_loss.item():.4f}"
            )

            torch.save(transition.state_dict(), f"checkpoints/testing_iter{i}.pth")

            if loss < best_eval:
                best_eval = loss
                torch.save(transition.state_dict(), "checkpoints/best_testing.pth")

            transition.train()
            wandb.log(
                {
                    "eval_loss": loss.item(),
                    "front_loss": im1_loss.item(),
                    "state_loss": state_loss.item(),
                    "pred_front": wandb.Image(pred_im1),
                    "front": wandb.Image(im1),
                }
            )

    plt.legend()
    plt.savefig("training curve.png")
