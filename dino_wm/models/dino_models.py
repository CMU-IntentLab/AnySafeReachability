# Based on DINO-WM https://arxiv.org/abs/2411.04983


from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.spatial.transform import Rotation
from torch import nn
from torchvision import transforms


def batch_quat_to_rotvec(quaternions):
    """
    Convert a batch of quaternions to axis-angle using PyTorch and scipy.

    Args:
        quaternions (torch.Tensor): A tensor of shape (N, 4), where each quaternion is (w, x, y, z).

    Returns:
        axes (torch.Tensor): A tensor of shape (N, 3), representing the rotation axes.
        angles (torch.Tensor): A tensor of shape (N,), representing the rotation angles in radians.
    """
    # Convert PyTorch tensor to NumPy array
    quaternions_np = quaternions.cpu().numpy()

    # Use scipy for the quaternion-to-axis-angle conversion
    r = Rotation.from_quat(quaternions_np)
    rotvecs = r.as_rotvec()
    return rotvecs


def batch_rotvec_to_quat(rotvecs):
    """
    Convert a batch of quaternions to axis-angle using PyTorch and scipy.

    Args:
        quaternions (torch.Tensor): A tensor of shape (N, 4), where each quaternion is (w, x, y, z).

    Returns:
        axes (torch.Tensor): A tensor of shape (N, 3), representing the rotation axes.
        angles (torch.Tensor): A tensor of shape (N,), representing the rotation angles in radians.
    """
    # Convert PyTorch tensor to NumPy array
    rotvecs_np = rotvecs.cpu().numpy()

    # Use scipy for the quaternion-to-axis-angle conversion
    r = Rotation.from_rotvec(rotvecs_np)
    quaternions = r.as_quat()
    return quaternions


def select_xyyaw_from_state(state):
    """Select x, y, yaw from the state."""
    # Assuming state is of shape [B, T, 8] where last dimension is [x, y, z, qx, qy, qz, qw, gripper]
    return state[..., [0, 1, 5]]


def normalize_acs(acs, device="cuda:0"):
    assert acs.shape[-1] == 7, "Actions should have 7 dimensions"
    max_ac = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.8750, 1.0]).to(device)
    min_ac = torch.tensor([-1.0, -1.0, 0.0, 0.0, 0.0, -0.8967, 1.0]).to(device)

    norm_acs = (acs - min_ac) / (max_ac - min_ac)

    # Only get (x, y, yaw)
    norm_acs = norm_acs[..., [0, 1, 5]]

    return norm_acs


def unnormalize_acs(acs, device="cuda:0"):
    max_ac = torch.tensor(
        [0.89928758, 0.71893158, 0.69869383, 0.32456627, 0.51343921, 0.28401476, 1.0]
    ).to(device)
    min_ac = torch.tensor(
        [-0.78933347, -1.0, -0.95038878, -0.3243517, -0.30636792, -0.30071826, -1.0]
    ).to(device)

    acs = (acs * (max_ac - min_ac)) + min_ac

    return acs


class ResidualBlock2(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock2, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Decoder(nn.Module):
    def __init__(self, in_channels=384, out_channels=3):
        super(Decoder, self).__init__()

        # Two residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock2(in_channels),
            ResidualBlock2(in_channels),
            ResidualBlock2(in_channels),
            ResidualBlock2(in_channels),
        )

        # Three transposed convolutions to go from 16x16 to 224x224
        self.transposed_convs = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, in_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels // 2, in_channels // 4, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            # New intermediate layer
            nn.ConvTranspose2d(
                in_channels // 4, in_channels // 8, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            # New additional layer with no change in resolution
            nn.ConvTranspose2d(
                in_channels // 8, in_channels // 8, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels // 8, out_channels, kernel_size=3, stride=1, padding=1
            ),
        )

        self.resize_transform = transforms.Resize((224, 224))

    def forward(self, x):
        x = x.view(
            -1, 16, 16, 384
        )  # Reshape to (16, 16, 384) where 16x16 is the spatial grid
        x = x.permute(0, 3, 1, 2)  # Pass through residual blocks
        x = self.residual_blocks(x)
        # Pass through transposed convolutions
        x = self.transposed_convs(x)
        x = self.resize_transform(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        num_frames: int = 2,
        patches_per_frame: int = 256,
    ):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

        # Register buffer instead of creating mask every forward pass
        mask = self._create_causal_mask(num_frames, patches_per_frame)
        self.register_buffer("mask", mask)

    def _create_causal_mask(
        self, num_frames: int, patches_per_frame: int
    ) -> torch.Tensor:
        total_patches = num_frames * patches_per_frame
        mask = torch.zeros(total_patches, total_patches)

        for i in range(num_frames):
            start_idx = i * patches_per_frame
            end_idx = (i + 1) * patches_per_frame

            # Allow attention within current frame
            mask[start_idx:end_idx, start_idx:end_idx] = 1

            # Allow attention to previous frames
            if i > 0:
                mask[start_idx:end_idx, :start_idx] = 1

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use registered mask buffer
        mask = self.mask[:seq_len, :seq_len]
        dots = dots.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        num_frames: int = 2,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads, dim_head, dropout, num_frames)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class VideoTransformer(nn.Module):
    def __init__(
        self,
        *,
        image_size: Tuple[int, int],
        dim: int,  # DINO feature dimension
        ac_dim: int,  # Action embedding dimension
        state_dim: int,  # State dimension
        depth: int,
        heads: int,
        mlp_dim: int,
        num_frames: int = 2,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        device: str = "cuda",
    ):
        super().__init__()

        self.device = device
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(
            device
        )

        # Improved action embedding
        self.action_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, ac_dim),
            nn.LayerNorm(ac_dim),
        ).to(device)

        total_dim = dim + ac_dim + state_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, total_dim) * 0.02)
        self.temp_embedding = nn.Parameter(torch.randn(1, num_frames, total_dim) * 0.02)

        self.dropout = nn.Dropout(emb_dropout)

        # Use TransformerBlock instead of separate components
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    total_dim, heads, dim_head, mlp_dim, dropout, num_frames
                )
                for _ in range(depth)
            ]
        )

        # Separate prediction heads
        self.front_head = nn.Sequential(
            LayerNorm(total_dim),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, dim),
        )

        self.state_head = nn.Sequential(
            LayerNorm(total_dim),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, state_dim),
        )

        self.xy_state_head = nn.Sequential(
            LayerNorm(total_dim),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, 2),
        )

        semantic_dim = dim + state_dim
        self.semantic_encoder = nn.Sequential(
            LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, 512),
        )

        self.margin_head = nn.Sequential(
            LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, 1),
        )

    def forward(
        self,
        video1: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        return_latent: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the video transformer.
        Args:
            video1 (torch.Tensor): Input video tensor of shape (B, T, N, P) where
                B is batch size, T is number of frames, N is number of patches, and P is patch size.
            states (torch.Tensor): Robot joint states of shape (B, T, S) where S is the state dimension.
            actions (torch.Tensor): Actions of shape (B, T, A) where A is the action dimension.
            return_latent (bool): Whether to return latent features.
        Returns:
            Tuple containing:
                - pred1 (torch.Tensor): Predictions from the front head.
                - state_preds (torch.Tensor): State predictions.
                - failure_preds (torch.Tensor): Failure predictions.
                - semantic_features (torch.Tensor): Encoded semantic features.
                - x (torch.Tensor, optional): Latent features if return_latent is True.
        """
        # x: [B (T-1) N (P + A + S)]
        x = self.forward_features(video1, states, actions)

        # Generate predictions
        pred1 = self.front_head(x)  # [B (T-1) N P]
        state_preds = self.state_pred(x)  # [B (T-1) S]
        # fail_preds = self.fail_pred(x)  # [B (T-1) 1]
        fail_preds = self.fail_pred(inp1=pred1, state=state_preds)  # [B (T-1) 1]

        semantic_features = (  # [ B (T-1) E ] E - embedding dimension
            self.semantic_embed(inp1=pred1, state=state_preds)
        )

        if return_latent:
            return (
                pred1,
                state_preds,
                fail_preds,
                semantic_features,
                x,  # Return latent features
            )
        else:
            return pred1, state_preds, fail_preds, semantic_features

    def forward_features(
        self,
        video1: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode actions
        action_embeddings = (
            self.action_encoder(actions).unsqueeze(2).expand(-1, -1, 256, -1)
        )
        state_embeddings = states.unsqueeze(2).expand(-1, -1, 256, -1)

        # Combine features
        batch_size, num_frames, _, _ = video1.shape

        # x [B (T-1) N (P + A + S)]
        x = torch.cat((video1, action_embeddings, state_embeddings), dim=3)
        # Add positional embeddings
        x = x + self.pos_embedding
        x = x + self.temp_embedding[:, :num_frames].unsqueeze(2)

        # Reshape for transformer
        # x: [B (T-1) N (P + A + S)] -> [B (T-1 * N) (P + A + S)]
        x = rearrange(x, "b s n d -> b (s n) d")
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.transformer:
            x = block(x)

        # Reshape back
        # x = [B (T-1 * N) (P + A + S)] -> [B (T-1) N (P + A + S)]
        x = rearrange(x, "b (s n) d -> b s n d", s=num_frames)
        return x

    # def fail_pred(self, features):
    #     # features = torch.mean(features, dim=-2)
    #     fail_preds = self.margin_head(features)
    #     fail_preds = torch.mean(fail_preds, dim=2)  # Average over patches
    #     return fail_preds

    def fail_pred(self, inp1, state):
        features = torch.cat(  # [B T N (P + S)]
            (
                inp1,  # [B T N P]
                einops.repeat(state, "b t s -> b t n s", n=inp1.shape[2]),  # [B T N S]
            ),
            dim=-1,
        )
        fail_preds = self.margin_head(features)
        fail_preds = torch.mean(fail_preds, dim=2)  # Average over patches
        return fail_preds

    def semantic_embed(self, inp1, state):
        """
        Combine features from two inputs and state, then pass through semantic encoder.
        Args:
            inp1 (torch.Tensor): Features from the first input (e.g., front camera).
            state (torch.Tensor): State features.
        Returns:
            torch.Tensor: Encoded semantic features.
        """
        features = torch.cat(  # [B T N (P + S)]
            (
                inp1,  # [B T N P]
                einops.repeat(state, "b t s -> b t n s", n=inp1.shape[2]),  # [B T N S]
            ),
            dim=-1,
        )
        # features = inp1
        # semantic_features: [B T N E]
        # features = torch.norm(features, dim=-2)  # Average over patches
        # features = torch.mean(features, dim=-2)  # Average over patches
        semantic_features = self.semantic_encoder(features)
        # Average over patches
        # semantic_features: [B T E]
        semantic_features = torch.mean(semantic_features, dim=2)
        # semantic_features = F.normalize(semantic_features, p=2, dim=-1)
        return semantic_features

    def state_pred(self, features):
        state_preds = self.state_head(features)
        state_preds = torch.mean(state_preds, dim=2)  # Average over patches
        return state_preds

    @torch.no_grad()
    def get_dino_features(self, video: torch.Tensor) -> torch.Tensor:
        """Extract DINO features from video frames."""
        b, f, c, h, w = video.shape
        video = video.view(b * f, c, h, w)
        features = self.dino.forward_features(video)["x_norm_patchtokens"]
        return features.view(b, f, -1, features.shape[-1])


import einops
import torch
import torch.nn as nn


def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def horizontal_forward(network, x, input_shape=(-1,), output_shape=(-1,)):
    batch_with_horizon_shape = x.shape[: -len(input_shape)]
    if not batch_with_horizon_shape:
        batch_with_horizon_shape = (1,)
    x = x.reshape(-1, *input_shape)
    x = network(x)
    x = x.reshape(*batch_with_horizon_shape, *output_shape)
    return x


def create_normal_dist(
    x,
    std=None,
    mean_scale=1,
    init_std=0,
    min_std=0.1,
    activation=None,
    event_shape=None,
):
    if std == None:
        mean, std = torch.chunk(x, 2, -1)
        mean = mean / mean_scale
        if activation:
            mean = activation(mean)
        mean = mean_scale * mean
        std = F.softplus(std + init_std) + min_std
    else:
        mean = x
    dist = torch.distributions.Normal(mean, std)
    if event_shape:
        dist = torch.distributions.Independent(dist, event_shape)
    return dist


class TransposedConvDecoder(nn.Module):
    def __init__(
        self,
        observation_shape=(3, 224, 224),
        emb_dim=512,
        activation=nn.ReLU,
        depth=64,
        kernel_size=5,
        stride=3,
    ):
        super().__init__()

        activation = activation()
        self.observation_shape = observation_shape
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.emb_dim = emb_dim

        self.network = nn.Sequential(
            nn.Linear(emb_dim, self.depth * 32),
            nn.Unflatten(1, (self.depth * 32, 1)),
            nn.Unflatten(2, (1, 1)),
            nn.ConvTranspose2d(
                self.depth * 32,
                self.depth * 8,
                self.kernel_size,
                self.stride,
                padding=1,
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 8, self.depth * 4, self.kernel_size, self.stride, padding=1
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 4, self.depth * 2, self.kernel_size, self.stride, padding=1
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 2, self.depth * 1, self.kernel_size, self.stride, padding=1
            ),
            activation,
            nn.ConvTranspose2d(
                self.depth * 1,
                self.observation_shape[0],
                self.kernel_size,
                self.stride,
                padding=1,
            ),
            nn.Upsample(
                size=(observation_shape[1], observation_shape[2]),
                mode="bilinear",
                align_corners=False,
            ),
        )
        self.network.apply(initialize_weights)

    def forward(self, posterior):
        x = horizontal_forward(
            self.network,
            posterior,
            input_shape=[self.emb_dim],
            output_shape=self.observation_shape,
        )
        dist = create_normal_dist(x, std=1, event_shape=len(self.observation_shape))
        img = dist.mean.squeeze(2)
        img = einops.rearrange(img, "b t c h w -> (b t) c h w")
        return img, torch.zeros(1).to(posterior.device)  # dummy placeholder
