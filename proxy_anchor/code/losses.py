import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing

    if nb_classes == 2:
        # Manual one-hot encoding for binary case
        T = np.eye(2)[T.astype(int)]
    else:
        class_labels = range(nb_classes)
        T = sklearn.preprocessing.label_binarize(T, classes=class_labels)

    T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        """
        Forward pass for Proxy Anchor loss calculation.
        Args:
            X (torch.Tensor): Input embeddings of shape (B, T, Z) or (B * T, Z).
            T (torch.Tensor): Target labels of shape (B, T) or (B * T,).
            U (torch.Tensor, optional): Unlabelled data of shape (B, T, Z).

        Returns:
            torch.Tensor: Computed Proxy Anchor loss.
        """
        P = self.proxies  # [N, Z]
        if X.ndim != 2:  # [(B, T) Z]
            X = einops.rearrange(
                X, "B T Z -> (B T) Z"
            )  # Ensure X is in the correct shape
        if T.ndim != 1:  # [(B, T)]
            T = einops.rearrange(T, "B T -> (B T)")  # Ensure T is in the correct shape

        cos_labeled = F.linear(  # Calculate cosine similarity [B, N]
            l2_norm(X), l2_norm(P)
        )

        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)  # [B, N]
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos_labeled - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos_labeled + self.mrg))

        pos_exp_weighted = pos_exp * P_one_hot
        neg_exp_weighted = neg_exp * N_one_hot

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1
        )  # The set of positive proxies of data in the batch

        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = pos_exp_weighted.sum(dim=0)
        N_sim_sum = neg_exp_weighted.sum(dim=0)

        pos_term = (
            torch.log(1 + P_sim_sum).sum() / num_valid_proxies
            if num_valid_proxies > 0
            else 0
        )
        # pos_term = torch.log(1 + P_sim_sum).sum() / self.nb_classes
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        neg_term *= 1.0
        loss = pos_term + neg_term

        return (
            loss,
            pos_term,
            neg_term,
        )


# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(
            num_classes=self.nb_classes,
            embedding_size=self.sz_embed,
            softmax_scale=self.scale,
        ).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class MultiSimilarityLoss(torch.nn.Module):
    def __init__(
        self,
    ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50

        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(
            self.scale_pos, self.scale_neg, self.thresh
        )

    def forward(self, X, T):
        hard_pairs = self.miner(X, T)
        loss = self.loss_func(X, T, hard_pairs)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin)

    def forward(self, X, T):
        loss = self.loss_func(X, T)
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets="semihard")
        self.loss_func = losses.TripletMarginLoss(margin=self.margin)

    def forward(self, X, T):
        hard_pairs = self.miner(X, T)
        loss = self.loss_func(X, T, hard_pairs)
        return loss


class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(
            # l2_reg_weight=self.l2_reg,
            # normalize_embeddings=False
        )

    def forward(self, X, T):
        loss = self.loss_func(X, T)
        return loss


class PrivilegedTeacherForcingLoss(nn.Module):
    def __init__(self, mapping_fn):
        torch.nn.Module.__init__(self)
        self.mapping_fn = mapping_fn

    def forward(self, X, T):
        # X: [B* T, Z]
        # T: [B* T]
        # mapping_fn: maps distance to cosine similarity
        sem_norm = F.normalize(X, p=2, dim=1)  # [B* T, Z]
        cos_sim = sem_norm @ sem_norm.T  # [B* T, B* T]

        diff = T.unsqueeze(0) - T.unsqueeze(1)  # [B, B, 2]
        dists = torch.norm(diff, dim=2)
        labels = self.mapping_fn(dists)  # [B, B]

        # loss = F.mse_loss(cos_sim, labels)
        loss = torch.nn.SmoothL1Loss()(cos_sim, labels)
        return loss
