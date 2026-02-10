import warnings

import numpy as np
import torch
from scipy.stats import entropy, wasserstein_distance


def custom_formatwarning(msg, category, filename, lineno, line=None):
    return f"\033[93m{category.__name__}: {msg}\033[0m\n"  # yellow text


warnings.formatwarning = custom_formatwarning


def load_state_dict_flexible(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # If checkpoint is a wrapper (like {"state_dict": ...})
    state_dict = checkpoint.get("state_dict", checkpoint)

    model_state_dict = model.state_dict()

    loaded_state_dict = {}

    for name, param in state_dict.items():
        if name not in model_state_dict:
            warnings.warn(f"Skipping '{name}' as it is not in the model.")
            continue

        if model_state_dict[name].shape != param.shape:
            warnings.warn(
                f"Shape mismatch for '{name}': "
                f"model={model_state_dict[name].shape}, "
                f"checkpoint={param.shape}. Skipping."
            )
            continue

        loaded_state_dict[name] = param

    # Load only the matching parameters
    missing_keys, unexpected_keys = model.load_state_dict(
        loaded_state_dict, strict=False
    )

    if missing_keys:
        warnings.warn(f"Missing keys in checkpoint (not loaded): {missing_keys}")
    if unexpected_keys:
        warnings.warn(f"Unexpected keys in checkpoint (ignored): {unexpected_keys}")


def kl_divergence_kde(p_kde, q_kde, num_samples=10_000):
    """
    Approximate KL(P || Q) where P and Q are gaussian_kde objects.

    Args:
        p_kde: gaussian_kde object representing distribution P
        q_kde: gaussian_kde object representing distribution Q
        samples: Optional array of sample points to evaluate at (shape: [d, N])
        num_samples: Number of points to sample if samples not provided

    Returns:
        Approximate KL divergence D_KL(P || Q)
    """
    # Sample from P
    samples = p_kde.resample(num_samples)

    # Evaluate log densities
    p_vals = p_kde.evaluate(samples)
    q_vals = q_kde.evaluate(samples)

    # Add small epsilon to avoid log(0) or division by zero
    eps = 1e-10
    p_vals = np.clip(p_vals, eps, None)
    q_vals = np.clip(q_vals, eps, None)

    # Compute KL divergence
    kl_div = np.mean(np.log(p_vals / q_vals))
    return kl_div


def compare_kdes(kde1, kde2, grid_size=1000, num_samples=1000, eps=1e-12):
    """
    Compare two gaussian_kde objects using:
    - Jensen-Shannon Divergence
    - Wasserstein Distance

    Args:
        kde1, kde2: gaussian_kde objects
        grid_size: number of points for JSD grid
        num_samples: number of samples to draw for Wasserstein
        eps: small constant to avoid log(0)

    Returns:
        jsd (float): Jensen-Shannon Divergence
        wass (float): Wasserstein distance
    """
    # Determine joint support range
    data_min = min(kde1.dataset.min(), kde2.dataset.min())
    data_max = max(kde1.dataset.max(), kde2.dataset.max())
    x = np.linspace(data_min, data_max, grid_size)

    # Evaluate KDEs on the grid
    p = kde1(x) + eps
    q = kde2(x) + eps

    # Normalize
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    # Jensen-Shannon Divergence
    jsd = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

    # Wasserstein Distance using samples
    samples1 = kde1.resample(num_samples)[0]
    samples2 = kde2.resample(num_samples)[0]
    wass = wasserstein_distance(samples1, samples2)

    return jsd, wass
