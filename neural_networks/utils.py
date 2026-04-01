import os
import sys
from pathlib import Path
import json
import yaml
import shutil
from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import WeightedRandomSampler
from collections import Counter

# Define project root at the module level
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"Added {PROJECT_ROOT} to Python path")

# NOTE: absolute imports from project root
# Import application-specific modules
from neural_networks.system_registry import SystemType, get_visualization_bounds
from neural_networks.drm import LinearProbe


def set_all_seeds(seed):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Also set deterministic behavior for CUDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def calculate_state_weights(targets, verbose=False):
    """
    Calculate sample-level weights based on ground truth state frequency

    Args:
        targets: [num_samples, num_halfspaces] binary targets

    Returns:
        sample_weights: [num_samples] weight for each sample
    """
    # Convert each sample's halfspace pattern to a state ID
    state_ids = []
    for i in range(targets.shape[0]):
        # Convert binary pattern to unique state ID
        state_pattern = tuple(targets[i].numpy())
        state_ids.append(state_pattern)

    # Count frequency of each state
    state_counts = Counter(state_ids)
    total_samples = len(state_ids)

    if verbose:
        print("State distribution:")
        for state, count in state_counts.items():
            print(f"  State {state}: {count} samples ({count/total_samples*100:.1f}%)")

    # Calculate inverse frequency weights
    sample_weights = []
    for state_id in state_ids:
        # Weight = 1 / (num_states * frequency)
        weight = total_samples / (len(state_counts) * state_counts[state_id])
        sample_weights.append(weight)

    return torch.tensor(sample_weights, dtype=torch.float32)


def train_probe(features, targets, probe_name, num_epochs=20, lr=0.01, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = features.shape[1]
    num_saddles = targets.shape[1]
    probe = LinearProbe(input_dim, num_saddles).to(device)

    # Calculate weights from the extracted targets
    sample_weights = calculate_state_weights(targets)

    # Create dataset and weighted sampler
    dataset = torch.utils.data.TensorDataset(features, targets)
    weighted_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=weighted_sampler
    )

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        for batch_features, batch_targets in dataloader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(
                device
            )
            optimizer.zero_grad()
            predictions = probe(batch_features)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()

    # Final evaluation with same balanced accuracy calculation
    with torch.no_grad():
        full_predictions = probe(features.to(device))
        binary_preds = (full_predictions > 0.5).float()
        targets_device = targets.to(device)

        # Balanced accuracy calculation (same as above)
        unique_states = []
        state_accuracies = []

        for i in range(targets_device.shape[0]):
            state_pattern = tuple(targets_device[i].cpu().numpy())
            if state_pattern not in unique_states:
                unique_states.append(state_pattern)

        for state_pattern in unique_states:
            state_mask = torch.zeros(targets_device.shape[0], dtype=torch.bool)
            for i in range(targets_device.shape[0]):
                if tuple(targets_device[i].cpu().numpy()) == state_pattern:
                    state_mask[i] = True

            if state_mask.sum() > 0:
                state_preds = binary_preds[state_mask]
                state_targets = targets_device[state_mask]
                state_acc = (state_preds == state_targets).all(dim=1).float().mean()
                state_accuracies.append(state_acc.item())

        state_accuracy = np.mean(state_accuracies)
        unweighted_accuracy = (
            (binary_preds == targets_device).all(dim=1).float().mean().item()
        )

    return probe, state_accuracy, unweighted_accuracy


def extract_features_and_targets(model, probing_loader, device, system_type):
    """
    Extract features and targets for layer probing
    """
    model.eval()

    dataset = probing_loader.dataset
    probing_indices = list(probing_loader.sampler.indices)

    discrete_features = []
    targets = []

    with torch.no_grad():
        batch_size = probing_loader.batch_size
        for i in range(0, len(probing_indices), batch_size):
            batch_indices = probing_indices[i : i + batch_size]

            batch_x = []
            for idx in batch_indices:
                x, _, _, _ = dataset[idx]
                batch_x.append(x)

            batch_x = torch.stack(batch_x)

            # Calculate targets for entire batch at once - EFFICIENT!
            if system_type == "saddle_system":
                batch_targets = dataset._halfspace_for_batch(batch_x)
                targets.append(batch_targets)
            else:
                raise NotImplementedError(
                    f"Layer probing not implemented for system_type: {system_type}"
                )

            # Process batch through model
            batch_x_gpu = batch_x.to(device)
            discrete_x = model.get_state_probs(
                batch_x_gpu, training=False, hard=True, use_target=False
            )

            discrete_features.append(discrete_x.cpu())

    return (torch.cat(discrete_features), torch.cat(targets))


def run_layer_probing(
    model, probing_loader, device, system_type, db_path, verbose=False
):
    """Simplified layer probing with guaranteed data alignment"""

    if verbose:
        print("\n" + "=" * 50)
        print("STARTING LAYER PROBING")
        print("=" * 50)

    # Save original gradient states
    original_requires_grad = {}
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
        # freeze
        param.requires_grad = False

    # Extract everything together - guaranteed alignment
    discrete_features, probe_targets = extract_features_and_targets(
        model, probing_loader, device, system_type
    )
    if verbose:
        print(
            f"Extracted paired data - Features: {discrete_features.shape}, Targets: {probe_targets.shape}"
        )

    # Train probes
    discrete_probe, discrete_acc, discrete_acc_unweighted = train_probe(
        discrete_features, probe_targets, "discrete"
    )

    # Restore original gradient states
    for name, param in model.named_parameters():
        param.requires_grad = original_requires_grad[name]

    # Ensure model is back in training mode
    model.train()

    return {
        "discrete_accuracy": discrete_acc,
        "discrete_accuracy_unweighted": discrete_acc_unweighted,
    }


def extract_state_assignment_data(
    model,
    device,
    num_states,
    system_type,
    bounds=None,
    grid_size=100,
    epoch=None,
    softmax_temp=1.0,
):
    """
    Extract state assignment data for visualization without creating plots.

    Args:
        model: Trained DRM model
        transformations: List of transformation functions for each dimension
        device: PyTorch device
        num_states: Number of discrete states
        system_type: Type of system ('tech_substitution', 'saddle_system')
        bounds: [(x1_min, x1_max), (x2_min, x2_max)] or None for defaults
        grid_size: Number of points per dimension (grid_size x grid_size total)
        epoch: Current epoch number (for tracking)
        softmax_temp: Temperature for softmax (default 1.0 for visualization)

    Returns:
        pd.DataFrame: Contains grid points, transformations, and state probabilities
    """

    # Set default bounds if not provided
    if bounds is None:
        bounds = [(-5, 5), (-5, 5)]

    # Generate grid points in original space
    x_range = np.linspace(bounds[0][0], bounds[0][1], grid_size)
    y_range = np.linspace(bounds[1][0], bounds[1][1], grid_size)
    xx, yy = np.meshgrid(x_range, y_range)

    # Flatten grid points
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Get model predictions
    model.eval()
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(grid_points).to(device)

        state_probs = model.get_state_probs(obs_tensor, training=False, soft=True)

        state_probs = state_probs.cpu().numpy()

    # Create DataFrame
    df_data = []
    for i, orig_point in enumerate(grid_points):
        row = {"x1": orig_point[0], "x2": orig_point[1], "grid_idx": i}

        # Add state probabilities
        for state_idx in range(num_states):
            row[f"state_{state_idx}_prob"] = state_probs[i, state_idx]

        # Add dominant state and its probability
        dominant_state = np.argmax(state_probs[i])
        row["dominant_state"] = dominant_state
        row["dominant_prob"] = state_probs[i, dominant_state]

        # Add metadata
        if epoch is not None:
            row["epoch"] = epoch
        row["softmax_temp"] = softmax_temp
        row["system_type"] = system_type

        df_data.append(row)

    df = pd.DataFrame(df_data)

    # Add grid metadata as attributes
    df.attrs = {
        "grid_size": grid_size,
        "bounds": bounds,
        "num_states": num_states,
        "x_range": x_range.tolist(),
        "y_range": y_range.tolist(),
    }

    return df


def create_optimizer_with_bias_exclusion(
    model, optimizer_type, lr, weight_decay, verbose=False
):
    """
    Create optimizer with weight decay and bias exclusion (following I-JEPA approach)
    """
    if optimizer_type == "adamw" and weight_decay > 0:
        # Separate parameter groups: regular weights vs bias/layernorm
        param_groups = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ],
                "weight_decay": 0.0,  # No weight decay for bias/layernorm
            },
        ]
        optimizer = optim.AdamW(param_groups, lr=lr)
        if verbose:
            print(
                f"Using AdamW optimizer with weight_decay={weight_decay} (bias excluded)"
            )

    elif optimizer_type == "adamw":
        # AdamW without weight decay
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        if verbose:
            print(f"Using AdamW optimizer without weight decay")

    else:  # adam
        optimizer = optim.Adam(model.parameters(), lr=lr)
        if verbose:
            print(f"Using Adam optimizer")

    return optimizer


def create_scheduler(
    optimizer,
    scheduler_type,
    epochs,
    warmup_epochs,
    use_warmup,
    min_lr,
    restart_period,
    restart_mult,
    verbose=False,
):
    """
    Create learning rate scheduler
    """
    if scheduler_type == "fixed":
        # No scheduler - constant learning rate
        if verbose:
            print("Using fixed learning rate (no scheduling)")
        return None

    elif scheduler_type == "warm_restarts":
        # WarmRestarts - starts after warmup period
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=restart_period, T_mult=restart_mult, eta_min=min_lr
        )
        if verbose:
            print(
                f"Using CosineAnnealingWarmRestarts scheduler (T_0: {restart_period}, T_mult: {restart_mult}, min_lr: {min_lr:.2e})"
            )

    else:  # cosine
        T_max = epochs - warmup_epochs if use_warmup else epochs
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, T_max), eta_min=min_lr
        )
        if verbose:
            print(
                f"Using CosineAnnealingLR scheduler (T_max: {T_max}, min_lr: {min_lr:.2e})"
            )

    return lr_scheduler


def calculate_epoch_metrics(
    grid_points,
    state_probs,
    epoch,
    num_states,
    prev_dominant_states=None,
    grid_size=100,
):
    """
    Calculate all state assignment metrics for a single epoch.

    Args:
        grid_points: np.array of shape (grid_size^2, 2) - the x1, x2 coordinates
        state_probs: np.array of shape (grid_size^2, num_states) - state probabilities
        epoch: current epoch number
        num_states: number of discrete states
        prev_dominant_states: np.array from previous epoch for stability calculation
        grid_size: grid size for reshaping

    Returns:
        dict: All metrics for this epoch
    """

    # Basic info
    metrics = {"epoch": epoch}

    # 1. State usage (how often each state is dominant)
    dominant_states = np.argmax(state_probs, axis=1)

    for state_idx in range(num_states):
        usage = np.mean(dominant_states == state_idx)
        metrics[f"state_{state_idx}_usage"] = usage

    # 2. Mean probabilities per state
    for state_idx in range(num_states):
        mean_prob = np.mean(state_probs[:, state_idx])
        metrics[f"state_{state_idx}_mean"] = mean_prob

    # 3. Sharpness: per-grid-point entropy (discreteness measure)
    # Calculate entropy per grid point: -sum(p * log(p)) for each row
    per_point_entropy = -np.sum(state_probs * np.log(state_probs + 1e-8), axis=1)
    metrics["sharpness_mean"] = np.mean(per_point_entropy)
    metrics["sharpness_std"] = np.std(per_point_entropy)

    # 4. Stability metrics (if we have previous epoch data)
    if prev_dominant_states is not None:
        # Dominant state stability: % of points keeping same dominant state
        same_dominant = np.mean(dominant_states == prev_dominant_states) * 100
        metrics["dominant_stability"] = same_dominant

        # Note: For probability stability, we'd need previous state_probs
        # For now, we'll skip this or store prev_state_probs too if needed

    return metrics, dominant_states


def save_state_data_frame(
    grid_points,
    state_probs,
    epoch,
    num_states,
    output_dir,
    run_id,
    bounds,
    grid_size=100,
):
    """
    Save lightweight state data instead of PNG frame.
    """
    import pickle

    # Create lightweight data structure
    frame_data = {
        "grid_points": grid_points,
        "state_probs": state_probs,
        "epoch": epoch,
        "num_states": num_states,
        "bounds": bounds,
        "grid_size": grid_size,
        "run_id": run_id,
    }

    # Save as pickle file (much faster than PNG)
    data_filename = f"state_data_epoch_{epoch}_{run_id}.pkl"
    output_dir = Path(output_dir)
    data_path = output_dir / data_filename

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_path, "wb") as f:
        pickle.dump(frame_data, f)

    return str(data_path)


def extract_and_calculate_metrics(
    model,
    device,
    num_states,
    system_type,
    bounds=None,
    grid_size=100,
    epoch=None,
    prev_dominant_states=None,
):
    """
    Extract state assignments and immediately calculate metrics.

    This replaces the extract_state_assignment_data + metric calculation pipeline.

    Args:
        model: DRM model
        device: PyTorch device
        num_states: number of discrete states
        system_type: system type string for bounds lookup
        bounds: visualization bounds (if None, will get from system registry)
        grid_size: grid size for visualization
        epoch: current epoch
        prev_dominant_states: previous epoch's dominant states for stability

    Returns:
        tuple: (metrics_dict, dominant_states, grid_points, state_probs)
    """

    # Get proper bounds if not provided
    if bounds is None:
        bounds = get_visualization_bounds(SystemType[system_type.upper()])

    # Generate grid points
    x_range = np.linspace(bounds[0][0], bounds[0][1], grid_size)
    y_range = np.linspace(bounds[1][0], bounds[1][1], grid_size)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Get model predictions
    model.eval()
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(grid_points).to(device)
        state_probs = model.get_state_probs(obs_tensor, training=False, soft=False)
        state_probs = state_probs.cpu().numpy()

    # Calculate metrics immediately
    metrics, dominant_states = calculate_epoch_metrics(
        grid_points, state_probs, epoch, num_states, prev_dominant_states, grid_size
    )

    return metrics, dominant_states, grid_points, state_probs


# Softmax
def compute_numerical_rank(matrix, threshold_ratio=0.01):
    """
    Compute numerical rank using SVD with threshold.

    Args:
        matrix: torch.Tensor of shape (features, samples)
        threshold_ratio: Threshold as ratio of largest singular value (default: 1%)

    Returns:
        int: Numerical rank
        np.ndarray: Singular values
    """
    # Ensure matrix is 2D
    if matrix.dim() > 2:
        matrix = matrix.view(-1, matrix.size(-1))

    # Convert to numpy for SVD
    matrix_np = matrix.detach().cpu().numpy()

    # Compute SVD
    try:
        U, s, Vt = np.linalg.svd(matrix_np, full_matrices=False)
    except np.linalg.LinAlgError:
        # Handle degenerate case
        return 0, np.array([])

    # Compute numerical rank with threshold
    if len(s) == 0:
        return 0, s

    threshold = threshold_ratio * s[0]
    rank = np.sum(s > threshold)

    return int(rank), s


def extract_hidden_and_logit_data(
    model, dataloader, device, max_samples=1000, verbose=False
):
    """
    Extract last hidden layer activations and final logits from model.

    Args:
        model: DRM model
        dataloader: DataLoader (should be validation set)
        device: torch device
        max_samples: Maximum number of samples to use for analysis

    Returns:
        dict: Contains 'hidden_activations', 'pre_softmax_logits', 'post_softmax_probs'
    """
    model.eval()

    captured_hidden = []
    captured_logits = []
    all_softmax_probs = []

    total_samples = 0

    # Hooks to capture activations
    def hidden_hook(module, input, output):
        captured_hidden.append(output.clone())

    def logit_hook(module, input, output):
        captured_logits.append(
            output.clone()
        )  # output of final layer = pre-softmax logits

    # Register hooks on correct encoder layers
    # Encoder structure: [Linear, ReLU, Linear, ReLU, Linear, ReLU, Linear, ReLU, Linear]
    # We want: layer 6 (last hidden Linear 32→32) and layer 8 (final Linear 32→4)
    if len(model.encoder) >= 9:
        hidden_handle = model.encoder[6].register_forward_hook(
            hidden_hook
        )  # Last hidden Linear
        logit_handle = model.encoder[8].register_forward_hook(
            logit_hook
        )  # Final Linear
    else:
        raise ValueError(
            f"Encoder structure unexpected - has {len(model.encoder)} layers, expected ≥9"
        )

    try:
        with torch.no_grad():
            for batch in dataloader:
                if total_samples >= max_samples:
                    break

                x_obs, x_control, x_next_obs, values = batch
                x_obs = x_obs.to(device)

                batch_size = x_obs.size(0)
                if total_samples + batch_size > max_samples:
                    take = max_samples - total_samples
                    x_obs = x_obs[:take]
                    batch_size = take

                # Clear previous captures
                captured_hidden.clear()
                captured_logits.clear()

                # Forward pass - triggers hooks
                softmax_probs = model.encoder(x_obs)

                # Store results
                if captured_hidden:
                    all_softmax_probs.append(softmax_probs)
                    total_samples += batch_size
                else:
                    print("Warning: No activations captured")
                    break

    finally:
        # Remove hooks
        hidden_handle.remove()
        logit_handle.remove()

    # Concatenate all batches
    hidden_activations = (
        torch.cat(captured_hidden, dim=0) if captured_hidden else torch.empty(0, 32)
    )
    pre_softmax_logits = (
        torch.cat(captured_logits, dim=0) if captured_logits else torch.empty(0, 4)
    )
    post_softmax_probs = (
        torch.cat(all_softmax_probs, dim=0) if all_softmax_probs else torch.empty(0, 4)
    )

    if verbose:
        print(
            f"Extracted activations: Hidden {hidden_activations.shape}, Logits {pre_softmax_logits.shape}"
        )

    # Verify shapes are correct
    if hidden_activations.size(1) != 32:
        print(
            f"⚠️ WARNING: Hidden activations should be 32-dim, got {hidden_activations.size(1)}"
        )
    if pre_softmax_logits.size(1) != 4:
        print(
            f"⚠️ WARNING: Pre-softmax logits should be 4-dim, got {pre_softmax_logits.size(1)}"
        )

    return {
        "hidden_activations": hidden_activations,  # (N, 32)
        "pre_softmax_logits": pre_softmax_logits,  # (N, 4)
        "post_softmax_probs": post_softmax_probs,  # (N, 4)
    }


def compute_softmax_rank_metrics(model, val_dataloader, device, max_samples=1000):
    """
    Compute softmax rank metrics focusing on last hidden layer and final logits.
    Raw data only - no theoretical maximums or efficiency calculations.

    Args:
        model: DRM model
        val_dataloader: Validation DataLoader
        device: torch device
        max_samples: Max samples for analysis

    Returns:
        dict: All computed metrics (raw data only)
    """
    # Extract activations and logits
    data = extract_hidden_and_logit_data(model, val_dataloader, device, max_samples)

    hidden_acts = data["hidden_activations"]  # (N, 32)
    pre_logits = data["pre_softmax_logits"]  # (N, 4)
    post_probs = data["post_softmax_probs"]  # (N, 4)

    if hidden_acts.numel() == 0:
        print("Warning: No data extracted, returning empty metrics")
        return {}

    metrics = {}

    # ===== LAST HIDDEN LAYER (A₃, 32-dim) =====

    # 1. Frobenius norm of activation matrix
    hidden_frobenius = torch.norm(hidden_acts, p="fro").item()
    metrics["hidden_frobenius_norm"] = hidden_frobenius

    # 2. Rank and SVD analysis
    hidden_acts_T = hidden_acts.T  # (32, N) for SVD
    hidden_rank, hidden_sv = compute_numerical_rank(hidden_acts_T)
    metrics["hidden_rank"] = hidden_rank

    # Store RAW singular values for hidden layer (for later global normalization)
    if len(hidden_sv) > 0:
        for i, sv in enumerate(hidden_sv[: min(8, len(hidden_sv))]):
            metrics[f"hidden_sv_raw_{i}"] = float(sv)

    # ===== FINAL LOGIT LAYER (M₄, A₄, 4-dim) =====

    # 1. Frobenius norms
    pre_logits_frobenius = torch.norm(pre_logits, p="fro").item()
    post_probs_frobenius = torch.norm(post_probs, p="fro").item()
    metrics["logit_frobenius_norm"] = pre_logits_frobenius
    metrics["softmax_frobenius_norm"] = post_probs_frobenius

    # 2. Ranks
    pre_logits_T = pre_logits.T  # (4, N)
    post_probs_T = post_probs.T  # (4, N)

    logit_rank, logit_sv = compute_numerical_rank(pre_logits_T)
    softmax_rank, softmax_sv = compute_numerical_rank(post_probs_T)

    metrics["logit_rank"] = logit_rank
    metrics["softmax_rank"] = softmax_rank

    # 3. Detailed SVD analysis for logits (M₄) - Store RAW values only
    if len(logit_sv) > 0:
        # Store raw singular values for global normalization
        for i, sv in enumerate(logit_sv[:4]):  # Only first 4 for 4-dim layer
            metrics[f"logit_sv_raw_{i}"] = float(sv)

        # σ₂/σ₁ ratio for collapse detection (still useful for real-time monitoring)
        if len(logit_sv) > 1:
            metrics["logit_sv_ratio_2nd_to_1st"] = (
                float(logit_sv[1] / logit_sv[0]) if logit_sv[0] > 0 else 0.0
            )
        else:
            metrics["logit_sv_ratio_2nd_to_1st"] = 0.0

    return metrics


def collect_softmax_rank_metrics(
    model, val_dataloader, device, epoch, history, max_samples=1000, verbose=False
):
    """
    Collect softmax rank metrics and add to history.

    Args:
        model: DRM model
        val_dataloader: Validation DataLoader
        device: torch device
        epoch: Current epoch number
        history: Training history dict
        max_samples: Max samples for analysis
    """
    try:
        if verbose:
            print(f"Computing softmax rank metrics for epoch {epoch}...")

        # Compute metrics
        metrics = compute_softmax_rank_metrics(
            model, val_dataloader, device, max_samples
        )

        if not metrics:  # Empty metrics
            print("  Skipping due to empty metrics")
            return

        # Add epoch info
        metrics["epoch"] = epoch

        # Add to history
        if "softmax_rank_metrics" not in history:
            history["softmax_rank_metrics"] = []
        history["softmax_rank_metrics"].append(metrics)

        # Print key metrics for monitoring
        if verbose:
            print(
                f"  Hidden layer (32-dim): rank={metrics['hidden_rank']}, "
                f"||A₃||_F={metrics['hidden_frobenius_norm']:.3f}"
            )
            print(
                f"  Logit layer (4-dim): rank={metrics['logit_rank']}, "
                f"||M₄||_F={metrics['logit_frobenius_norm']:.3f}"
            )
            print(
                f"  Post-softmax: rank={metrics['softmax_rank']}, "
                f"||A₄||_F={metrics['softmax_frobenius_norm']:.3f}"
            )

        # Show logit singular value ratio for collapse detection
        if "logit_sv_ratio_2nd_to_1st" in metrics:
            ratio = metrics["logit_sv_ratio_2nd_to_1st"]
            if verbose:
                print(f"  Logit σ₂/σ₁ ratio: {ratio:.4f}")

        # Check for potential issues
        if metrics["logit_rank"] < 2 and verbose:
            print(
                f"  ⚠️  WARNING: Very low logit rank ({metrics['logit_rank']}) - rank collapse!"
            )

    except Exception as e:
        print(f"Error computing softmax rank metrics: {e}")
        import traceback

        traceback.print_exc()


def add_global_normalized_singular_values(history):
    """
    Post-processing function to add globally normalized singular values to history.
    Call this AFTER training is complete.

    Args:
        history: Training history dict containing softmax_rank_metrics with raw SVs
    """
    if "softmax_rank_metrics" not in history or not history["softmax_rank_metrics"]:
        print("No softmax rank metrics found for global normalization")
        return

    metrics_list = history["softmax_rank_metrics"]

    # ========================================================================
    # Find global maximum singular values across ALL epochs
    # ========================================================================

    # Collect all raw singular values
    all_logit_sv_raw = {i: [] for i in range(4)}
    all_hidden_sv_raw = {i: [] for i in range(8)}  # Up to 8 for hidden layer

    for m in metrics_list:
        # Collect raw logit singular values
        for i in range(4):
            key = f"logit_sv_raw_{i}"
            if key in m:
                all_logit_sv_raw[i].append(m[key])

        # Collect raw hidden singular values
        for i in range(8):
            key = f"hidden_sv_raw_{i}"
            if key in m:
                all_hidden_sv_raw[i].append(m[key])

    # Find global maximum for each layer type
    logit_global_max = 0
    hidden_global_max = 0

    for i in range(4):
        if all_logit_sv_raw[i]:
            logit_global_max = max(logit_global_max, max(all_logit_sv_raw[i]))

    for i in range(8):
        if all_hidden_sv_raw[i]:
            hidden_global_max = max(hidden_global_max, max(all_hidden_sv_raw[i]))

    print(
        f"Global normalization: Logit max = {logit_global_max:.3f}, Hidden max = {hidden_global_max:.3f}"
    )

    # ========================================================================
    # Add globally normalized values to each epoch's metrics
    # ========================================================================

    for m in metrics_list:
        # Add globally normalized logit singular values
        if logit_global_max > 0:
            for i in range(4):
                raw_key = f"logit_sv_raw_{i}"
                if raw_key in m:
                    normalized_value = m[raw_key] / logit_global_max
                    m[f"logit_sv_global_norm_{i}"] = normalized_value

        # Add globally normalized hidden singular values
        if hidden_global_max > 0:
            for i in range(8):
                raw_key = f"hidden_sv_raw_{i}"
                if raw_key in m:
                    normalized_value = m[raw_key] / hidden_global_max
                    m[f"hidden_sv_global_norm_{i}"] = normalized_value

    print(f"Added globally normalized singular values to {len(metrics_list)} entries")


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy/torch types"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if torch is not None and isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super().default(obj)



def safe_json_dump(obj, file_handle, **kwargs):
    """JSON dump that handles numpy/torch types"""
    return json.dump(obj, file_handle, cls=NumpyEncoder, **kwargs)


def load_config(config_path):
    """Load configuration from YAML or JSON file."""

    config_path = Path(config_path)
    with open(config_path, "r") as f:
        if config_path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        else:
            return json.load(f)


def load_and_validate_config(config_path):
    """
    Load configuration from file and validate basic structure.

    Args:
        config_path: Path to YAML config file

    Returns:
        dict: Flattened configuration dictionary

    Raises:
        ValueError: If required sections are missing
    """
    # Load config file
    config = load_config(config_path)

    # Check required sections exist
    required_sections = ["meta", "data", "model", "training", "loss"]
    missing_sections = [s for s in required_sections if s not in config]
    if missing_sections:
        raise ValueError(f"Missing required config sections: {missing_sections}")

    # Flatten config (remove nested structure)
    params = {}
    for section_name, section_config in config.items():
        if isinstance(section_config, dict):
            params.update(section_config)

    print("Config loaded and validated successfully")
    return params


def parse_comma_separated(value):
    """Parse comma-separated values into list"""
    if not value:
        return []
    return [item.strip() for item in value.split(",")]


def set_nested_dict_value(config_dict, key_path, value):
    """
    Set a value in a nested dictionary using dot notation.

    Args:
        config_dict: Dictionary to modify
        key_path: Dot-separated path (e.g., "meta.seed")
        value: Value to set
    """
    keys = key_path.split(".")
    current = config_dict

    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Set the final value
    current[keys[-1]] = value


def create_run_name(override_values, db_path_key="meta.db_path"):
    """
    Create a run name from override values.

    Args:
        override_values: Dictionary of override key-value pairs
        db_path_key: Key that contains database path for naming

    Returns:
        String run name
    """
    # Extract seed and database name for run naming
    seed = override_values.get("meta.seed", "unknown")
    db_path = override_values.get(db_path_key, "unknown")

    if db_path != "unknown":
        db_name = Path(db_path).stem
    else:
        db_name = "unknown"

    return f"seed_{seed}_{db_name}"


def generate_config_combinations(
    base_config_path, config_id, override_params, output_configs_dir
):
    """
    Generate configuration files for all parameter combinations.

    Args:
        base_config_path: Path to base YAML config
        config_id: Identifier for this config set (used in run naming)
        override_params: Dict mapping parameter paths to lists of values
                        e.g., {"meta.seed": [11, 12], "meta.db_path": ["data1.db", "data2.db"]}
        output_configs_dir: Directory where config files will be saved

    Returns:
        List of tuples: (config_path, run_name, override_values_dict)
    """
    # Load base config
    base_config = load_config(base_config_path)

    # Create output directory
    configs_dir = Path(output_configs_dir)
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Generate all combinations using itertools.product
    param_names = list(override_params.keys())
    param_value_lists = [override_params[name] for name in param_names]

    generated_configs = []

    for value_combination in product(*param_value_lists):
        # Create override dictionary for this combination
        override_values = dict(zip(param_names, value_combination))

        # Create run name
        run_name = create_run_name(override_values)

        # Create modified config
        config_copy = base_config.copy()

        # Apply all overrides
        for param_path, value in override_values.items():
            set_nested_dict_value(config_copy, param_path, value)

        # Set run_id if meta section exists
        if "meta" in config_copy:
            config_copy["meta"]["run_id"] = run_name

        # Save individual config
        config_filename = f"config_{run_name}.yaml"
        config_path = configs_dir / config_filename

        with open(config_path, "w") as f:
            yaml.dump(config_copy, f, default_flow_style=False, indent=2)

        generated_configs.append((str(config_path), run_name, override_values))

    print(f"Generated {len(generated_configs)} config files in {configs_dir}")
    return generated_configs


def setup_output_structure(output_dir, config_id, base_config_path):
    """
    Create output directory structure and copy base config.

    Args:
        output_dir: Base output directory
        config_id: Config identifier
        base_config_path: Path to original base config

    Returns:
        Path to the config-specific output directory
    """
    # Create main output structure
    config_output_dir = Path(output_dir) / config_id
    config_output_dir.mkdir(parents=True, exist_ok=True)

    # Create individual_runs subdirectory
    individual_runs_dir = config_output_dir / "individual_runs"
    individual_runs_dir.mkdir(exist_ok=True)

    # Copy base config
    base_config_copy = config_output_dir / "base_config.yaml"
    shutil.copy2(base_config_path, base_config_copy)

    print(f"Set up output structure at {config_output_dir}")
    return config_output_dir
