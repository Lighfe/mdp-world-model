from typing import Any
import sys
from pathlib import Path
import torch

# Define project root at the module level
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NOTE: absolute imports from project root
from neural_networks.system_registry import SystemType, get_system_config
from neural_networks.drm_dataset import create_data_loaders
from neural_networks.drm_loss import StableDRMLoss
from neural_networks.drm import DiscreteRepresentationsModel, initialize_model_weights
from neural_networks.utils import (
    set_all_seeds,
    run_layer_probing,
    create_optimizer_with_bias_exclusion,
    create_scheduler,
    load_and_validate_config,
)


def train_drm_simple(config_path, shared_history=None):
    """
    Stripped-down DRM training function for the saddle_explorer marimo notebook.

    Runs the full training loop without any file I/O, visualizations, or
    console output. When shared_history is provided the history dict is merged
    into it at initialization so the notebook can read live training metrics.

    Args:
        config_path: Path to YAML config file.
        shared_history: Optional dict to update in-place with training progress.

    Returns:
        (model, history)
    """
    # Load config
    config = load_and_validate_config(config_path)

    # === META PARAMETERS ===
    db_path = config["db_path"]
    seed = config["seed"]

    # === DATA PARAMETERS ===
    val_size = config["val_size"]
    test_size = config["test_size"]

    # === MODEL PARAMETERS ===
    system_type = config["system_type"]
    num_states = config["num_states"]
    hidden_dim = config["hidden_dim"]
    predictor_type = config["predictor_type"]
    value_method = config["value_method"]
    use_target_encoder = config["use_target_encoder"]
    ema_decay = config["ema_decay"]
    use_gumbel = config["use_gumbel"]
    initial_temp = config["initial_temp"]
    min_temp = config["min_temp"]
    encoder_init_method = config["encoder_init_method"]

    # === TRAINING PARAMETERS ===
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    optimizer_type = config["optimizer_type"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    scheduler_type = config["scheduler_type"]
    min_lr = config["min_lr"]
    use_warmup = config["use_warmup"]
    warmup_epochs = config["warmup_epochs"]
    restart_period = config["restart_period"]
    restart_mult = config["restart_mult"]

    # === LOSS PARAMETERS ===
    state_loss_weight = config["state_loss_weight"]
    value_loss_weight = config["value_loss_weight"]
    state_loss_type = config["state_loss_type"]
    value_loss_type = config["value_loss_type"]
    use_entropy_reg = config["use_entropy_reg"]
    entropy_weight = config["entropy_weight"]
    use_entropy_decay = config["use_entropy_decay"]
    entropy_decay_proportion = config["entropy_decay_proportion"]

    # Set seeds
    set_all_seeds(seed)

    # System configuration
    system_config = get_system_config(SystemType[system_type.upper()])

    if value_method is None:
        value_method = system_config["default_value_method"]

    if value_loss_type is None:
        value_loss_type = system_config["default_value_loss"][value_method]

    # Setup paths
    db_path = Path(db_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        system_type=system_type,
        db_path=db_path,
        value_method=value_method,
        batch_size=batch_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        num_workers=0,
        verbose=False,
    )

    # Get dimensions from first batch
    for x, c, y, v_true in train_loader:
        obs_dim = x.shape[1]
        control_dim = c.shape[1]
        value_dim = v_true.shape[1]
        break

    value_activation = system_config["value_activation"].get(value_method, None)
    if value_activation is None:
        value_activation = "identity"

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = DiscreteRepresentationsModel(
        obs_dim=obs_dim,
        control_dim=control_dim,
        value_dim=value_dim,
        num_states=num_states,
        hidden_dim=hidden_dim,
        predictor_type=predictor_type,
        use_target_encoder=use_target_encoder,
        ema_decay=ema_decay,
        use_gumbel=use_gumbel,
        initial_temp=initial_temp,
        min_temp=min_temp,
        value_activation=value_activation,
    ).to(device)

    current_temp = initial_temp

    # Initialize model weights
    initialize_model_weights(model, encoder_init_method=encoder_init_method)

    # Create optimizer
    optimizer = create_optimizer_with_bias_exclusion(
        model, optimizer_type, lr, weight_decay
    )

    # Create scheduler
    lr_scheduler = create_scheduler(
        optimizer,
        scheduler_type,
        epochs,
        warmup_epochs,
        use_warmup,
        min_lr,
        restart_period,
        restart_mult,
    )

    # Gradient clipping
    max_grad_norm = 1.0

    # Create loss function
    loss_fn = StableDRMLoss(
        state_loss_weight=state_loss_weight,
        value_loss_weight=value_loss_weight,
        value_method=value_method,
        use_entropy_reg=use_entropy_reg,
        entropy_weight=entropy_weight,
        use_entropy_decay=use_entropy_decay,
        state_loss_type=state_loss_type,
        value_loss_type=value_loss_type,
        entropy_decay_proportion=entropy_decay_proportion,
    )

    # Initialize history
    history: dict[str, Any] = {
        "train_loss": [],
        "train_state_loss": [],
        "train_value_loss": [],
        "train_entropy_loss": [],
        "train_batch_entropy": [],
        "train_individual_entropy": [],
        "train_entropy_weight": [],
        "train_softmax_temp": [],
        "val_loss": [],
        "val_state_loss": [],
        "val_value_loss": [],
        "val_entropy_loss": [],
        "val_batch_entropy": [],
        "val_individual_entropy": [],
    }
    if shared_history is not None:
        shared_history.update(history)
        history = shared_history

    # TRAINING LOOP
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_state_loss = 0.0
        train_value_loss = 0.0
        train_entropy_loss = 0.0
        train_batch_entropy = 0.0
        train_individual_entropy = 0.0

        if use_gumbel:
            current_temp = model.update_temperature(epoch, epochs)

        history["train_softmax_temp"].append(current_temp)

        if loss_fn.use_entropy_reg:
            current_entropy_weight = loss_fn.update_entropy_weight(epoch, epochs)
            history["train_entropy_weight"].append(current_entropy_weight)

        if use_warmup and epoch < warmup_epochs:
            progress = epoch / warmup_epochs
            new_lr = lr * (0.1 + 0.9 * progress)  # 10% to 100% of base LR
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr

        for x, c, y, v_true in train_loader:
            x, c, y, v_true = (
                x.to(device),
                c.to(device),
                y.to(device),
                v_true.to(device),
            )

            if (
                torch.isnan(x).any()
                or torch.isnan(c).any()
                or torch.isnan(y).any()
                or torch.isnan(v_true).any()
            ):
                continue

            try:
                optimizer.zero_grad(set_to_none=True)

                s_x, s_y, s_y_pred, v_pred_for_all_states = model(
                    x, c, y, v_true, training=True
                )

                (
                    total_loss,
                    state_loss,
                    value_loss,
                    entropy_loss,
                    batch_entropy,
                    individual_entropy,
                ) = loss_fn(
                    s_y,
                    s_y_pred,
                    v_true,
                    v_pred_for_all_states,
                    s_x,
                    epoch=epoch,
                    max_epochs=epochs,
                )

                if torch.isnan(total_loss):
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )
                optimizer.step()

                if use_target_encoder:
                    model.update_target_encoder()

                train_loss += total_loss.item()
                train_state_loss += state_loss.item()
                train_value_loss += value_loss.item()
                train_entropy_loss += entropy_loss.item()
                train_batch_entropy += batch_entropy.item()
                train_individual_entropy += individual_entropy.item()

            except Exception:
                continue

        num_batches = len(train_loader)
        if num_batches > 0:
            train_loss /= num_batches
            train_state_loss /= num_batches
            train_value_loss /= num_batches
            train_entropy_loss /= num_batches
            train_batch_entropy /= num_batches
            train_individual_entropy /= num_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_state_loss = 0.0
        val_value_loss = 0.0
        val_entropy_loss = 0.0
        val_batch_entropy = 0.0
        val_individual_entropy = 0.0
        valid_batches = 0

        with torch.no_grad():
            for x, c, y, v_true in val_loader:
                x, c, y, v_true = (
                    x.to(device),
                    c.to(device),
                    y.to(device),
                    v_true.to(device),
                )

                if (
                    torch.isnan(x).any()
                    or torch.isnan(c).any()
                    or torch.isnan(y).any()
                    or torch.isnan(v_true).any()
                ):
                    continue

                try:
                    s_x, s_y, s_y_pred, v_pred_for_all_states = model(
                        x, c, y, v_true, training=False
                    )

                    (
                        total_loss,
                        state_loss,
                        value_loss,
                        entropy_loss,
                        batch_entropy,
                        individual_entropy,
                    ) = loss_fn(
                        s_y,
                        s_y_pred,
                        v_true,
                        v_pred_for_all_states,
                        s_x,
                        epoch=epoch,
                        max_epochs=epochs,
                    )

                    if torch.isnan(total_loss):
                        continue

                    val_loss += total_loss.item()
                    val_state_loss += state_loss.item()
                    val_value_loss += value_loss.item()
                    val_entropy_loss += entropy_loss.item()
                    val_batch_entropy += batch_entropy.item()
                    val_individual_entropy += individual_entropy.item()
                    valid_batches += 1

                except Exception:
                    continue

        if valid_batches > 0:
            val_loss /= valid_batches
            val_state_loss /= valid_batches
            val_value_loss /= valid_batches
            val_entropy_loss /= valid_batches
            val_batch_entropy /= valid_batches
            val_individual_entropy /= valid_batches

        # Update history
        history["train_loss"].append(train_loss)
        history["train_state_loss"].append(train_state_loss)
        history["train_value_loss"].append(train_value_loss)
        history["train_entropy_loss"].append(train_entropy_loss)
        history["train_batch_entropy"].append(train_batch_entropy)
        history["train_individual_entropy"].append(train_individual_entropy)
        history["val_loss"].append(val_loss)
        history["val_state_loss"].append(val_state_loss)
        history["val_value_loss"].append(val_value_loss)
        history["val_entropy_loss"].append(val_entropy_loss)
        history["val_batch_entropy"].append(val_batch_entropy)
        history["val_individual_entropy"].append(val_individual_entropy)

        # INTERMEDIATE LAYER PROBING - every 5 epochs
        if ((epoch + 1) % 5 == 0 or epoch == 0) and system_type == "saddle_system":
            intermediate_results = run_layer_probing(
                model, val_loader, device, system_type, db_path
            )
            if "intermediate_probing" not in history:
                history["intermediate_probing"] = []
            history["intermediate_probing"].append(
                {
                    "epoch": epoch + 1,
                    "discrete_accuracy": intermediate_results["discrete_accuracy"],
                    "discrete_accuracy_unweighted": intermediate_results[
                        "discrete_accuracy_unweighted"
                    ],
                }
            )

        # Apply scheduler
        if lr_scheduler is not None and (not use_warmup or epoch >= warmup_epochs):
            lr_scheduler.step()

    # FINAL TEST EVALUATION
    model.eval()
    test_loss = 0.0
    test_state_loss = 0.0
    test_value_loss = 0.0
    test_entropy_loss = 0.0
    test_batch_entropy = 0.0
    test_individual_entropy = 0.0
    test_samples = 0

    with torch.no_grad():
        for x, c, y, v_true in test_loader:
            x, c, y, v_true = (
                x.to(device),
                c.to(device),
                y.to(device),
                v_true.to(device),
            )

            if (
                torch.isnan(x).any()
                or torch.isnan(c).any()
                or torch.isnan(y).any()
                or torch.isnan(v_true).any()
            ):
                continue

            try:
                s_x, s_y, s_y_pred, v_pred_for_all_states = model(
                    x, c, y, v_true, training=False
                )

                if (
                    torch.isnan(s_y).any()
                    or torch.isnan(s_y_pred).any()
                    or torch.isnan(v_pred_for_all_states).any()
                ):
                    continue

                (
                    total_loss,
                    state_loss,
                    value_loss,
                    entropy_loss,
                    batch_entropy,
                    individual_entropy,
                ) = loss_fn(
                    s_y,
                    s_y_pred,
                    v_true,
                    v_pred_for_all_states,
                    s_x,
                    epoch=epoch,
                    max_epochs=epochs,
                )

                if torch.isnan(total_loss):
                    continue

                test_loss += total_loss.item() * len(x)
                test_state_loss += state_loss.item() * len(x)
                test_value_loss += value_loss.item() * len(x)
                test_entropy_loss += entropy_loss.item() * len(x)
                test_batch_entropy += batch_entropy.item() * len(x)
                test_individual_entropy += individual_entropy.item() * len(x)
                test_samples += len(x)

            except Exception:
                continue

    if test_samples > 0:
        test_loss /= test_samples
        test_state_loss /= test_samples
        test_value_loss /= test_samples
        test_entropy_loss /= test_samples
        test_batch_entropy /= test_samples
        test_individual_entropy /= test_samples

    # FINAL LAYER PROBING on test set
    probing_results = None
    if system_type == "saddle_system":
        probing_results = run_layer_probing(
            model, test_loader, device, system_type, db_path
        )

    history["test_metrics"] = {
        "test_loss": float(test_loss),
        "test_state_loss": float(test_state_loss),
        "test_value_loss": float(test_value_loss),
        "test_entropy_loss": float(test_entropy_loss),
        "test_batch_entropy": float(test_batch_entropy),
        "test_individual_entropy": float(test_individual_entropy),
        "test_samples": test_samples,
        "prob_discrete_accuracy": (
            probing_results["discrete_accuracy"] if probing_results else None
        ),
        "prob_discrete_accuracy_unweighted": (
            probing_results["discrete_accuracy_unweighted"] if probing_results else None
        ),
    }

    return model, history
