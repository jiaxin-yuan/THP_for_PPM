

import os
import time
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import Utils
from transformer.Constants import PAD


def train_epoch(
    model,
    train_loader,
    optimiser,
    writer: SummaryWriter = None,
    epoch: int = 0,
):
    """Run one training epoch.

    Args:
        model:       The neural network to train.
        train_loader: DataLoader for training data.
        optimiser:   Optimizer used for parameter updates.
        writer:      Optional TensorBoard writer for batch-level scalars.
        epoch:       Current epoch index (used to compute global step).

    Returns:
        Tuple ``(avg_log_likelihood, mae_next, mae_remaining, elapsed_seconds)``.
    """
    model.train()
    total_event_ll = total_time_ae = total_rt_ae = 0.0
    total_num_event = total_num_pred = 0
    start = time.time()

    for batch_idx, batch in enumerate(
        tqdm(train_loader, mininterval=2, desc="  - (Training)   ", leave=False)
    ):
        event_time, time_gap, r_time, event_type = (
            x.to(next(model.parameters()).device) for x in batch
        )

        optimiser.zero_grad()
        enc_out, prediction = model(event_type, event_time)

        # Losses
        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)
        ae    = Utils.time_loss(prediction[0], event_time)
        rt_ae = Utils.time_loss(prediction[1], r_time)

        loss = ae          # primary training signal
        loss.backward()
        optimiser.step()

        # Bookkeeping
        n_events = event_type.ne(PAD).sum().item()
        n_preds  = n_events - event_time.shape[0]   # exclude first events
        total_event_ll  += -event_loss.item()
        total_time_ae   +=  ae.item()
        total_rt_ae     +=  rt_ae.item()
        total_num_event += n_events
        total_num_pred  += n_preds

        if writer is not None:
            gs = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Train/Batch/Total_Loss",  loss.item(),        gs)
            writer.add_scalar("Train/Batch/Event_Loss",  event_loss.item(),  gs)
            writer.add_scalar("Train/Batch/Time_AE",     ae.item(),          gs)
            writer.add_scalar("Train/Batch/RT_AE",       rt_ae.item(),       gs)

    mae    = total_time_ae / total_num_pred
    mae_rt = total_rt_ae  / total_num_pred
    elapsed = time.time() - start

    if writer is not None:
        writer.add_scalar("Train/Epoch/Log_Likelihood", total_event_ll / total_num_event, epoch)
        writer.add_scalar("Train/Epoch/MAE",            mae,    epoch)
        writer.add_scalar("Train/Epoch/MAE_RT",         mae_rt, epoch)
        writer.add_scalar("Train/Epoch/Time_Minutes",   elapsed / 60, epoch)

    return total_event_ll / total_num_event, mae, mae_rt, elapsed


def eval_epoch(
    model,
    val_loader,
    writer: SummaryWriter = None,
    epoch: int = 0,
    phase: str = "Validation",
):
    """Run one evaluation epoch (no gradient updates).

    Args:
        model:      The neural network to evaluate.
        val_loader: DataLoader for validation or test data.
        writer:     Optional TensorBoard writer for epoch-level scalars.
        epoch:      Current epoch index.
        phase:      Tag used in TensorBoard (e.g. ``"Validation"`` or ``"Test"``).

    Returns:
        Tuple ``(avg_log_likelihood, mae_next, mae_remaining, elapsed_seconds)``.
    """
    model.eval()
    total_event_ll = total_time_ae = total_rt_ae = 0.0
    total_num_event = total_num_pred = 0
    start = time.time()

    with torch.no_grad():
        for batch in tqdm(val_loader, mininterval=2, desc=f"  - ({phase})   ", leave=False):
            event_time, time_gap, r_time, event_type = (
                x.to(next(model.parameters()).device) for x in batch
            )

            enc_out, prediction = model(event_type, event_time)

            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)
            ae    = Utils.time_loss(prediction[0], event_time)
            rt_ae = Utils.time_loss(prediction[1], r_time)

            n_events = event_type.ne(PAD).sum().item()
            n_preds  = n_events - event_time.shape[0]
            total_event_ll  += -event_loss.item()
            total_time_ae   +=  ae.item()
            total_rt_ae     +=  rt_ae.item()
            total_num_event += n_events
            total_num_pred  += n_preds

    mae    = total_time_ae / total_num_pred
    mae_rt = total_rt_ae  / total_num_pred
    elapsed = time.time() - start

    if writer is not None:
        writer.add_scalar(f"{phase}/Epoch/Log_Likelihood", total_event_ll / total_num_event, epoch)
        writer.add_scalar(f"{phase}/Epoch/MAE",            mae,    epoch)
        writer.add_scalar(f"{phase}/Epoch/MAE_RT",         mae_rt, epoch)
        writer.add_scalar(f"{phase}/Epoch/Time_Minutes",   elapsed / 60, epoch)

    return total_event_ll / total_num_event, mae, mae_rt, elapsed


def train_model(
    model,
    n_epochs: int,
    train_loader,
    val_loader,
    optimiser,
    scheduler,
    model_save_path: str,
    filename: str,
    lr: float,
    batch_size: int,
    results_dir: str = "results",
    log_dir: str = "runs",
):
    """Train for *n_epochs* with checkpoint saving and TensorBoard logging.

    Three checkpoints are maintained independently:
    - ``*_best_mae.pth``  – best next-event time MAE
    - ``*_best_mae_rt.pth`` – best remaining-time MAE
    - ``*_best_ll.pth``   – best event log-likelihood

    Args:
        model:           Neural network to train.
        n_epochs:        Total number of training epochs.
        train_loader:    DataLoader for training data.
        val_loader:      DataLoader for validation data.
        optimiser:       Optimizer instance.
        scheduler:       Learning-rate scheduler instance.
        model_save_path: Base path for checkpoint files (extension is replaced).
        filename:        Dataset stem – used for raw results log and TensorBoard tag.
        lr:              Initial learning rate (recorded as hyperparameter).
        batch_size:      Mini-batch size (recorded as hyperparameter).
        results_dir:     Directory where the per-epoch text log is written.
        log_dir:         Root directory for TensorBoard runs.
    """
    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{filename}_{timestamp}"
    writer          = SummaryWriter(log_dir=os.path.join(log_dir, experiment_name))

    print(f"\n{'='*80}")
    print(f"TensorBoard → {os.path.join(log_dir, experiment_name)}")
    print(f"Run: tensorboard --logdir={log_dir}")
    print(f"{'='*80}\n")

    base = os.path.splitext(model_save_path)[0]
    save_paths = {
        "mae":            f"{base}_best_mae.pth",
        "mae_rt":         f"{base}_best_mae_rt.pth",
        "log_likelihood": f"{base}_best_ll.pth",
    }

    best = {
        "mae":            {"value": float("inf"),  "epoch": 0, "minimize": True},
        "mae_rt":         {"value": float("inf"),  "epoch": 0, "minimize": True},
        "log_likelihood": {"value": float("-inf"), "epoch": 0, "minimize": False},
    }

    print("Checkpoint paths:")
    for k, p in save_paths.items():
        print(f"  {k:20s} → {p}")
    print()

    raw_log_path = os.path.join(results_dir, f"raw_{filename}.txt")
    epoch_times  = []
    total_start  = time.time()

    for i in range(n_epochs):
        print(f"[Epoch {i+1}/{n_epochs}]")
        epoch_start = time.time()

        # ---- train ----
        tr_ll, tr_mae, tr_rt, tr_t = train_epoch(model, train_loader, optimiser, writer, i)
        print(
            f"  Train  ll={tr_ll:8.5f}  MAE={tr_mae:8.5f}  "
            f"MAE_rt={tr_rt:8.5f}  t={tr_t/60:.2f}min"
        )

        # ---- validate ----
        vl_ll, vl_mae, vl_rt, vl_t = eval_epoch(model, val_loader, writer, i)
        print(
            f"  Val    ll={vl_ll:8.5f}  MAE={vl_mae:8.5f}  "
            f"MAE_rt={vl_rt:8.5f}  t={vl_t/60:.2f}min"
        )

        epoch_total = time.time() - epoch_start
        epoch_times.append(epoch_total)

        # ---- raw log ----
        with open(raw_log_path, "a") as f:
            f.write(f"{i}, {vl_ll:8.5f}, {vl_mae:8.5f}, {vl_rt:8.5f}\n")

        # ---- checkpointing ----
        current = {"mae": vl_mae, "mae_rt": vl_rt, "log_likelihood": vl_ll}
        checkpoint = {
            "epoch":               i,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimiser.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_mae":             vl_mae,
            "val_mae_rt":          vl_rt,
            "val_log_likelihood":  vl_ll,
        }

        saved = []
        for metric, info in best.items():
            val = current[metric]
            better = (val < info["value"]) if info["minimize"] else (val > info["value"])
            if better:
                best[metric]["value"] = val
                best[metric]["epoch"] = i + 1
                torch.save(checkpoint, save_paths[metric])
                writer.add_text(f"Model_Save/{metric}", f"Saved at epoch {i+1}", i)
                saved.append(metric)

        if saved:
            print(f"  Saved: {', '.join(saved).upper()}")

        # ---- TensorBoard scalars ----
        writer.add_scalar("Train/Learning_Rate",     optimiser.param_groups[0]["lr"], i)
        writer.add_scalar("Best/MAE",                best["mae"]["value"],            i)
        writer.add_scalar("Best/MAE_RT",             best["mae_rt"]["value"],         i)
        writer.add_scalar("Best/Log_Likelihood",     best["log_likelihood"]["value"], i)
        writer.add_scalar("Time/Epoch_Total_Minutes", epoch_total / 60,               i)
        writer.add_scalar("Time/Cumulative_Hours",    sum(epoch_times) / 3600,        i)
        if len(epoch_times) > 1:
            writer.add_scalar("Time/Average_Epoch_Minutes", np.mean(epoch_times) / 60, i)

        scheduler.step()

    # ---- final summary ----
    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    for metric, info in best.items():
        print(f"  {metric:20s}  {info['value']:.5f}  (epoch {info['epoch']})")
    print(f"  Total time : {total_time/3600:.2f}h  |  Avg/epoch: {np.mean(epoch_times)/60:.2f}min")
    print(f"{'='*80}")

    writer.add_hparams(
        {"lr": lr, "batch_size": batch_size, "epochs": n_epochs},
        {
            "best_val_mae":     best["mae"]["value"],
            "best_val_mae_rt":  best["mae_rt"]["value"],
            "best_val_ll":      best["log_likelihood"]["value"],
            "total_train_hours": total_time / 3600,
            "avg_epoch_min":    np.mean(epoch_times) / 60,
        },
    )
    writer.close()
    print(f"\nTensorBoard logs: {os.path.join(log_dir, experiment_name)}")