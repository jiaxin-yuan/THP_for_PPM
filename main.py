
import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.optim as optim

from preprocess.dataset import df_to_dict, get_dataloader
from trainer.train import train_model, eval_epoch
from utils.reproducibility import set_all_seeds
from transformer.model import Transformer


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--fold_dataset", help="Raw dataset to prepare", required=True)
    parser.add_argument("--full_dataset", help="Raw dataset to prepare", required=True)


    # training hyper-parameters
    parser.add_argument("--batch_size", type=int,   default=32,   help="Batch size")
    parser.add_argument("--epoch",      type=int,   default=1,  help="Number of epochs")
    parser.add_argument("--lr",         type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed",       type=int,   default=42,   help="Random seed")

    # runtime
    parser.add_argument("--device",  default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument("--log_dir", default="runs", help="TensorBoard log directory")

    # mode flags
    parser.add_argument("--train", action="store_true", help="Run training loop")
    parser.add_argument("--test",  action="store_true", help="Run test evaluation")
    parser.add_argument("--model_path", default=None,
                        help="Checkpoint path for --test (default: best MAE checkpoint)")

    args = parser.parse_args()

    if not (args.train or args.test):
        parser.error("At least one of --train or --test is required.")

    return args


def setup_directories(log_dir: str) -> None:
    """Create output directories if they do not exist.

    Args:
        log_dir: Root directory for TensorBoard logs.
    """
    os.makedirs("results",      exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs(log_dir,        exist_ok=True)


def init_result_file(output_file: str) -> None:
    """Write a timestamped header to the per-run result text file.

    Args:
        output_file: Path to the result text file.
    """
    current_time = time.strftime("%d.%m.%y-%H.%M", time.localtime())
    with open(output_file, "w") as f:
        f.write(f"Starting time: {current_time}\n")


def resolve_device(requested: str) -> torch.device:
    """Return a torch.device, falling back to CPU if CUDA is unavailable.

    Args:
        requested: "cuda" or "cpu".

    Returns:
        Resolved torch.device.
    """
    if requested == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available – falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def load_data(directory: str, fold_filename: str, full_filename: str, batch_size: int):
    """Load and preprocess data, return DataLoaders and metadata.

    Args:
        directory:      Folder that contains the CSV splits.
        fold_filename:  Stem used for train_/val_/test_ prefixes.
        full_filename:  Stem for the combined train+val CSV.
        batch_size:     Mini-batch size.

    Returns:
        train_loader: DataLoader for training data.
        val_loader:   DataLoader for validation data.
        test_loader:  DataLoader for test data.
        num_types:    Number of distinct activity types.
    """
    print("Loading and preprocessing data ...")
    train_out, val_out, test_out = df_to_dict(
        directory=directory,
        fold_filename=fold_filename,
        full_filename=full_filename,
    )
    num_types = train_out["dim_process"]

    print(f"  Activity types : {num_types}")
    print(f"  Train cases    : {len(train_out['train'])}")
    print(f"  Val   cases    : {len(val_out['val'])}")
    print(f"  Test  cases    : {len(test_out['test'])}")

    train_loader = get_dataloader(train_out["train"], batch_size, shuffle=True)
    val_loader   = get_dataloader(val_out["val"],     batch_size, shuffle=False)
    test_loader  = get_dataloader(test_out["test"],   batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_types


def build_model(num_types: int, device: torch.device) -> Transformer:
    """Instantiate and move the Transformer to device.

    Args:
        num_types: Vocabulary size (number of distinct activity types).
        device:    Target device.

    Returns:
        Initialised Transformer.
    """
    model = Transformer(
        num_types=num_types,
        d_model=36,
        d_rnn=256,
        d_inner=128,
        n_layers=4,
        n_head=4,
        d_k=16,
        d_v=16,
        dropout=0.1,
    )
    model.to(device)
    return model


def run_training(
    model: Transformer,
    train_loader,
    val_loader,
    model_save_path: str,
    fold_filename: str,
    args: argparse.Namespace,
) -> None:
    """Set up optimizer / scheduler and run the full training loop.

    Args:
        model:            The Transformer to train.
        train_loader:     DataLoader for training data.
        val_loader:       DataLoader for validation data.
        model_save_path:  Base path for checkpoint files.
        fold_filename:    Dataset stem used for naming logs and results.
        args:             Parsed CLI arguments (uses epoch, lr, batch_size, log_dir).
    """
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    optimiser = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.99),
        eps=1e-5,
    )
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.5)

    train_model(
        model=model,
        n_epochs=args.epoch,
        train_loader=train_loader,
        val_loader=val_loader,
        optimiser=optimiser,
        scheduler=scheduler,
        model_save_path=model_save_path,
        filename=fold_filename,
        lr=args.lr,
        batch_size=args.batch_size,
        results_dir="results",
        log_dir=args.log_dir,
    )

    print("Training completed!")
    print(f"\nTo view TensorBoard:  tensorboard --logdir={args.log_dir}")
    print("Then open: http://localhost:6006")


def run_testing(
    model: Transformer,
    test_loader,
    model_save_path: str,
    output_file: str,
    args: argparse.Namespace,
) -> None:
    """Load a checkpoint and evaluate the model on the test split.

    Args:
        model:            The Transformer to evaluate.
        test_loader:      DataLoader for test data.
        model_save_path:  Base path used to infer the default checkpoint path.
        output_file:      Path to the per-run result text file.
        args:             Parsed CLI arguments (uses model_path, device).
    """
    print("\n" + "=" * 80)
    print("STARTING TESTING")
    print("=" * 80)

    # resolve checkpoint path
    if args.model_path:
        ckpt_path = args.model_path
    else:
        base      = os.path.splitext(model_save_path)[0]
        ckpt_path = f"{base}_best_mae.pth"

    if not os.path.isfile(ckpt_path):
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint (epoch {checkpoint['epoch'] + 1}): {ckpt_path}")

    test_ll, test_mae, test_mae_rt, elapsed = eval_epoch(
        model, test_loader, phase="Test"
    )

    print(f"\nTest results:")
    print(f"  Log-likelihood      : {test_ll:.5f}")
    print(f"  MAE  (next event)   : {test_mae:.5f}  days")
    print(f"  MAE  (remaining)    : {test_mae_rt:.5f}  days")
    print(f"  Elapsed             : {elapsed / 60:.2f} min")

    with open(output_file, "a") as f:
        f.write(
            f"TEST  ll={test_ll:.5f}  mae={test_mae:.5f}  mae_rt={test_mae_rt:.5f}\n"
        )


def main() -> None:
    """Orchestrate argument parsing, data loading, model building, and
    training / testing according to the CLI flags."""

    # ---- setup ----
    args          = parse_args()
    set_all_seeds(args.seed)

    directory     = str(Path(args.fold_dataset).parent)
    fold_filename = Path(args.fold_dataset).stem
    full_filename = Path(args.full_dataset).stem
    output_file   = os.path.join("results", f"{fold_filename}.txt")

    setup_directories(args.log_dir)
    init_result_file(output_file)

    device = resolve_device(args.device)

    # ---- data ----
    train_loader, val_loader, test_loader, num_types = load_data(
        directory, fold_filename, full_filename, args.batch_size
    )

    # ---- model ----
    model           = build_model(num_types, device)
    model_save_path = os.path.join("saved_models", f"{fold_filename}_best_model.pth")

    # ---- train ----
    if args.train:
        run_training(model, train_loader, val_loader,
                     model_save_path, fold_filename, args)

    # ---- test ----
    if args.test:
        run_testing(model, test_loader, model_save_path, output_file, args)

    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()