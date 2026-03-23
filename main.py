
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

from preprocess import df_to_dict, get_dataloader
from trainer import train_model
from utils import set_all_seeds
from transformer.model import Transformer



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="THPPPM – Transformer-based remaining-time prediction"
    )
    parser.add_argument("--fold_dataset", required=True,
                        help="Path stem for train_/val_/test_ CSV splits "
                             "(e.g. data/bpic2012/fold1/bpic2012)")
    parser.add_argument("--full_dataset", required=True,
                        help="Path stem for combined train+val CSV "
                             "(e.g. data/bpic2012/full/bpic2012)")
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--epoch",        type=int,   default=200)
    parser.add_argument("--lr",           type=float, default=0.01)
    parser.add_argument("--device",       default="cuda",
                        help="'cuda' or 'cpu'")
    parser.add_argument("--log_dir",      default="runs",
                        help="Root directory for TensorBoard logs")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_types: int, device: torch.device) -> Transformer:
    """Instantiate the Transformer with default hyper-parameters."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    # ---- paths ----
    directory     = str(Path(args.fold_dataset).parent)
    fold_filename = Path(args.fold_dataset).stem
    full_filename = Path(args.full_dataset).stem

    os.makedirs("results",      exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs(args.log_dir,   exist_ok=True)

    output_file = os.path.join("results", f"{fold_filename}.txt")
    with open(output_file, "w") as f:
        f.write(f"Start: {time.strftime('%d.%m.%y-%H.%M', time.localtime())}\n")

    # ---- device ----
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available – falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    # ---- data ----
    print("Loading data …")
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

    train_loader = get_dataloader(train_out["train"], args.batch_size, shuffle=True)
    val_loader   = get_dataloader(val_out["val"],     args.batch_size, shuffle=False)

    # ---- model ----
    model           = build_model(num_types, device)
    model_save_path = os.path.join("saved_models", f"{fold_filename}_model.pth")

    # ---- train ----
    print("\n" + "=" * 80)
    print("TRAINING")
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

    print("\nDone. To view TensorBoard:")
    print(f"  tensorboard --logdir={args.log_dir}")


if __name__ == "__main__":
    main()