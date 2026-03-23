#!/usr/bin/env python3
"""
测试脚本 - 完整版（包含 gen_prefix 实现）
"""

import pathlib
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import argparse, os, sys
from transformer.model import Transformer
from transformer.Constants import *
from tqdm import tqdm
import Utils
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

import random
from typing import Dict, List, Tuple
import time

def set_all_seeds(seed_value): 
    os.environ['PYTHONHASHSEED'] = str(seed_value) 
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.Generator().manual_seed(seed_value)

pytorch_generator = set_all_seeds(42)

EvaluationDict = Dict[int, List[Tuple[float, int]]]

def update_evaluation_mean(eval_dict: EvaluationDict, prefix_length: int, new_evaluation_value: float) -> None:
    if prefix_length not in eval_dict:
        eval_dict[prefix_length] = [new_evaluation_value, 1]
    else:
        current_mean, current_count = eval_dict[prefix_length]
        current_sum = current_mean * current_count
        new_sum = current_sum + new_evaluation_value
        new_count = current_count + 1
        new_mean = new_sum / new_count
        eval_dict[prefix_length] = [new_mean, new_count]


def gen_prefix(batch):
    """
    将 batch 中的每个序列展开成所有可能的前缀
    
    Example:
        Input:  [[a, b, c], [x, y]]
        Output: [[a, 0, 0], [a, b, 0], [a, b, c], [x, 0], [x, y]]
    
    这样可以一次性评估所有前缀长度，与 benchmark papers 保持一致。
    
    Args:
        batch: tuple of (event_time, time_gap, r_time, event_type)
               shape: [batch_size, max_seq_len]
    
    Returns:
        expanded_batch: tuple of 4 tensors
                       shape: [total_prefixes, max_seq_len]
    """
    event_time, time_gap, r_time, event_type = batch
    
    batch_size, max_len = event_type.shape
    device = event_type.device
    
    all_event_time = []
    all_time_gap = []
    all_r_time = []
    all_event_type = []
    
    for i in range(batch_size):
        # 找到序列实际长度（不包括 PAD）
        seq_len = (event_type[i] != PAD).sum().item()
        
        if seq_len == 0:
            continue  # 跳过空序列
        
        # 生成所有前缀：长度从 1 到 seq_len
        for k in range(1, seq_len + 1):
            # 创建前缀：保留前 k 个元素，其余填 PAD
            prefix_event_type = torch.full((max_len,), PAD, dtype=event_type.dtype, device=device)
            prefix_event_type[:k] = event_type[i, :k]
            
            prefix_event_time = torch.full((max_len,), PAD, dtype=event_time.dtype, device=device)
            prefix_event_time[:k] = event_time[i, :k]
            
            prefix_time_gap = torch.full((max_len,), PAD, dtype=time_gap.dtype, device=device)
            prefix_time_gap[:k] = time_gap[i, :k]
            
            prefix_r_time = torch.full((max_len,), PAD, dtype=r_time.dtype, device=device)
            prefix_r_time[:k] = r_time[i, :k]
            
            all_event_type.append(prefix_event_type)
            all_event_time.append(prefix_event_time)
            all_time_gap.append(prefix_time_gap)
            all_r_time.append(prefix_r_time)
    
    # Stack 成新的 batch
    if len(all_event_type) == 0:
        return (
            torch.empty(0, max_len, dtype=event_time.dtype, device=device),
            torch.empty(0, max_len, dtype=time_gap.dtype, device=device),
            torch.empty(0, max_len, dtype=r_time.dtype, device=device),
            torch.empty(0, max_len, dtype=event_type.dtype, device=device),
        )
    
    expanded_event_time = torch.stack(all_event_time, dim=0)
    expanded_time_gap = torch.stack(all_time_gap, dim=0)
    expanded_r_time = torch.stack(all_r_time, dim=0)
    expanded_event_type = torch.stack(all_event_type, dim=0)
    
    return expanded_event_time, expanded_time_gap, expanded_r_time, expanded_event_type


def df_to_json(full_filename, filename):
    case_col = "CaseID"
    time_col = "Timestamp"
    act_col = "Activity"
    rt_col = "remaining_time"
    
    test_df = pd.read_csv(os.path.join(directory, "test_" + filename + extension))
    train_val_df = pd.read_csv(os.path.join(directory, full_filename + extension))

    test_df[time_col] = pd.to_datetime(test_df[time_col])
    train_val_df[time_col] = pd.to_datetime(train_val_df[time_col])

    act_uni_train_val = np.unique(train_val_df[act_col]) 
    dim_process = act_uni_train_val.size + 1
    event_types = {name: idx for idx, name in enumerate(act_uni_train_val)}
    event_types['END'] = dim_process
    max_len = 0

    def gen_extend_features(df, max_len):
        g = df.groupby(case_col, sort=False)
        df["time_since_start"] = (df[time_col] - g[time_col].transform("min")).dt.total_seconds().div(86400.0).astype("float32")
        df["time_since_last_event"] = (df[time_col] - g[time_col].shift(1)).fillna(pd.Timedelta(0)).dt.total_seconds().div(86400.0).astype("float32")
        max_t = g[time_col].transform('max')               
        df[rt_col] = ((max_t - df[time_col]).dt.total_seconds().div(86400.0).astype('float32'))
        if len(g) > max_len:
            max_len = len(g)
        
        act_codes = df[act_col].map(event_types) 
        if act_codes.isna().any():
            act_codes = act_codes.fillna(-1) 
        df["type_event"] = act_codes
        return df, max_len

    test_df, max_len = gen_extend_features(test_df, max_len)

    def df_to_res(df):
        data = []
        for _, grp in df.groupby(case_col, sort=False):
            seq = [
                {
                    "time_since_start": float(t0), 
                    "time_since_last_event": float(dt),
                    "remaining_time": float(rt),
                    "type_event": int(act),
                }
                for t0, dt, rt, act in zip(
                    grp["time_since_start"].values,
                    grp["time_since_last_event"].values,
                    grp["remaining_time"].values,
                    grp["type_event"].values
                )
            ]
            data.append(seq)
        return data

    test_res = df_to_res(test_df)
    test_out = {
        "dim_process": dim_process,
        "max_length": max_len, 
        'test': test_res
    }
    return test_out


def test_model(model, test_loader, filename):
    model.to(device)
    model.eval()
    total_time_ae = 0
    total_rt_ae = 0
    total_num_event = 0
    total_num_pred = 0

    ae_evaluation_results: EvaluationDict = {}
    rt_ae_evaluation_results: EvaluationDict = {}

    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Testing)   ', leave=False):
            # expand to all prefixes
            batch = gen_prefix(batch)

            if batch[0].size(0) == 0:  # skip empty batch
                continue
            
            event_time, time_gap, r_time, event_type = map(lambda x: x.to(device), batch)
            enc_out, prediction = model(event_type, event_time)

            ae = Utils.time_loss(prediction[0], event_time)
            rt_ae = Utils.time_loss(prediction[1], r_time)

            # for each prefix calculate per-position MAE
            for idx in range(event_type.size(0)):  # iterate over batch prefix
                # find the actual length of the current prefix
                prefix_len = (event_type[idx] != PAD).sum().item()
                
                if prefix_len == 0:
                    continue
                
                # only the last k（prefix_len - 1）
                k = prefix_len - 1
                
                t_pred_k = prediction[0][idx, k].cpu().numpy()
                rt_pred_k = prediction[1][idx, k].cpu().numpy()
                t_y_k = event_time[idx, k].cpu().numpy()
                rt_y_k = r_time[idx, k].cpu().numpy()
                
                # skip PAD positions
                if t_y_k != PAD and rt_y_k != PAD:
                    ae_k = abs(t_y_k - t_pred_k)
                    rt_ae_k = abs(rt_y_k - rt_pred_k)
                    
                    update_evaluation_mean(ae_evaluation_results, k, ae_k)
                    update_evaluation_mean(rt_ae_evaluation_results, k, rt_ae_k)

            total_time_ae += ae.item()
            total_rt_ae += rt_ae.item()
            total_num_event += event_type.ne(PAD).sum().item()
            total_num_pred += event_type.ne(PAD).sum().item() - event_time.shape[0]
            
    mae = total_time_ae / total_num_pred if total_num_pred > 0 else 0
    mae_rt = total_rt_ae / total_num_pred if total_num_pred > 0 else 0

    inference_time = time.time() - start_time

    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"MAE (days): {mae:.5f}")
    print(f"MAE Remaining Time (days): {mae_rt:.5f}")
    print("="*80)
    print(f"number of events: {total_num_event}")
    print(f"number of predictions: {total_num_pred}")
    print(f"inference time (seconds): {inference_time:.2f}")

    with open(os.path.join("results", "test_" + filename + ".txt"), 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEST RESULTS\n")
        f.write("="*80 + "\n")
        f.write("mae_in_days: " + str(mae) + "\n")
        f.write("mae_in_days rt: " + str(mae_rt) + "\n")
        f.write("\n" + "="*80 + "\n")
        f.write("PER-PREFIX EVALUATION\n")
        f.write("="*80 + "\n")
        f.write("MAE per prefix: " + str(ae_evaluation_results) + "\n")
        f.write("MAE remaining time per prefix: " + str(rt_ae_evaluation_results) + "\n")
        f.write(f"inference time (seconds): {inference_time:.2f}\n")

    return mae, mae_rt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained model")
    parser.add_argument("--fold_dataset", required=True)
    parser.add_argument("--full_dataset", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    
    arguments = parser.parse_args()
    batch_size = arguments.batch_size
    device = arguments.device
    logfile = arguments.fold_dataset
    log_filename = Path(logfile).stem

    os.makedirs("results", exist_ok=True)

    directory = Path(logfile).parent
    filename = Path(logfile).stem
    full_filename = Path(arguments.full_dataset).stem
    extension = ".csv"

    print("="*80)
    print("LOADING CHECKPOINT")
    print("="*80)
    print(f"Path: {arguments.model_path}")
    
    if not os.path.exists(arguments.model_path):
        print(f"ERROR: Model file not found")
        sys.exit(-1)
    
    checkpoint = torch.load(arguments.model_path, map_location=device)
    
    emb_weight_shape = checkpoint['model_state_dict']['encoder.event_emb.weight'].shape
    linear_weight_shape = checkpoint['model_state_dict']['linear.weight'].shape
    
    emb_num_types = emb_weight_shape[0]
    d_model = emb_weight_shape[1]
    
    print(f"Checkpoint embedding shape: {emb_weight_shape}")
    print(f"Checkpoint linear shape:    {linear_weight_shape}")
    
    # Transformer 内部会对 num_types +1，所以传入时减 1
    num_types_to_pass = emb_num_types - 1
    
    print(f"\n→ Passing num_types={num_types_to_pass} to Transformer (will become {emb_num_types} internally)")
    
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    
    model = Transformer(
        num_types=num_types_to_pass,
        d_model=d_model,
        d_rnn=256,
        d_inner=128,
        n_layers=4,
        n_head=4,
        d_k=16,
        d_v=16,
        dropout=0.1,
    )
    model.to(torch.device(device))
    
    actual_emb_shape = model.encoder.event_emb.weight.shape
    print(f"Initialized embedding shape: {actual_emb_shape}")
    
    if actual_emb_shape != emb_weight_shape:
        print(f"\n⚠ ERROR: Shape mismatch!")
        print(f"  Expected: {emb_weight_shape}")
        print(f"  Got:      {actual_emb_shape}")
        sys.exit(-1)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✓ Model loaded successfully!")
        if 'epoch' in checkpoint:
            print(f"  - Trained for {checkpoint['epoch'] + 1} epochs")
        if 'val_loss' in checkpoint:
            print(f"  - Val MAE: {checkpoint['val_loss']:.5f}")
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(-1)
    
    print("\n" + "="*80)
    print("LOADING TEST DATA")
    print("="*80)
    
    test_out = df_to_json(full_filename, log_filename)
    test_data = test_out["test"]
    
    import torch.utils.data
    
    class EventData(torch.utils.data.Dataset):
        def __init__(self, data):
            self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
            self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
            self.r_time = [[elem['remaining_time'] for elem in inst] for inst in data]
            self.activity = [[elem['type_event']+1 for elem in inst] for inst in data]
            self.length = len(data)

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return self.time[idx], self.time_gap[idx], self.r_time[idx], self.activity[idx]

    def pad_time(insts):
        max_len = max(len(inst) for inst in insts)
        batch_seq = np.array([inst + [PAD] * (max_len - len(inst)) for inst in insts])
        return torch.tensor(batch_seq, dtype=torch.float32)

    def pad_type(insts):
        max_len = max(len(inst) for inst in insts)
        batch_seq = np.array([inst + [PAD] * (max_len - len(inst)) for inst in insts])
        return torch.tensor(batch_seq, dtype=torch.long)

    def collate_fn(insts):
        time, time_gap, rt, activity = list(zip(*insts))
        return pad_time(time), pad_time(time_gap), pad_time(rt), pad_type(activity)

    def get_dataloader(data, batch_size, shuffle=False):
        ds = EventData(data)
        return torch.utils.data.DataLoader(
            ds, num_workers=2, batch_size=batch_size,
            collate_fn=collate_fn, shuffle=shuffle)

    test_loader = get_dataloader(test_data, batch_size, shuffle=False)
    
    print("\n" + "="*80)
    print("RUNNING TEST (with prefix expansion)")
    print("="*80)
    
    
    test_model(model, test_loader, filename)
    
    print("\n" + "="*80)
    print("✓ TESTING COMPLETED")
    print("="*80)