#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import copy
import pandas as pd
import time
import psutil
import gc
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

from utils.data_loader import (
    load_train_data, load_test_data,
    split_train_test, split_train_test2, 
    dataframe_to_list, list_to_dataframe
)
from utils.config import Config, DEFAULT_CONFIG
from models.autoint import AutoInt

config = DEFAULT_CONFIG
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for directory in [config.experiment.figures_dir, config.experiment.results_dir, config.experiment.models_dir]:
    os.makedirs(directory, exist_ok=True)

performance_results = []

SPLIT_METHOD_MAP = {
    "全局随机划分": "global_random",
    "按用户划分": "user_based"
}

def get_model_size(model):
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024**2

def _prepare_dataset_with_ratings(u_map, i_map, df):
    users = torch.tensor([u_map[u] for u in df['user']], dtype=torch.long)
    items = torch.tensor([i_map[i] for i in df['item']], dtype=torch.long)
    ratings = torch.tensor([r for r in df['rating']], dtype=torch.float32)
    return TensorDataset(users, items, ratings)

def train_model(model, name, split_method_name, split_method_en, train_loader, val_loader):
    model.to(DEVICE)
    
    lr = 1e-4  # AutoInt适合的学习率
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.model.weight_decay)
    loss_fn = torch.nn.MSELoss()

    train_losses, val_losses, val_aucs, val_rmses, val_maes = [], [], [], [], []
    
    best_val_auc = -1.0
    early_stopping_counter = 0
    best_model_state = None

    start_time = time.time()
    memory_before = get_memory_usage()
    model_size = get_model_size(model)

    print(f"开始训练 {name} 模型...")
    print(f"   学习率: {lr}, 设备: {DEVICE}")

    for epoch in range(config.model.epochs):
        model.train()
        total_train_loss = 0.0
        epoch_start_time = time.time()
        
        for u, i, r in tqdm(train_loader, desc=f'[{name}] Epoch {epoch+1}/{config.model.epochs} (Training)'):
            u, i, r = u.to(DEVICE), i.to(DEVICE), r.to(DEVICE)
            optim.zero_grad()
            
            pred = model(u, i)
            loss = loss_fn(pred, r)
            loss.backward()
            optim.step()
            total_train_loss += loss.item() * len(u)
        
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        all_preds, all_reals = [], []
        with torch.no_grad():
            for u, i, r in tqdm(val_loader, desc=f'[{name}] Epoch {epoch+1}/{config.model.epochs} (Validating)'):
                u, i, r = u.to(DEVICE), i.to(DEVICE), r.to(DEVICE)
                pred = model(u, i)
                
                total_val_loss += loss_fn(pred, r).item() * len(u)
                all_preds.extend(pred.cpu().numpy())
                all_reals.extend(r.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        current_val_auc = float('nan')
        if len(set(all_reals)) > 1:
            bin_label = (np.array(all_reals) > np.mean(all_reals)).astype(int)
            current_val_auc = roc_auc_score(bin_label, all_preds)
        val_aucs.append(current_val_auc)
        
        current_val_rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_reals)) ** 2))
        current_val_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_reals)))
        val_rmses.append(current_val_rmse)
        val_maes.append(current_val_mae)

        print(f'  >> Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val AUC={current_val_auc:.4f} | Val RMSE={current_val_rmse:.4f} | Val MAE={current_val_mae:.4f} | Time: {epoch_time:.2f}s')

        if not math.isnan(current_val_auc) and current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            early_stopping_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f'  >> 在 epoch {epoch+1} 保存最佳模型，Val AUC: {best_val_auc:.4f}')
        else:
            early_stopping_counter += 1
            print(f'  >> Val AUC 连续 {early_stopping_counter}/{config.model.patience} epochs 未改善')
            if early_stopping_counter >= config.model.patience:
                print(f'  >> 早停触发于 epoch {epoch+1}')
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'  >> 已加载基于验证AUC的最佳模型')
    else:
        print(f'  >> 未找到改善，返回最终模型状态')

    total_time = time.time() - start_time
    memory_after = get_memory_usage()
    memory_overhead = memory_after - memory_before

    final_auc = val_aucs[-1] if val_aucs else float('nan')
    final_rmse = val_rmses[-1] if val_rmses else float('nan')
    final_mae = val_maes[-1] if val_maes else float('nan')
    final_loss = val_losses[-1] if val_losses else float('nan')
    
    performance_results.append({
        'split_method': split_method_name,
        'split_method_en': split_method_en,
        'model': name,
        'model_type': 'AutoInt',
        'best_val_auc': best_val_auc,
        'final_val_loss': final_loss,
        'final_rmse': final_rmse,
        'final_mae': final_mae,
        'epochs_trained': len(train_losses),
        'total_time_s': total_time,
        'time_per_epoch_s': total_time / len(train_losses) if train_losses else 0,
        'model_size_mb': model_size,
        'memory_overhead_mb': max(0, memory_overhead)
    })

    print(f'  >> 训练完成: {total_time:.2f}s ({total_time/len(train_losses):.2f}s/epoch)')
    print(f'  >> 模型大小: {model_size:.2f}MB, 内存开销: {memory_overhead:.2f}MB')

    plot_training_curves(name, split_method_en, train_losses, val_losses, val_aucs, val_rmses, val_maes)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model

def plot_training_curves(name, split_method_en, train_losses, val_losses, val_aucs, val_rmses, val_maes):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 4, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 2)
    valid_aucs = [auc for auc in val_aucs if not math.isnan(auc)]
    if valid_aucs:
        plt.plot(range(len(valid_aucs)), valid_aucs, label='Val AUC', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('AUC Curve')
        plt.legend()
        plt.grid(True)

    plt.subplot(1, 4, 3)
    plt.plot(val_rmses, label='Val RMSE', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 4)
    plt.plot(val_maes, label='Val MAE', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE Curve')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f'{name} ({split_method_en}) - Training Metrics', fontsize=14)
    fig_path = os.path.join(config.experiment.figures_dir, f'{name}_{split_method_en}_training_curves.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  >> 训练曲线已保存: {fig_path}')

def predict_and_save(model, name, raw_test_df, u_map, i_map, split_method_en, mean_rating):
    model.eval()
    model.to(DEVICE)

    outfile = os.path.join(config.experiment.results_dir, f'{name}_{split_method_en}_predictions.txt')
    with open(outfile, 'w', encoding='utf-8') as f:
        last_user, items = None, []
        for idx, row in raw_test_df.iterrows():
            u_raw, i_raw = row['user'], row['item']
            if last_user is None:
                last_user = u_raw
            if u_raw != last_user:
                f.write(f"{last_user}|{len(items)}\n")
                for itm, score in items:
                    f.write(f"{itm} {int(round(score))}\n")
                last_user, items = u_raw, []

            u_idx = u_map.get(u_raw, None)
            i_idx = i_map.get(i_raw, None)
            if u_idx is not None and i_idx is not None:
                with torch.no_grad():
                    pred = model(
                        torch.tensor([u_idx], device=DEVICE),
                        torch.tensor([i_idx], device=DEVICE)
                    ).item()
                if math.isnan(pred):
                    pred = mean_rating
            else:
                pred = mean_rating
            items.append((i_raw, pred))

        if items:
            f.write(f"{last_user}|{len(items)}\n")
            for itm, score in items:
                f.write(f"{itm} {int(round(score))}\n")

    print(f'  >> 预测结果已保存: {outfile}')

def run_experiment_with_split_method(split_method, split_method_name):
    global train_loader, val_loader, train_df
    
    split_method_en = SPLIT_METHOD_MAP[split_method_name]
    
    print(f'\n{"="*80}')
    print(f'运行AutoInt实验 - 数据划分方法: {split_method_name} ({split_method_en})')
    print(f'{"="*80}')
    
    if split_method == 1:
        print('使用全局随机划分 (9:1 比例)')
        data_list = dataframe_to_list(train_df)
        train_list, val_list = split_train_test(data_list, train_ratio=0.9)
        train_df_actual = list_to_dataframe(train_list)
        val_df = list_to_dataframe(val_list)
    elif split_method == 2:
        print('使用按用户划分方法')
        data_list = dataframe_to_list(train_df)
        train_list, val_list = split_train_test2(data_list)
        train_df_actual = list_to_dataframe(train_list)
        val_df = list_to_dataframe(val_list)

    print(f'训练集大小: {len(train_df_actual)}, 验证集大小: {len(val_df)}')

    train_set = _prepare_dataset_with_ratings(u_map, i_map, train_df_actual)
    val_set = _prepare_dataset_with_ratings(u_map, i_map, val_df)

    train_loader = DataLoader(train_set, batch_size=config.model.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.model.batch_size*2, shuffle=False, num_workers=0, pin_memory=True)

    print(f'\n训练AutoInt模型')
    print('=' * 40)
    
    model = AutoInt(len(u_map), len(i_map), embed_dim=config.model.latent_dim)
    trained_model = train_model(model, 'AutoInt', split_method_name, split_method_en, train_loader, val_loader)
    predict_and_save(trained_model, 'AutoInt', final_test_df, u_map, i_map, split_method_en, train_df['rating'].mean())

def print_performance_comparison():
    print(f'\n{"="*80}')
    print(' ' * 25 + 'AutoInt模型性能报告')
    print(f'{"="*80}')
    
    df_results = pd.DataFrame(performance_results)
    
    for split_method in df_results['split_method'].unique():
        print(f'\n{"[ " + split_method + " ]":^80}')
        print(f'{"-"*80}')
        subset = df_results[df_results['split_method'] == split_method].copy()
        
        print(f"{'模型':<15} {'RMSE':<8} {'AUC':<8} {'MAE':<8} {'训练时间':<12} {'模型大小':<12}")
        print(f'{"-"*80}')
        
        for _, row in subset.iterrows():
            model_name = row['model']
            rmse = f"{row['final_rmse']:.3f}"
            auc = f"{row['best_val_auc']:.3f}"
            mae = f"{row['final_mae']:.3f}"
            time_info = f"{row['time_per_epoch_s']:.2f}s/epoch"
            size_info = f"{row['model_size_mb']:.2f}MB"
            
            print(f"{model_name:<15} {rmse:<8} {auc:<8} {mae:<8} {time_info:<12} {size_info:<12}")
    
    results_file = os.path.join(config.experiment.results_dir, 'autoint_performance_comparison.csv')
    df_results.to_csv(results_file, index=False, float_format='%.4f')
    print(f'\n详细性能比较结果已保存: {results_file}')
    print(f'{"="*80}\n')

if __name__ == '__main__':
    train_df = load_train_data()
    final_test_df = load_test_data()

    users = sorted(train_df['user'].unique())
    items = sorted(train_df['item'].unique())
    u_map = {u: i for i, u in enumerate(users)}
    i_map = {v: i for i, v in enumerate(items)}

    run_experiment_with_split_method(1, "全局随机划分")
    run_experiment_with_split_method(2, "按用户划分")
    
    print_performance_comparison()
    
    print("AutoInt完成！") 