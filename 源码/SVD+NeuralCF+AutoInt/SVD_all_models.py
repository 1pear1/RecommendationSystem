import os, math, numpy as np, torch, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import copy
import sys
import pandas as pd
import time
import psutil
import gc

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  
plt.rcParams['axes.unicode_minus'] = False  

# 假设这些路径和模型定义文件都在
from utils.data_loader import (
    load_train_data, load_test_data,
    prepare_train_data, prepare_test_data, 
    split_train_test, split_train_test2, dataframe_to_list, list_to_dataframe 
)
from models.classical_svd import ClassicalSVD
from models.funk_svd      import FunkSVD
from models.bias_svd      import BiasSVD
from models.svd_attr      import SVDattrModel 

SAVE_FIG_DIR   = 'figures'
SAVE_RES_DIR   = 'results'
os.makedirs(SAVE_FIG_DIR, exist_ok=True)
os.makedirs(SAVE_RES_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LATENT = 32
EPOCHS = 100
BATCH  = 512
LR     = 1e-3
WD     = 1e-5
PATIENCE = 10 

# 存储性能结果的全局变量
performance_results = []

# 划分方法映射（中文到英文）
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

def train_model(model, name, split_method_name, split_method_en):
    model.to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = torch.nn.MSELoss()

    train_losses, val_losses, val_aucs, val_rmses, val_maes = [], [], [], [], []
    
    best_val_auc = -1.0 
    early_stopping_counter = 0
    best_model_state = None

    start_time = time.time()
    memory_before = get_memory_usage()
    model_size = get_model_size(model)

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        epoch_start_time = time.time()
        
        for u,i,r in tqdm(train_loader, desc=f'[{name}] Epoch {epoch+1}/{EPOCHS} (Training)'):
            u,i,r = u.to(DEVICE), i.to(DEVICE), r.to(DEVICE)
            optim.zero_grad()
            
            if isinstance(model, SVDattrModel):
                pred = model.forward_without_attrs(u, i)  
            else:
                pred = model(u,i)
                
            loss = loss_fn(pred, r)
            loss.backward()
            optim.step()
            total_train_loss += loss.item()*len(u)
        
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        all_preds, all_reals = [], []
        with torch.no_grad():
            for u,i,r in tqdm(val_loader, desc=f'[{name}] Epoch {epoch+1}/{EPOCHS} (Validating)'):
                u,i,r = u.to(DEVICE), i.to(DEVICE), r.to(DEVICE)
                
                if isinstance(model, SVDattrModel):
                    pred = model.forward_without_attrs(u, i)  
                else:
                    pred = model(u,i)
                
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
        current_val_mae  = np.mean(np.abs(np.array(all_preds) - np.array(all_reals)))
        val_rmses.append(current_val_rmse)
        val_maes.append(current_val_mae)

        print(f'  >> Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val AUC={current_val_auc:.4f} | Val RMSE={current_val_rmse:.4f} | Val MAE={current_val_mae:.4f} | Time: {epoch_time:.2f}s')

        if not math.isnan(current_val_auc) and current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            early_stopping_counter = 0
            best_model_state = copy.deepcopy(model.state_dict()) # 保存当前最佳模型权重
            print(f'  >> Saved best model state at epoch {epoch+1} with Val AUC: {best_val_auc:.4f}')
        else:
            early_stopping_counter += 1
            print(f'  >> Val AUC did not improve for {early_stopping_counter}/{PATIENCE} epochs.')
            if early_stopping_counter >= PATIENCE:
                print(f'  >> Early stopping triggered at epoch {epoch+1}.')
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'  >> Loaded best model state based on validation AUC.')
    else:
        print(f'  >> No improvement found, returning final model state.')

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
        'best_val_auc': best_val_auc,
        'final_val_loss': final_loss,
        'final_rmse': final_rmse,
        'final_mae': final_mae,
        'epochs_trained': len(train_losses),
        'total_time_s': total_time,
        'time_per_epoch_s': total_time / len(train_losses) if train_losses else 0,
        'model_size_mb': model_size,
        'memory_overhead_mb': max(0, memory_overhead)  # 确保不为负数
    })

    print(f'  >> Training completed in {total_time:.2f}s ({total_time/len(train_losses):.2f}s/epoch)')
    print(f'  >> Model size: {model_size:.2f}MB, Memory overhead: {memory_overhead:.2f}MB')

    plt.figure(figsize=(15,5))
    
    plt.subplot(1,3,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(val_aucs,  label='Val AUC', c='orange')
    plt.xlabel('Epoch'); plt.ylabel('AUC'); plt.title('AUC Curve')
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(val_rmses, label='Val RMSE', c='green')
    plt.plot(val_maes,  label='Val MAE', c='red')
    plt.xlabel('Epoch'); plt.ylabel('Error'); plt.title('RMSE & MAE Curve')
    plt.legend()

    plt.suptitle(f'{name} ({split_method_en}) Training & Validation Metrics Over Epochs')
    fig_path = os.path.join(SAVE_FIG_DIR, f'{name}_{split_method_en}_metrics_curve.png')
    plt.tight_layout(); plt.savefig(fig_path); plt.close()
    print(f'  >> Curves saved to {fig_path}')

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model

def predict_and_save(model, name, raw_test_df, u_map, i_map, split_method_en, mean_rating): 
    model.eval(); model.to(DEVICE)

    outfile = os.path.join(SAVE_RES_DIR, f'{name}_{split_method_en}_predictions.txt')
    with open(outfile, 'w', encoding='utf-8') as f:
        last_user, items = None, []
        for idx,row in raw_test_df.iterrows(): 
            u_raw, i_raw = row['user'], row['item']
            if last_user is None: last_user = u_raw
            if u_raw != last_user:
                f.write(f"{last_user}|{len(items)}\n")
                for itm,score in items:
                    f.write(f"{itm} {int(round(score))}\n")
                last_user, items = u_raw, []

            u_idx = u_map.get(u_raw, None)
            i_idx = i_map.get(i_raw, None)
            if u_idx is not None and i_idx is not None:
                with torch.no_grad():
                    if isinstance(model, SVDattrModel):
                        pred = model.forward_without_attrs(
                            torch.tensor([u_idx], device=DEVICE),
                            torch.tensor([i_idx], device=DEVICE)
                        ).item()
                    else:
                        pred = model(
                            torch.tensor([u_idx], device=DEVICE),
                            torch.tensor([i_idx], device=DEVICE)
                        ).item()
                if math.isnan(pred): pred = mean_rating
            else:
                pred = mean_rating
            items.append((i_raw, pred))

        f.write(f"{last_user}|{len(items)}\n")
        for itm,score in items:
            f.write(f"{itm} {int(round(score))}\n")

    print(f'  >> Predictions saved to: {outfile}\n')

def run_experiment_with_split_method(split_method, split_method_name):
    global train_loader, val_loader, train_df
    
    split_method_en = SPLIT_METHOD_MAP[split_method_name]  # 获取英文命名
    
    print(f'\n{"="*80}')
    print(f'Running experiment with split method: {split_method_name} ({split_method_en})')
    print(f'{"="*80}')
    
    if split_method == 1:
        print('Using global random split method (9:1 ratio)')
        data_list = dataframe_to_list(train_df)
        train_list, val_list = split_train_test(data_list, train_ratio=0.9)
        train_df_actual = list_to_dataframe(train_list)
        val_df = list_to_dataframe(val_list)
    elif split_method == 2:
        print('Using user-based split method')
        data_list = dataframe_to_list(train_df)
        train_list, val_list = split_train_test2(data_list)
        train_df_actual = list_to_dataframe(train_list)
        val_df = list_to_dataframe(val_list)

    print(f'Train set size: {len(train_df_actual)}, Validation set size: {len(val_df)}')

    train_set = _prepare_dataset_with_ratings(u_map, i_map, train_df_actual)
    val_set   = _prepare_dataset_with_ratings(u_map, i_map, val_df)

    num_workers_for_dataloader = 0
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=num_workers_for_dataloader, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH*2, shuffle=False, num_workers=num_workers_for_dataloader, pin_memory=True)

    models = {
        'ClassicalSVD': ClassicalSVD(len(u_map), len(i_map), LATENT),
        'FunkSVD'     : FunkSVD     (len(u_map), len(i_map), LATENT),
        'BiasSVD'     : BiasSVD     (len(u_map), len(i_map), LATENT),
        'SVDattr'     : SVDattrModel(len(u_map), len(i_map), n_attributes=100, k=LATENT)
    }

    for name, m in models.items():
        print(f'\n==== Training {name} ({split_method_en}) ====')
        trained = train_model(m, name, split_method_name, split_method_en)
        predict_and_save(trained, name, final_test_df, u_map, i_map, split_method_en, train_df['rating'].mean())

def print_performance_comparison():
    print(f'\n{"="*100}')
    print(' ' * 30 + 'PERFORMANCE COMPARISON OF RECOMMENDATION MODELS')
    print(f'{"="*100}')
    
    df_results = pd.DataFrame(performance_results)
    
    for split_method in df_results['split_method'].unique():
        print(f'\n{"[ " + split_method + " ]":^100}')
        print(f'{"-"*100}')
        subset = df_results[df_results['split_method'] == split_method].copy()
        
        print(f"{'Model':<15} {'RMSE':<8} {'AUC':<8} {'MAE':<8} {'Time Overhead':<15} {'Space Overhead':<15}")
        print(f'{"-"*80}')
        
        for _, row in subset.iterrows():
            model_name = row['model']
            rmse = f"{row['final_rmse']:.3f}"
            auc = f"{row['best_val_auc']:.3f}"
            mae = f"{row['final_mae']:.3f}"
            time_overhead = f"{row['time_per_epoch_s']:.2f}s/epoch"
            space_overhead = f"{row['model_size_mb']:.2f}MB"
            
            print(f"{model_name:<15} {rmse:<8} {auc:<8} {mae:<8} {time_overhead:<15} {space_overhead:<15}")
    
    results_file = os.path.join(SAVE_RES_DIR, 'performance_comparison.csv')
    df_results.to_csv(results_file, index=False, float_format='%.4f')
    print(f'\nPerformance comparison results saved to: {results_file}')
    print(f'{"="*100}\n')



if __name__ == '__main__':
    train_df = load_train_data() 
    final_test_df = load_test_data() 

    users = sorted(train_df['user'].unique())
    items = sorted(train_df['item'].unique())
    u_map = {u:i for i,u in enumerate(users)}
    i_map = {v:i for i,v in enumerate(items)}

    print("Starting comprehensive performance evaluation...")
    print(f"Dataset info: {len(users)} users, {len(items)} items, {len(train_df)} ratings")
    print(f"Device: {DEVICE}")

    run_experiment_with_split_method(1, "全局随机划分")
    run_experiment_with_split_method(2, "按用户划分")
    
    print_performance_comparison()

