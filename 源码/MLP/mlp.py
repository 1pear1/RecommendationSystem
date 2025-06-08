import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import os
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import sys
import io
import random
from collections import defaultdict


def read_data(file_path, is_train=True):
    data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            user_info = lines[i].strip().split('|')
            user_id = int(user_info[0])
            item_count = int(user_info[1])
            i += 1
            if is_train:
                items = [list(map(int, lines[i+j].strip().split())) for j in range(item_count)]
            else:
                items = [int(lines[i+j]) for j in range(item_count)]
            i += item_count
            data[user_id] = items
    return data

def process_train_data(data_dict):
    return [[user_id, item_id, score] for user_id, items in data_dict.items() for item_id, score in items]

def process_test_data(data_dict):
    return [[user_id, item_id] for user_id, items in data_dict.items() for item_id in items]


def split_train_test(raw_data, test_size=0.1, random_state=2):
    return train_test_split(raw_data, test_size=test_size, random_state=random_state)

def split_train_test2(raw_data, random_state=2):
    random.seed(random_state)
    user_records = defaultdict(list)
    for row in raw_data:
        user_records[row[0]].append(row)
    train_data, val_data = [], []
    for records in user_records.values():
        if len(records) == 1:
            train_data.extend(records)
        else:
            idx = random.randint(0, len(records)-1)
            val_data.append(records[idx])
            train_data.extend([r for i, r in enumerate(records) if i != idx])
    return train_data, val_data


class MLP(nn.Module):
    def __init__(self, n_user, d_user, n_item, d_item, dropout=0.4):
        super().__init__()
        self.embedding_user = nn.Embedding(n_user, d_user)
        self.embedding_item = nn.Embedding(n_item, d_item)
        self.fc_layers = nn.Sequential(
            nn.Linear(d_user + d_item, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, user_ids, item_ids):
        vec_user = self.embedding_user(user_ids)
        vec_item = self.embedding_item(item_ids)
        x = torch.cat((vec_user, vec_item), 1)
        return self.fc_layers(x) * 100


def run_experiment(train_data, val_data, n_user, n_item, d_user=128, d_item=64, batch_size=128, epoch=5, lr=0.01, model_dir='model', dropout=0.4, patience=5, verbose=True, mode='default'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLP(n_user, d_user, n_item, d_item, dropout=dropout).to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    best_rmse, counter = float('inf'), 0
    history = {k: [] for k in ['epoch','train_loss','train_rmse','train_mae','train_time','valid_loss','valid_rmse','valid_mae','train_mape','valid_mape']}
    os.makedirs('figures', exist_ok=True)
    for ep in range(epoch):
        if verbose: print(f'Epoch {ep+1}/{epoch}')
        ep_start = time.time()
        for phase, data_ in [('Train', train_data), ('Valid', val_data)]:
            model.train() if phase == 'Train' else model.eval()
            total_loss, total_mse, total_mae, all_labels, all_preds = 0, 0, 0, [], []
            n_batches = (len(data_) + batch_size - 1) // batch_size
            for i in range(0, len(data_), batch_size):
                batch = np.array(data_[i:i+batch_size])
                users = torch.tensor(batch[:,0], dtype=torch.long, device=device)
                items = torch.tensor(batch[:,1], dtype=torch.long, device=device)
                ratings = torch.tensor(batch[:,2], dtype=torch.float, device=device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='Train'):
                    outputs = model(users, items).squeeze()
                    loss = loss_func(outputs, ratings)
                    if phase=='Train':
                        loss.backward()
                        optimizer.step()
                preds = outputs.cpu().detach().numpy()
                labs = ratings.cpu().numpy()
                all_labels.extend(labs)
                all_preds.extend(preds)
                total_loss += loss.item()
                total_mse += mean_squared_error(labs, preds)
                total_mae += mean_absolute_error(labs, preds)
                if verbose:
                    percent = ((i//batch_size+1) / n_batches) * 100
                    sys.stdout.write(f'\r[{phase} Epoch {ep+1}/{epoch}] Batch {i//batch_size+1}/{n_batches} ({percent:.1f}%)')
                    sys.stdout.flush()
            if verbose:
                sys.stdout.write('\n')
            avg_loss = total_loss/(n_batches)
            avg_rmse = math.sqrt(total_mse/(n_batches))
            avg_mae = total_mae/(n_batches)
            mape = np.mean(np.abs((np.array(all_labels) - np.array(all_preds)) / (np.array(all_labels) + 1e-8))) * 100
            if verbose:
                print(f'{phase} | Loss:{avg_loss:.5f} RMSE: {avg_rmse:.3f} MAE: {avg_mae:.3f} MAPE: {mape:.2f}%')
            if phase=='Train':
                history['train_loss'].append(avg_loss)
                history['train_rmse'].append(avg_rmse)
                history['train_mae'].append(avg_mae)
                history['train_mape'].append(mape)
            else:
                history['valid_loss'].append(avg_loss)
                history['valid_rmse'].append(avg_rmse)
                history['valid_mae'].append(avg_mae)
                history['valid_mape'].append(mape)
                scheduler.step(avg_loss)
                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    counter = 0
                    torch.save(model.state_dict(), f"{model_dir}/ckpt.model")
                    if verbose:
                        print(f'Saving model with RMSE {avg_rmse:.3f}')
                else:
                    counter += 1
                if counter >= patience:
                    if verbose:
                        print('Early stopping!')
                    break
        ep_time = time.time() - ep_start
        history['epoch'].append(ep+1)
        history['train_time'].append(ep_time)
        if verbose:
            print(f'Epoch {ep+1} time: {ep_time:.2f}s')
        if counter >= patience:
            break
    
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters()) / 1024 / 1024
    if verbose:
        print(f"Model size: {param_size:.2f} MB")
    
    x = history['epoch']
    for metric in ['loss','rmse','mae','mape']:
        if f'train_{metric}' in history:
            plt.figure()
            plt.plot(x, history.get(f'train_{metric}', []), label=f'Train {metric.upper()}')
            plt.plot(x, history.get(f'valid_{metric}', []), label=f'Valid {metric.upper()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.upper())
            plt.legend()
            plt.savefig(f'figures/{metric}_curve_{mode}.png')
            plt.close()
    
    with open(f'mlp_performance_comparison_{mode}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_rmse', 'train_mae', 'train_mape', 'valid_loss', 'valid_rmse', 'valid_mae', 'valid_mape', 'train_time', 'model_size_MB'])
        for i in range(len(history['epoch'])):
            writer.writerow([
                history['epoch'][i],
                history['train_loss'][i],
                history['train_rmse'][i],
                history['train_mae'][i],
                history['train_mape'][i],
                history['valid_loss'][i],
                history['valid_rmse'][i],
                history['valid_mae'][i],
                history['valid_mape'][i],
                history['train_time'][i],
                param_size
            ])
    return model, history


def predict_test(model, d_user, d_item, max_user, max_item, test_file='data/test.txt', out_file='result.txt'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    results, users, items = [], [], []
    with open(test_file, 'r') as f:
        for line in f:
            if '|' in line:
                if users:
                    users_t = torch.tensor(users, dtype=torch.long).to(device)
                    items_t = torch.tensor(items, dtype=torch.long).to(device)
                    ratings = model(users_t, items_t).squeeze()
                    results.extend(f'{items[i]}  {ratings[i].item()}\n' for i in range(len(ratings)))
                results.append(line)
                user_id = int(line.split('|')[0])
                users, items = [], []
            else:
                item_id = int(line.strip())
                users.append(user_id)
                items.append(item_id)
    if users:
        users_t = torch.tensor(users, dtype=torch.long).to(device)
        items_t = torch.tensor(items, dtype=torch.long).to(device)
        ratings = model(users_t, items_t).squeeze()
        results.extend(f'{items[i]}  {ratings[i].item()}\n' for i in range(len(ratings)))
    with open(out_file, 'w') as f:
        f.writelines(results)


def main():
    args = sys.argv
    file_path = 'data/train.txt'
    data_dict = read_data(file_path, is_train=True)
    raw_data = process_train_data(data_dict)
    batch_size, epoch, lr = 128, 5, 0.001
    d_user, d_item = 100, 100
    max_user, max_item, _ = np.max(raw_data, 0)
    n_user, n_item = int(max_user)+1, int(max_item)+1

    def redirect_stdout(logfile):
        orig_stdout = sys.stdout
        f = open(logfile, 'w', encoding='utf-8')
        sys.stdout = f
        return orig_stdout, f

    if len(args) > 1 and args[1] == 'test':
        # python mlp.py test split1/split2
        if len(args) > 2 and args[2] in ['split1', 'split2']:
            model_dir = f'model/{args[2]}'
            result_file = f'result_{args[2]}.txt'
            log_file = f'log_test_{args[2]}.txt'
            orig_stdout, f = redirect_stdout(log_file)
            model = MLP(n_user, d_user, n_item, d_item).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            model.load_state_dict(torch.load(f'{model_dir}/ckpt.model'))
            predict_test(model, d_user, d_item, max_user, max_item, out_file=result_file)
            sys.stdout = orig_stdout
            f.close()
        else:
            print("Usage: python mlp.py test split1|split2")
    elif len(args) > 1 and args[1] == 'split1':
        print("=== 全局随机划分（split_train_test） ===")
        model_dir = 'model/split1'
        result_file = 'result_split1.txt'
        log_file = 'log_split1.txt'
        os.makedirs(model_dir, exist_ok=True)
        orig_stdout, f = redirect_stdout(log_file)
        train_data, val_data = split_train_test(raw_data)
        model, _ = run_experiment(train_data, val_data, n_user, n_item, d_user, d_item, batch_size, epoch, lr, model_dir, mode='split1')
        predict_test(model, d_user, d_item, max_user, max_item, out_file=result_file)
        sys.stdout = orig_stdout
        f.close()
    elif len(args) > 1 and args[1] == 'split2':
        print("=== 按用户划分（split_train_test2） ===")
        model_dir = 'model/split2'
        result_file = 'result_split2.txt'
        log_file = 'log_split2.txt'
        os.makedirs(model_dir, exist_ok=True)
        orig_stdout, f = redirect_stdout(log_file)
        train_data, val_data = split_train_test2(raw_data)
        model, _ = run_experiment(train_data, val_data, n_user, n_item, d_user, d_item, batch_size, epoch, lr, model_dir, mode='split2')
        predict_test(model, d_user, d_item, max_user, max_item, out_file=result_file)
        sys.stdout = orig_stdout
        f.close()
    elif len(args) > 1 and args[1] == 'search':
        orig_stdout, f = redirect_stdout('log_search.txt')
        search_best_params()
        sys.stdout = orig_stdout
        f.close()
    else:
        print("请使用如下命令之一：\npython mlp.py split1\npython mlp.py split2\npython mlp.py search\npython mlp.py test split1\npython mlp.py test split2")


def search_best_params():
    print("=== 参数搜索开始 ===")
    sys.stdout.flush()
    lr_list = [0.01, 0.001, 0.0005]
    epoch_list = [5]
    dropout_list = [0.4]
    d_user_list = [64, 100, 128]
    d_item_list = [64, 100, 128]
    best_rmse = float('inf')
    best_params = None
    results = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    file_path = 'data/train.txt'
    data_dict = read_data(file_path, is_train=True)
    raw_data = process_train_data(data_dict)
    max_user, max_item, _ = np.max(raw_data, 0)
    n_user, n_item = int(max_user)+1, int(max_item)+1
    train_data, val_data = split_train_test(raw_data)
    idx = 0
    for lr in lr_list:
        for epoch in epoch_list:
            for dropout in dropout_list:
                for d_user in d_user_list:
                    for d_item in d_item_list:
                        try:
                            print(f"Trying idx={idx} | lr={lr}, epoch={epoch}, dropout={dropout}, d_user={d_user}, d_item={d_item}")
                            sys.stdout.flush()
                            model = MLP(n_user, d_user, n_item, d_item, dropout=dropout).to(device)
                            _, history = run_experiment(train_data, val_data, n_user, n_item, d_user, d_item, 128, epoch, lr, 'model', dropout, 5, False, mode=f'search_{idx}')
                            min_val_rmse = min(history['valid_rmse'])
                            print(f"val RMSE: {min_val_rmse:.4f}")
                            sys.stdout.flush()
                            results.append({
                                'idx': idx,
                                'lr': lr, 'epoch': epoch, 'dropout': dropout,
                                'd_user': d_user, 'd_item': d_item,
                                'val_rmse': min_val_rmse
                            })
                            if min_val_rmse < best_rmse:
                                best_rmse = min_val_rmse
                                best_params = (lr, epoch, dropout, d_user, d_item)
                            idx += 1
                        except Exception as e:
                            print(f"Error with params idx={idx}: {e}")
                            sys.stdout.flush()
                            idx += 1
    print(f"Best params: lr={best_params[0]}, epoch={best_params[1]}, dropout={best_params[2]}, d_user={best_params[3]}, d_item={best_params[4]}, val RMSE={best_rmse:.4f}")
    sys.stdout.flush()
    # 画RMSE对比图
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(10,5))
    rmse_list = [r['val_rmse'] for r in results]
    labels = [f"{r['idx']}" for r in results]
    plt.plot(labels, rmse_list, marker='o')
    plt.xlabel('Param Combo Index')
    plt.ylabel('Validation RMSE')
    plt.title('Validation RMSE for Each Param Combo')
    plt.grid(True)
    plt.savefig('figures/param_search_rmse.png')
    plt.close()
    # 用最佳参数再训练一次
    print(f"Retraining with best params: lr={best_params[0]}, epoch={best_params[1]}, dropout={best_params[2]}, d_user={best_params[3]}, d_item={best_params[4]}")
    sys.stdout.flush()
    model, _ = run_experiment(train_data, val_data, n_user, n_item, best_params[3], best_params[4], 128, best_params[1], best_params[0], 'model', best_params[2], 5, True, mode='search_best')
    print("Best model and curves saved in model/ and figures/")
    sys.stdout.flush()

if __name__ == '__main__':
    main()