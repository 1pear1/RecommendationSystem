import os
from collections import defaultdict, Counter
import time
import numpy as np
import psutil
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# 文件名
TRAIN_FILE = 'train.txt'
TEST_FILE = 'test.txt'
RESULT_FILE = 'ResultForm.txt'

def parse_train(filename):
    user_count = 0
    item_set = set()
    rating_count = 0
    rating_values = []
    user_rating_count = []
    item_rating_counter = Counter()
    with open(filename, 'r', encoding='utf-8') as f:
        while True:
            head = f.readline()
            if not head:
                break
            user_count += 1
            user_id, num = head.strip().split('|')
            num = int(num)
            user_rating_count.append(num)
            for _ in range(num):
                line = f.readline()
                if not line:
                    break
                item_id, score = line.strip().split()
                item_set.add(item_id)
                rating_count += 1
                rating_values.append(int(score))
                item_rating_counter[item_id] += 1
    return {
        'user_count': user_count,
        'item_count': len(item_set),
        'rating_count': rating_count,
        'rating_values': rating_values,
        'user_rating_count': user_rating_count,
        'item_rating_counter': item_rating_counter,
    }

def parse_test(filename):
    user_count = 0
    item_set = set()
    rating_count = 0
    user_rating_count = []
    item_rating_counter = Counter()
    with open(filename, 'r', encoding='utf-8') as f:
        while True:
            head = f.readline()
            if not head:
                break
            user_count += 1
            user_id, num = head.strip().split('|')
            num = int(num)
            user_rating_count.append(num)
            for _ in range(num):
                line = f.readline()
                if not line:
                    break
                item_id = line.strip()
                item_set.add(item_id)
                rating_count += 1
                item_rating_counter[item_id] += 1
    return {
        'user_count': user_count,
        'item_count': len(item_set),
        'rating_count': rating_count,
        'user_rating_count': user_rating_count,
        'item_rating_counter': item_rating_counter,
    }

def stat_list(lst):
    arr = np.array(lst)
    return {
        'min': int(arr.min()),
        'max': int(arr.max()),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'median': float(np.median(arr)),
    }

# 读取训练集，返回user_items, item_users, rating三元组列表
def read_train_full(filename):
    user_items = defaultdict(dict)
    item_users = defaultdict(dict)
    ratings = []  # (user, item, rating)
    with open(filename, 'r', encoding='utf-8') as f:
        while True:
            head = f.readline()
            if not head:
                break
            user_id, num = head.strip().split('|')
            num = int(num)
            for _ in range(num):
                line = f.readline()
                if not line:
                    break
                item_id, score = line.strip().split()
                score = float(score)
                user_items[user_id][item_id] = score
                item_users[item_id][user_id] = score
                ratings.append((user_id, item_id, score))
    return user_items, item_users, ratings

def read_test(filename):
    test_data = []
    with open(filename, 'r', encoding='utf-8') as f:
        while True:
            head = f.readline()
            if not head:
                break
            user_id, num = head.strip().split('|')
            num = int(num)
            items = []
            for _ in range(num):
                line = f.readline()
                if not line:
                    break
                item_id = line.strip()
                items.append(item_id)
            test_data.append((user_id, items))
    return test_data

def cosine_similarity(item_users, item1, item2):
    users1 = item_users[item1]
    users2 = item_users[item2]
    common_users = set(users1.keys()) & set(users2.keys())
    if not common_users:
        return 0.0
    v1 = np.array([users1[u] for u in common_users])
    v2 = np.array([users2[u] for u in common_users])
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def predict(user_id, item_id, user_items, item_users, k=20, sim_threshold=0.5):
    # 冷启动用户
    if user_id not in user_items:
        if item_id in item_users:
            return np.mean(list(item_users[item_id].values()))
        else:
            return 70.0
    # 冷启动物品
    if item_id not in item_users:
        return np.mean(list(user_items[user_id].values()))
    # 只用与目标物品有共同评分用户的物品计算相似度
    sims = []
    for neighbor in user_items[user_id]:
        if neighbor == item_id:
            continue
        # 只考虑有共同评分用户的物品
        if len(set(item_users[item_id].keys()) & set(item_users[neighbor].keys())) == 0:
            continue
        sim = cosine_similarity(item_users, item_id, neighbor)
        if sim > sim_threshold:
            sims.append((neighbor, sim))
    # 如果有高相似度邻居，使用它们
    if sims:
        sims = sorted(sims, key=lambda x: x[1], reverse=True)[:k]
        numerator, denominator = 0.0, 0.0
        for neighbor, sim in sims:
            numerator += sim * user_items[user_id][neighbor]
            denominator += abs(sim)
        if denominator > 0:
            return numerator / denominator
    # 否则，遍历所有物品，找与目标物品有共同评分用户的前K个最相似物品
    sim_all = []
    for neighbor in item_users:
        if neighbor == item_id:
            continue
        if user_id not in user_items or neighbor not in user_items[user_id]:
            continue
        if len(set(item_users[item_id].keys()) & set(item_users[neighbor].keys())) == 0:
            continue
        sim = cosine_similarity(item_users, item_id, neighbor)
        sim_all.append((neighbor, sim))
    sim_all = sorted(sim_all, key=lambda x: x[1], reverse=True)[:k]
    numerator, denominator = 0.0, 0.0
    for neighbor, sim in sim_all:
        numerator += sim * user_items[user_id][neighbor]
        denominator += abs(sim)
    if denominator > 0:
        return numerator / denominator
    # 最后兜底：用用户均值
    return np.mean(list(user_items[user_id].values()))

def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mean_true = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_true) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0

def plot_metrics(y_true, y_pred):
    # RMSE/MAE分布图
    errors = np.abs(np.array(y_true) - np.array(y_pred))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(errors, bins=30, color='skyblue', edgecolor='black')
    plt.title('Absolute Error Distribution')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    # 预测-真实散点图
    plt.subplot(1,2,2)
    plt.scatter(y_true, y_pred, alpha=0.3, s=5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.title('Prediction vs. Ground Truth')
    plt.xlabel('True Rating')
    plt.ylabel('Predicted Rating')
    plt.tight_layout()
    plt.savefig('val_metrics.png')
    plt.close()

def main():
    total_start = time.time()
    process = psutil.Process(os.getpid())

    print('读取训练集...')
    user_items, item_users, ratings = read_train_full(TRAIN_FILE)
    print(f'总评分数: {len(ratings)}')

    # 划分90%训练，10%验证
    random.seed(42)
    random.shuffle(ratings)
    split_idx = int(len(ratings) * 0.9)
    train_ratings = ratings[:split_idx]
    val_ratings = ratings[split_idx:]

    # 构建新训练集user_items, item_users
    train_user_items = defaultdict(dict)
    train_item_users = defaultdict(dict)
    train_start = time.time()
    for u, i, r in train_ratings:
        train_user_items[u][i] = r
        train_item_users[i][u] = r
    train_end = time.time()
    print(f'训练集构建耗时: {train_end - train_start:.2f} 秒')

    # 验证集评估RMSE/MAE/R2
    print('在验证集上评估RMSE/MAE/R2...')
    y_true, y_pred = [], []
    pred_start = time.time()
    for u, i, r in tqdm(val_ratings, desc="验证集预测进度"):
        pred = predict(u, i, train_user_items, train_item_users, k=20, sim_threshold=0.5)
        pred = max(10, min(100, pred))
        y_true.append(r)
        y_pred.append(pred)
    pred_end = time.time()
    val_rmse = rmse(y_true, y_pred)
    val_mae = mae(y_true, y_pred)
    val_r2 = r2_score(y_true, y_pred)
    print(f'验证集RMSE: {val_rmse:.4f}')
    print(f'验证集MAE: {val_mae:.4f}')
    print(f'验证集R2: {val_r2:.4f}')
    print(f'预测耗时: {pred_end - pred_start:.2f} 秒')
    plot_metrics(y_true, y_pred)
    print('已保存val_metrics.png')

    # 预测测试集并写入结果
    print('读取测试集...')
    test_data = read_test(TEST_FILE)
    print('预测评分并写入结果...')
    test_pred_start = time.time()
    with open(RESULT_FILE, 'w', encoding='utf-8') as fout:
        for user_id, items in test_data:
            fout.write(f'{user_id}|{len(items)}\n')
            for item_id in items:
                pred = predict(user_id, item_id, train_user_items, train_item_users, k=20, sim_threshold=0.5)
                pred = int(round(pred))
                pred = max(10, min(100, pred))
                fout.write(f'{item_id}  {pred}\n')
    test_pred_end = time.time()
    print(f'测试集预测耗时: {test_pred_end - test_pred_start:.2f} 秒')

    total_end = time.time()
    mem = process.memory_info().rss / 1024 / 1024  # MB
    print(f'总耗时: {total_end - total_start:.2f} 秒')
    print(f'内存消耗: {mem:.2f} MB')

if __name__ == '__main__':
    main() 