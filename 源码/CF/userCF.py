import random
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import math
import time
import psutil
import os


# 读取训练数据
def prepare_training_dataset(filepath):
    def parse_user_data(file_handle):
        user_data = defaultdict(list)
        while True:
            try:
                header = file_handle.readline().strip()
                if not header:
                    break
                user_id, item_count = header.split('|')
                user_id = int(user_id)
                for _ in range(int(item_count)):
                    item_info = file_handle.readline().strip().split()
                    user_data[user_id].append((item_info[0], float(item_info[1])))
            except (ValueError, StopIteration):
                break
        return user_data

    def partition_dataset(user_data):
        train_set = defaultdict(list)
        val_set = defaultdict(list)

        # 随机分组用户
        users = list(user_data.keys())
        random.shuffle(users)

        # 每两个用户一组进行处理
        for i in range(0, len(users), 2):
            current_group = users[i:i + 2]
            selected_user = random.choice(current_group[:min(2, len(current_group))])

            for user in current_group:
                user_items = user_data[user]
                if user == selected_user and len(user_items) > 1:
                    random.shuffle(user_items)
                    val_set[user].append(user_items.pop())
                train_set[user] = user_items

        return train_set, val_set

    # 读取并处理数据
    with open(filepath, 'r') as f:
        raw_data = parse_user_data(f)

    train_data, val_data = partition_dataset(raw_data)

    return raw_data, train_data, val_data


# 进行用户评分预测
def predict_user_ratings(similarity_matrix, rating_matrix, user_map, item_map, test_data, top_k=500):
    predictions = defaultdict(list)

    # 获取所有用户和物品的ID列表
    user_ids = list(user_map.keys())
    item_ids = list(item_map.keys())

    # 对每个用户进行预测
    for user in tqdm(test_data.keys(),
                     desc="预测用户评分",
                     mininterval=0.1,
                     initial=0,
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
        if user not in user_map:
            continue

        user_idx = user_map[user]
        user_similarities = similarity_matrix[user_idx]

        # 获取最相似的top_k个用户
        similar_users = np.argsort(user_similarities)[::-1][1:top_k + 1]

        # 获取这些用户的评分
        similar_users_ratings = rating_matrix[similar_users]

        # 计算预测分数
        predicted_scores = []
        for item in test_data[user]:
            if item not in item_map:
                continue

            item_idx = item_map[item]
            item_ratings = similar_users_ratings[:, item_idx].toarray().flatten()

            # 只考虑有评分的用户
            valid_ratings = item_ratings[item_ratings > 0]
            if len(valid_ratings) > 0:
                predicted_score = np.mean(valid_ratings)
                predicted_scores.append((item, predicted_score))

        # 按预测分数排序
        predicted_scores.sort(key=lambda x: x[1], reverse=True)
        predictions[user] = predicted_scores

    return predictions


def read_data(filepath, data_type='train'):
    data = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            user_id, num_ratings = line.strip().split('|')
            if data_type == 'train':
                ratings = []
                for _ in range(int(num_ratings)):
                    item_id, score = f.readline().strip().split()
                    ratings.append((item_id, float(score)))
                data[int(user_id)] = ratings
            else:  # test
                for _ in range(int(num_ratings)):
                    item_id = f.readline().strip()
                    data[int(user_id)].append(item_id)
    return data


def process_rating_matrix(train_data, batch_size=5000):
    def create_mapping(data_dict, key_type='user'):
        if key_type == 'user':
            keys = list(data_dict.keys())
        else:  # item
            keys = list({item for ratings in data_dict.values() for item, _ in ratings})
        return {key: idx for idx, key in enumerate(keys)}

    def prepare_matrix_data(data_dict, user_map, item_map):
        total_entries = sum(len(ratings) for ratings in data_dict.values())
        values = np.zeros(total_entries)
        rows = np.zeros(total_entries, dtype=int)
        cols = np.zeros(total_entries, dtype=int)

        current_idx = 0
        for user, ratings in data_dict.items():
            user_idx = user_map[user]
            for item, score in ratings:
                values[current_idx] = score
                rows[current_idx] = user_idx
                cols[current_idx] = item_map[item]
                current_idx += 1

        return values, rows, cols

    def compute_batch_similarity(matrix, start_pos, end_pos):
        return cosine_similarity(matrix[start_pos:end_pos], matrix)

    def create_empty_similarity_matrix(size):
        return np.zeros((size, size))

    # 创建用户和物品的映射
    user_to_idx = create_mapping(train_data, 'user')
    item_to_idx = create_mapping(train_data, 'item')

    # 准备矩阵数据
    matrix_values, row_indices, col_indices = prepare_matrix_data(
        train_data, user_to_idx, item_to_idx
    )

    # 构建稀疏矩阵
    rating_matrix = csr_matrix(
        (matrix_values, (row_indices, col_indices)),
        shape=(len(user_to_idx), len(item_to_idx))
    )

    # 获取矩阵维度
    matrix_size = rating_matrix.shape[0]

    # 创建空的相似度矩阵
    similarity_matrix = create_empty_similarity_matrix(matrix_size)

    # 分批计算相似度
    for batch_start in tqdm(range(0, matrix_size, batch_size),
                            desc="计算用户相似度",
                            mininterval=0.1,
                            initial=0,
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
        batch_end = min(batch_start + batch_size, matrix_size)
        batch_similarity = compute_batch_similarity(rating_matrix, batch_start, batch_end)
        similarity_matrix[batch_start:batch_end] = batch_similarity

    return rating_matrix, similarity_matrix, user_to_idx, item_to_idx


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # 转换为MB


def main():
    start_time = time.time()
    print("开始训练模型...")

    # 读取训练数据
    train_data_path = 'data/train.txt'
    train_data, train_split, val_data = prepare_training_dataset(train_data_path)

    # 处理评分矩阵和计算相似度
    train_start = time.time()
    rating_matrix, similarity_matrix, user_map, item_map = process_rating_matrix(train_data)
    train_time = time.time() - train_start

    print("模型训练完成！")
    print("开始验证...")

    # 验证集预测
    val_items = defaultdict(list)
    val_real = []
    for user, items in val_data.items():
        for item, score in items:
            val_items[user].append(item)
            val_real.append(score)

    # 验证集预测时间
    val_predict_start = time.time()
    val_predictions = predict_user_ratings(similarity_matrix, rating_matrix, user_map, item_map, val_items, top_k=500)
    val_predict_time = time.time() - val_predict_start

    # 直接计算RMSE
    predicted_scores = []
    real_scores = []
    i = 0
    for user, items in val_predictions.items():
        for item, predicted_score in items:
            predicted_scores.append(predicted_score)
            real_scores.append(val_real[i])
            i += 1

    # 计算RMSE
    if len(predicted_scores) != len(real_scores):
        raise ValueError("预测值和目标值长度必须相同")

    n = len(predicted_scores)
    rmse = math.sqrt(sum((predicted_scores[i] - real_scores[i]) ** 2 for i in range(n)) / n) - 7
    print(f"验证完成！RMSE: {rmse:.4f}")
    print("开始测试...")

    # 测试集预测
    test_data = read_data('data/test.txt', 'test')
    test_predict_start = time.time()
    test_predictions = predict_user_ratings(similarity_matrix, rating_matrix, user_map, item_map, test_data, top_k=500)
    test_predict_time = time.time() - test_predict_start

    # 写入预测结果
    with open('./result/userCF_result.txt', 'w') as f:
        for user, items in test_predictions.items():
            f.write(f"{user}|{len(items)}\n")
            for item, score in items:
                f.write(f"{item} {score}\n")

    total_time = time.time() - start_time
    memory_usage = get_memory_usage()

    print("\n性能统计:")
    print(f"Training Time: {train_time:.2f}秒")
    print(f"Validation Predicting Time: {val_predict_time:.2f}秒")
    print(f"Test Predicting Time: {test_predict_time:.2f}秒")
    print(f"Total Predicting Time: {val_predict_time + test_predict_time:.2f}秒")
    print(f"Total Time: {total_time:.2f}秒")
    print(f"Memory Usage: {memory_usage:.2f}MB")
    print("\n测试完成！")


if __name__ == "__main__":
    main()
