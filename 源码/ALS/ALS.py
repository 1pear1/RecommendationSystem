import numpy as np
# from datapreprocess import *
import gc
import matplotlib.pyplot as plt
from matplotlib import font_manager
import time
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def initialize_matrices(num_users, num_items, num_factors):
    """初始化用户和物品矩阵"""
    return (
        np.random.normal(0, 0.1, size=(num_users, num_factors)),
        np.random.normal(0, 0.1, size=(num_items, num_factors))
    )


def create_mapping_dicts(users, items):
    """创建用户和物品的索引映射字典"""
    return (
        {user: idx for idx, user in enumerate(users)},
        {item: idx for idx, item in enumerate(items)}
    )


def build_rating_matrix(train_data, user_to_index, item_to_index):
    """构建评分矩阵"""
    num_users = len(user_to_index)
    num_items = len(item_to_index)
    rating_matrix = np.zeros((num_users, num_items), dtype=np.float32)

    for user, user_items in train_data.items():
        for item, rating in user_items:
            rating_matrix[user_to_index[user], item_to_index[item]] = rating

    return rating_matrix


def update_factors(matrix, other_matrix, rating_matrix, regularization, num_factors, is_user=True):
    """更新用户或物品因子矩阵"""
    num_entities = matrix.shape[0]
    for i in range(num_entities):
        if is_user:
            ratings = rating_matrix[i, :]
            non_zero = ratings.nonzero()[0]
            if len(non_zero) > 0:
                submatrix = other_matrix[non_zero, :]
                ratings_sub = ratings[non_zero]
        else:
            ratings = rating_matrix[:, i]
            non_zero = ratings.nonzero()[0]
            if len(non_zero) > 0:
                submatrix = other_matrix[non_zero, :]
                ratings_sub = ratings[non_zero]

        if len(non_zero) > 0:
            A = submatrix.T @ submatrix + regularization * np.eye(num_factors)
            b = submatrix.T @ ratings_sub
            matrix[i, :] = np.linalg.solve(A, b)
    return matrix


def rmse(rating_matrix, prediction_matrix):
    mask = rating_matrix > 0
    mse = np.sum((rating_matrix[mask] - prediction_matrix[mask]) ** 2) / np.sum(mask)
    rmse = np.sqrt(mse)
    return rmse


def evaluate_model(user_matrix, item_matrix, rating_matrix):
    """评估模型性能"""
    prediction = user_matrix @ item_matrix.T
    return rmse(rating_matrix, prediction)


def als(train_data, num_factors=10, num_iterations=10, regularization=0.1):
    """交替最小二乘法实现推荐系统"""
    # 数据准备
    users = sorted(train_data.keys())
    items = sorted({item for user_items in train_data.values() for item, _ in user_items})

    # 创建映射
    user_to_index, item_to_index = create_mapping_dicts(users, items)

    # 构建评分矩阵
    rating_matrix = build_rating_matrix(train_data, user_to_index, item_to_index)

    # 初始化
    user_matrix, item_matrix = initialize_matrices(len(users), len(items), num_factors)

    # 训练过程
    history = []
    best_rmse = float('inf')
    best_matrices = None

    # 记录开始时间
    start_time = time.time()

    for epoch in range(num_iterations):
        epoch_start = time.time()

        # 更新因子
        user_matrix = update_factors(user_matrix, item_matrix, rating_matrix, regularization, num_factors, True)
        item_matrix = update_factors(item_matrix, user_matrix, rating_matrix, regularization, num_factors, False)

        # 评估
        current_rmse = evaluate_model(user_matrix, item_matrix, rating_matrix)
        history.append(current_rmse)

        # 保存最佳模型
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_matrices = (user_matrix.copy(), item_matrix.copy())

        # 计算每个epoch的时间
        epoch_time = time.time() - epoch_start

        # 输出进度
        print(f"训练进度: {epoch + 1}/{num_iterations} | 当前RMSE: {current_rmse:.4f} | Epoch Time: {epoch_time:.2f}s")

        # 清理内存
        gc.collect()

    # 计算总训练时间
    total_time = time.time() - start_time

    # 使用最佳模型
    if best_matrices is not None:
        user_matrix, item_matrix = best_matrices

    # 计算模型大小（MB）
    # 计算矩阵大小
    matrix_size = (user_matrix.nbytes + item_matrix.nbytes) / (1024 * 1024)

    # 计算评分矩阵大小
    rating_matrix_size = rating_matrix.data.nbytes / (1024 * 1024)

    # 计算映射字典大小
    user_dict_size = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in user_to_index.items()) / (1024 * 1024)
    item_dict_size = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in item_to_index.items()) / (1024 * 1024)

    # 总模型大小
    total_model_size = matrix_size + rating_matrix_size + user_dict_size + item_dict_size

    # 输出最终结果
    final_rmse = evaluate_model(user_matrix, item_matrix, rating_matrix)
    print("\n训练完成!")
    print(f"最终RMSE: {final_rmse:.4f}")
    print(f"最佳RMSE: {best_rmse:.4f}")
    print(f"总训练时间: {total_time:.2f}秒")
    print(f"模型大小详情:")
    print(f"- 用户-物品矩阵: {matrix_size:.2f}MB")
    print(f"- 评分矩阵: {rating_matrix_size:.2f}MB")
    print(f"- 用户映射字典: {user_dict_size:.2f}MB")
    print(f"- 物品映射字典: {item_dict_size:.2f}MB")
    print(f"总模型大小: {total_model_size:.2f}MB")

    # 绘制RMSE变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_iterations + 1), history, 'b-', label='RMSE')
    plt.title('训练过程中RMSE的变化', fontsize=12)
    plt.xlabel('迭代次数', fontsize=10)
    plt.ylabel('RMSE', fontsize=10)
    plt.grid(True)
    plt.legend(fontsize=10)

    # 保存图片
    plt.savefig('result/rmse_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    return user_matrix, item_matrix, user_to_index, item_to_index


def predict(user_id, item_id, user_matrix, item_matrix, user_to_index, item_to_index):
    user_idx = user_to_index.get(user_id)
    item_idx = item_to_index.get(item_id)
    if user_idx is not None and item_idx is not None:
        return user_matrix[user_idx, :].dot(item_matrix[item_idx, :])
    else:
        return None


if __name__ == "__main__":
    # 加载训练数据
    train_dict = {}
    with open('data/train.txt', 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            user_info = lines[i].strip().split('|')
            user_id = int(user_info[0])
            item_count = int(user_info[1])
            i += 1

            items = []
            for _ in range(item_count):
                item_info = lines[i].strip().split()
                item_id = int(item_info[0])
                score = int(item_info[1])
                items.append([item_id, score])
                i += 1

            train_dict[user_id] = items
    print(f"训练数据加载完成，共 {len(train_dict)} 个用户")

    # 训练模型
    print("\n开始训练模型...")
    user_matrix, item_matrix, user_to_index, item_to_index = als(
        train_dict,
        num_factors=10,
        num_iterations=20,
        regularization=0.1
    )

    # 加载测试数据
    test_dict = {}
    with open('data/test.txt', 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            user_info = lines[i].strip().split('|')
            user_id = int(user_info[0])
            item_count = int(user_info[1])
            i += 1

            items = []
            for _ in range(item_count):
                item_id = int(lines[i])
                items.append(item_id)
                i += 1

            test_dict[user_id] = items
    print(f"测试数据加载完成，共 {len(test_dict)} 个用户")

    # 生成预测结果
    predictions = {}
    total_predictions = 0

    for user_id, item_ids in test_dict.items():
        user_predictions = []
        for item_id in item_ids:
            pred_rating = predict(user_id, item_id, user_matrix, item_matrix, user_to_index, item_to_index)
            user_predictions.append([item_id, pred_rating])
        predictions[user_id] = user_predictions
        total_predictions += len(user_predictions)

    # 保存预测结果
    output_file = 'result/ALS_result.txt'
    with open(output_file, 'w') as f:
        for user_id, items in predictions.items():
            f.write(f"{user_id}|{len(items)}\n")
            for item_id, rating in items:
                f.write(f"{item_id} {rating}\n")
    print(f"\n已保存预测结果到 {output_file}...")


