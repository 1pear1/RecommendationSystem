import os
from collections import defaultdict, Counter

# 文件名
TRAIN_FILE = 'train.txt'
TEST_FILE = 'test.txt'

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
    import numpy as np
    arr = np.array(lst)
    return {
        'min': int(arr.min()),
        'max': int(arr.max()),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'median': float(np.median(arr)),
    }

def main():
    print('==== 训练集统计 ===')
    train_stat = parse_train(TRAIN_FILE)
    print(f"用户数: {train_stat['user_count']}")
    print(f"物品数: {train_stat['item_count']}")
    print(f"评分数: {train_stat['rating_count']}")
    print(f"评分分布: {stat_list(train_stat['rating_values'])}")
    print(f"每个用户评分数分布: {stat_list(train_stat['user_rating_count'])}")
    print(f"每个物品被评分数分布: {stat_list(list(train_stat['item_rating_counter'].values()))}")
    print('==== 测试集统计 ===')
    test_stat = parse_test(TEST_FILE)
    print(f"用户数: {test_stat['user_count']}")
    print(f"物品数: {test_stat['item_count']}")
    print(f"交互数: {test_stat['rating_count']}")
    print(f"每个用户交互数分布: {stat_list(test_stat['user_rating_count'])}")
    print(f"每个物品被交互数分布: {stat_list(list(test_stat['item_rating_counter'].values()))}")
    # 统计训练集和测试集的用户、物品交集
    print('==== 训练集与测试集交集 ===')
    train_items = set(train_stat['item_rating_counter'].keys())
    test_items = set(test_stat['item_rating_counter'].keys())
    print(f"训练集和测试集共有物品数: {len(train_items & test_items)}")
    print(f"只在测试集出现的物品数: {len(test_items - train_items)}")
    train_users = train_stat['user_count']
    test_users = test_stat['user_count']
    print(f"训练集用户数: {train_users}, 测试集用户数: {test_users}")

if __name__ == '__main__':
    main() 