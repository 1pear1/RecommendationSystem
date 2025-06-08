# 推荐系统作业

本项目实现了多种推荐算法，包括传统SVD系列模型和现代深度学习模型。

## 项目结构

```
recommend/
├── data/               # 数据目录
├── models/             # 模型实现
│   ├── funk_svd.py    # FunkSVD模型
│   ├── classical_svd.py # ClassicalSVD模型 
│   ├── bias_svd.py    # BiasSVD模型
│   ├── svd_attr.py    # SVDattr模型
│   ├── neural_cf.py   # NeuralCF模型
│   └── autoint.py     # AutoInt模型
├── utils/              # 工具函数
│   ├── data_loader.py  # 数据加载
│   ├── config.py       # 配置管理
│   └── feature_engineer.py # 特征工程
├── results/            # 结果输出
├── figures/            # 图表保存
├── saved_models/       # 保存的模型
├── SVD_all_models.py   # SVD系列模型主程序
├── neuralcf_model.py   # NeuralCF训练程序
├── autoint_model.py    # AutoInt训练程序
└── README.md
```

## 实现的模型

### 传统SVD系列模型
1. **FunkSVD**: 基础矩阵分解 `r = p_u^T * q_i`
2. **ClassicalSVD**: 带全局偏置的矩阵分解 `r = μ + p_u^T * q_i`
3. **BiasSVD**: 带完整偏置的矩阵分解 `r = μ + b_u + b_i + p_u^T * q_i`
4. **SVDattr**: 融入物品属性的矩阵分解

### 深度学习模型
1. **NeuralCF**: 神经协同过滤，结合GMF和MLP
2. **AutoInt**: 基于多头注意力机制的特征交互学习

## 数据划分方法

1. **全局随机划分**: 随机将训练数据按9:1划分为训练集和验证集
2. **按用户划分**: 对每个用户随机选择一个评分作为验证集

## 评估指标

- **AUC**: 分类性能评估
- **RMSE**: 均方根误差  
- **MAE**: 平均绝对误差
- **训练时间**: 每轮训练时间
- **模型大小**: 模型参数内存占用

## 运行方法

### SVD系列模型
```bash
python SVD_all_models.py
```

### NeuralCF模型
```bash  
python neuralcf_model.py
```

### AutoInt模型
```bash
python autoint_model.py
```

## 输出文件

### 预测结果
- SVD模型: `results/[模型名]_[划分方法]_predictions.txt`
- NeuralCF: `results/NeuralCF_[划分方法]_predictions.txt`  
- AutoInt: `results/AutoInt_[划分方法]_predictions.txt`

### 性能报告
- SVD系列: `results/svd_performance_comparison.csv`
- NeuralCF: `results/neuralcf_performance_comparison.csv`
- AutoInt: `results/autoint_performance_comparison.csv`

### 训练曲线
- 保存在 `figures/` 目录下的PNG文件
- 包含Loss、AUC、RMSE、MAE曲线

## 特性

### 模型特色
- **SVD系列**: 经典推荐算法，训练快速，效果稳定
- **NeuralCF**: 结合矩阵分解和深度学习优势
- **AutoInt**: 使用注意力机制自动学习特征交互

### 训练优化
- 早停机制防止过拟合
- GPU加速支持
- 内存使用监控
- 训练过程可视化

### 配置管理
- 统一的超参数配置
- 支持不同训练模式
- 模型保存和加载

## 依赖环境

```bash
pip install -r requirements.txt
```

主要依赖:
- torch>=1.9.0
- scikit-learn>=1.0.0
- pandas>=1.3.0
- numpy>=1.21.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- tqdm>=4.60.0
