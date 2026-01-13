# 数据生成脚本
"""用于生成训练AI模型所需的WSN节点数据和标注。"""

import numpy as np
import os

# 创建数据目录
os.makedirs('data', exist_ok=True)

def generate_wsn_data(node_counts: list):
    """
    生成不同规模的WSN节点数据。
    """
    datasets = []
    
    for count in node_counts:
        # 生成节点坐标
        nodes = [[np.random.random(), np.random.random()] for _ in range(count)]
        
        # 生成简单标注（10%的节点作为簇首）
        labels = [1 if i < int(count * 0.1) else 0 for i in range(count)]
        
        # 打乱标注
        np.random.shuffle(labels)
        
        datasets.append((nodes, labels))
    
    return datasets

def generate_diverse_distributions(node_count: int, num_distributions: int):
    """
    生成不同分布的节点数据。
    """
    distributions = []
    
    # 1. 均匀分布
    uniform_nodes = [[np.random.random(), np.random.random()] for _ in range(node_count)]
    distributions.append(uniform_nodes)
    
    # 2. 中心集中分布
    center_nodes = [[np.random.normal(0.5, 0.1), np.random.normal(0.5, 0.1)] for _ in range(node_count)]
    distributions.append(center_nodes)
    
    # 3. 随机分布
    random_nodes = [[np.random.random(), np.random.random()] for _ in range(node_count)]
    distributions.append(random_nodes)
    
    # 4. 环形分布
    ring_nodes = []
    for _ in range(node_count):
        angle = np.random.random() * 2 * np.pi
        radius = 0.3 + np.random.random() * 0.2
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        ring_nodes.append([x, y])
    distributions.append(ring_nodes)
    
    # 5. 网格分布
    grid_size = int(np.sqrt(node_count))
    grid_nodes = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = (i + 0.5) / grid_size + np.random.normal(0, 0.02)
            y = (j + 0.5) / grid_size + np.random.normal(0, 0.02)
            grid_nodes.append([x, y])
    distributions.append(grid_nodes)
    
    return distributions

def balance_dataset(nodes, labels):
    """
    平衡数据集，确保正负样本比例合理。
    """
    # 分离正负样本
    positive = [node for node, label in zip(nodes, labels) if label == 1]
    negative = [node for node, label in zip(nodes, labels) if label == 0]
    
    # 计算样本数量
    num_pos = len(positive)
    num_neg = len(negative)
    
    # 平衡样本
    if num_pos > num_neg:
        # 正样本过多，随机下采样
        sampled_pos = np.random.choice(len(positive), num_neg, replace=False)
        balanced_nodes = [positive[i] for i in sampled_pos] + negative
        balanced_labels = [1] * num_neg + [0] * num_neg
    else:
        # 负样本过多，随机下采样
        sampled_neg = np.random.choice(len(negative), num_pos, replace=False)
        balanced_nodes = positive + [negative[i] for i in sampled_neg]
        balanced_labels = [1] * num_pos + [0] * num_pos
    
    # 打乱顺序
    indices = np.arange(len(balanced_nodes))
    np.random.shuffle(indices)
    
    balanced_nodes = [balanced_nodes[i] for i in indices]
    balanced_labels = [balanced_labels[i] for i in indices]
    
    return balanced_nodes, balanced_labels

# 生成不同规模的WSN数据
print("生成不同规模的WSN数据...")
datasets = generate_wsn_data(node_counts=[10, 50, 100, 200, 500])

# 生成不同分布的节点数据
print("生成不同分布的节点数据...")
distributions = generate_diverse_distributions(node_count=200, num_distributions=5)

# 合并所有数据
all_nodes = []
all_labels = []

# 处理不同规模的数据
for i, (nodes, labels) in enumerate(datasets):
    print(f"处理规模 {i+1} 的数据...")
    
    # 平衡数据集
    balanced_nodes, balanced_labels = balance_dataset(nodes, labels)
    
    # 添加到总数据集
    all_nodes.extend(balanced_nodes)
    all_labels.extend(balanced_labels)

# 处理不同分布的数据
for i, nodes in enumerate(distributions):
    print(f"处理分布 {i+1} 的数据...")
    
    # 生成简单标注
    labels = [1 if i < int(len(nodes) * 0.1) else 0 for i in range(len(nodes))]
    np.random.shuffle(labels)
    
    # 平衡数据集
    balanced_nodes, balanced_labels = balance_dataset(nodes, labels)
    
    # 添加到总数据集
    all_nodes.extend(balanced_nodes)
    all_labels.extend(balanced_labels)

print(f"总数据量: {len(all_nodes)} 个节点")
print(f"簇首数量: {sum(all_labels)}")
print(f"簇首比例: {sum(all_labels) / len(all_nodes):.2f}")

# 保存数据到文件
np.save('data/all_nodes.npy', np.array(all_nodes))
np.save('data/all_labels.npy', np.array(all_labels))
print("数据已保存到 data/ 目录")

# 生成训练数据
print("生成训练数据...")
X = np.array(all_nodes)

# 添加简单特征：与中心点的距离
center = np.mean(X, axis=0)
distances = np.sqrt(np.sum((X - center)**2, axis=1)).reshape(-1, 1)
X_with_features = np.hstack([X, distances])

y = np.array(all_labels).reshape(-1, 1)

# 保存训练数据
np.save('data/train_X.npy', X_with_features)
np.save('data/train_y.npy', y)
print("训练数据已保存")
print("数据生成完成！")
