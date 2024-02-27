import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def distance(v_A: List[float], v_B: List[float]) -> float:
    """
    计算两个传感器节点之间的欧氏距离。
    
    参数:
    v_A: 第一个节点的坐标列表，形如[x, y]。
    v_B: 第二个节点的坐标列表，形如[x, y]。
    
    返回:
    两个节点之间的距离。
    """
    return np.sqrt((v_A[0] - v_B[0])**2 + (v_A[1] - v_B[1])**2)

def generate_nodes(N: int) -> Tuple[List[List[float]], List[int]]:
    """
    生成传感器节点的集合，并初始化每个节点的簇首标记。
    
    参数:
    N: 传感器节点的数量。
    
    返回:
    nodes: 生成的节点列表，每个节点是一个坐标列表[x, y]。
    sign_point: 每个节点是否被选为簇首的标记列表，初始化为0。
    """
    nodes = [[np.random.random(), np.random.random()] for _ in range(N)]
    sign_point = [0] * N  # 初始化所有节点的簇首标记为0
    
    print('生成：', len(nodes), '个节点')
    print('初始化标记列表为', sign_point)
    return nodes, sign_point

def select_heads(r: int, nodes: List[List[float]], flags: List[int], P: float = 0.05) -> Tuple[List[List[float]], List[List[float]]]:
    """
    根据LEACH协议选择簇首节点。
    
    参数:
    r: 当前的轮数。
    nodes: 所有节点的列表。
    flags: 节点的簇首标记列表。
    P: 选取簇首的概率因子，默认为0.05。
    
    返回:
    heads: 选取的簇首节点列表。
    members: 非簇首的成员节点列表。
    """
    Tn = P / (1 - P * (r % (1 / P)))  # 计算阈值T(n)
    print('阈值T(n)为：', Tn)

    heads, members = [], []
    rands = [np.random.random() for _ in range(len(nodes))]  # 为每个节点生成一个随机数
    
    for i, rand in enumerate(rands):
        if flags[i] == 0 and rand <= Tn:  # 如果节点未被选为簇首且随机数小于阈值，则选为簇首
            flags[i] = 1
            heads.append(nodes[i])
        else:
            members.append(nodes[i])
    
    print('簇首为：', len(heads), '个')
    print('成员节点为：', len(members), '个')
    return heads, members

def clustering(nodes: List[List[float]], flag: List[int], k: int = 1) -> List[List[List[float]]]:
    """
    对节点进行分簇操作。
    
    参数:
    nodes: 所有节点的列表。
    flag: 节点的簇首标记列表。
    k: 进行分簇的轮数。
    
    返回:
    iter_cluster: 每轮分簇的结果，是一个三维列表。
    """
    iter_cluster = []
    for r in range(k):
        heads, members = select_heads(r, nodes, flag)
        cluster = [[] for _ in range(len(heads))]
        
        for member in members:
            dist_min, head_clu = min((distance(member, head), i) for i, head in enumerate(heads))
            cluster[head_clu].append(member)
        
        for i, head in enumerate(heads):
            cluster[i].insert(0, head)  # 将簇首插入到对应簇的开头
        
        iter_cluster.append(cluster)
    return iter_cluster

def show_clusters(clusters: List[List[List[float]]]):
    """
    显示分簇结果的图形。
    
    参数:
    clusters: 分簇的结果，是一个三维列表。
    """
    fig, ax = plt.subplots()
    ax.set_title("WSN Clustering")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    colors = ['r', 'b', 'g', 'c', 'y', 'm']
    markers = ['o', '*', '.', 'x', '+', 's']
    
    for i, cluster in enumerate(clusters):
        centor = cluster[0]
        for point in cluster:
            ax.plot([centor[0], point[0]], [centor[1], point[1]], c=colors[i % len(colors)], marker=markers[i % len(markers)], alpha=0.4)
    
    plt.show()

def run():
    """
    主函数，用于运行整个分簇过程并显示结果。
    """
    N = int(input('请输入节点的个数：\n'))
    nodes, flag = generate_nodes(N)
    iter_cluster = clustering(nodes, flag, k=20)
    
    for cluster in iter_cluster:
        show_clusters(cluster)

if __name__ == '__main__':
    run()
