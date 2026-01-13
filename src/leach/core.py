# LEACH协议算法核心模块
"""LEACH协议算法的核心实现，包括节点生成、簇首选择和分簇功能。"""

from typing import List, Tuple, Optional, Dict
import numpy as np

from .utils import distance

# 能量模型参数
INITIAL_ENERGY = 0.5  # 节点初始能量（焦耳）
E_ELEC = 50e-9        # 接收/发送电路能耗（焦耳/比特）
E_FS = 10e-12         # 自由空间信道模型能耗（焦耳/比特/平方米）
E_MP = 0.0013e-12     # 多径信道模型能耗（焦耳/比特/立方米）
THRESHOLD_DISTANCE = 87  # 自由空间/多径信道切换距离（米）
DATA_PACKET_SIZE = 2000  # 数据分组大小（比特）
CLUSTER_HEAD_PACKET_SIZE = 4000  # 簇首数据包大小（比特）

class Node:
    """传感器节点类，包含位置、能量和状态信息"""
    
    def __init__(self, x: float, y: float, initial_energy: float = INITIAL_ENERGY):
        """
        初始化节点
        
        参数:
        x: 节点x坐标
        y: 节点y坐标
        initial_energy: 初始能量
        """
        self.x = x
        self.y = y
        self.energy = initial_energy
        self.is_cluster_head = False
        self.cluster_id = -1
        self.rounds_as_head = 0
    
    def get_coordinates(self) -> List[float]:
        """获取节点坐标"""
        return [self.x, self.y]
    
    def consume_energy(self, distance: float, packet_size: int) -> None:
        """
        计算并消耗节点能量
        
        参数:
        distance: 通信距离
        packet_size: 数据包大小
        """
        # 接收能量消耗（接收端）
        self.energy -= E_ELEC * packet_size
        
        # 发送能量消耗（发送端）
        if distance < THRESHOLD_DISTANCE:
            # 自由空间模型
            self.energy -= E_ELEC * packet_size + E_FS * packet_size * (distance ** 2)
        else:
            # 多径模型
            self.energy -= E_ELEC * packet_size + E_MP * packet_size * (distance ** 4)
        
        # 确保能量不会为负
        self.energy = max(0, self.energy)

def generate_nodes(N: int) -> Tuple[List[Node], List[int]]:
    """
    生成传感器节点的集合，并初始化每个节点的簇首标记。
    
    参数:
    N: 传感器节点的数量。
    
    返回:
    nodes: 生成的节点对象列表。
    sign_point: 每个节点是否被选为簇首的标记列表，初始化为0。
    """
    nodes = [Node(np.random.random(), np.random.random()) for _ in range(N)]
    sign_point = [0] * N  # 初始化所有节点的簇首标记为0
    
    print('生成：', len(nodes), '个节点')
    print('初始化标记列表为', sign_point)
    return nodes, sign_point

def select_heads(
    r: int, 
    nodes: List[Node], 
    P: float = 0.05
) -> Tuple[List[Node], List[Node]]:
    """
    根据LEACH协议选择簇首节点，考虑节点能量。
    
    参数:
    r: 当前的轮数。
    nodes: 所有节点对象的列表。
    P: 选取簇首的概率因子，默认为0.05。
    
    返回:
    heads: 选取的簇首节点列表。
    members: 非簇首的成员节点列表。
    """
    # 计算阈值T(n)
    Tn = P / (1 - P * (r % (1 / P)))
    print(f'轮数 {r}，阈值T(n)为：{Tn}')

    heads = []
    members = []
    
    # 计算平均能量
    total_energy = sum(node.energy for node in nodes)
    avg_energy = total_energy / len(nodes) if nodes else 0
    
    for node in nodes:
        # 重置节点状态
        node.is_cluster_head = False
        node.cluster_id = -1
        
        # 只有能量高于平均能量的节点才有资格成为簇首
        if node.energy > avg_energy:
            # 为每个节点生成一个随机数
            rand = np.random.random()
            
            # 考虑节点已作为簇首的次数，减少频繁当选
            head_penalty = 0.1 * node.rounds_as_head
            adjusted_Tn = Tn * (1 - head_penalty)
            
            if rand <= adjusted_Tn:
                node.is_cluster_head = True
                node.rounds_as_head += 1
                heads.append(node)
            else:
                members.append(node)
        else:
            members.append(node)
    
    # 确保至少有一个簇首被选中
    if len(heads) == 0 and nodes:
        print("警告：没有节点被选为簇首，随机选择一个能量最高的节点作为簇首")
        # 选择能量最高的节点作为簇首
        nodes_sorted_by_energy = sorted(nodes, key=lambda x: x.energy, reverse=True)
        selected_node = nodes_sorted_by_energy[0]
        
        selected_node.is_cluster_head = True
        selected_node.rounds_as_head += 1
        
        heads = [selected_node]
        members = [node for node in nodes if node != selected_node]
    
    # 更新节点状态
    for i, head in enumerate(heads):
        head.cluster_id = i
    
    print(f'簇首为：{len(heads)}个')
    print(f'成员节点为：{len(members)}个')
    
    # 打印簇首能量信息
    print(f'簇首平均能量：{sum(head.energy for head in heads) / len(heads):.6f}')
    print(f'成员平均能量：{sum(member.energy for member in members) / len(members) if members else 0:.6f}')
    
    return heads, members

def clustering(
    nodes: List[Node], 
    k: int = 1
) -> List[Dict]:
    """
    对节点进行分簇操作，使用改进的簇成员分配算法。
    
    参数:
    nodes: 所有节点对象的列表。
    k: 进行分簇的轮数。
    
    返回:
    iter_cluster: 每轮分簇的结果，是一个字典列表，包含簇首和成员节点。
    """
    iter_cluster = []
    
    # 基站坐标（假设在中心）
    base_station = [0.5, 0.5]
    
    for r in range(k):
        print(f"\n=== 第 {r+1} 轮分簇 ===")
        
        # 选择簇首
        heads, members = select_heads(r, nodes)
        
        # 创建簇结构
        clusters = {i: {'head': head, 'members': []} for i, head in enumerate(heads)}
        
        # 将成员节点分配到最近的簇首
        for member in members:
            min_dist = float('inf')
            best_cluster = -1
            
            for i, cluster in clusters.items():
                # 计算成员节点到簇首的距离
                dist = distance(member.get_coordinates(), cluster['head'].get_coordinates())
                
                # 考虑簇首到基站的距离，平衡负载
                head_to_base = distance(cluster['head'].get_coordinates(), base_station)
                adjusted_dist = dist * (1 + 0.1 * head_to_base)
                
                if adjusted_dist < min_dist:
                    min_dist = adjusted_dist
                    best_cluster = i
            
            if best_cluster != -1:
                clusters[best_cluster]['members'].append(member)
                member.cluster_id = best_cluster
                
                # 计算并消耗成员节点向簇首发送数据的能量
                member.consume_energy(min_dist, DATA_PACKET_SIZE)
        
        # 簇首向基站发送数据，消耗能量
        for i, cluster in clusters.items():
            head = cluster['head']
            # 簇首需要接收所有成员的数据，然后发送给基站
            num_members = len(cluster['members'])
            
            # 计算簇首到基站的距离
            head_to_base = distance(head.get_coordinates(), base_station)
            
            # 簇首接收所有成员的数据
            head.energy -= E_ELEC * DATA_PACKET_SIZE * num_members
            
            # 簇首向基站发送聚合数据
            head.consume_energy(head_to_base, CLUSTER_HEAD_PACKET_SIZE)
            
            # 打印簇信息
            print(f"簇 {i+1}: 簇首位于 {head.get_coordinates()}, 能量: {head.energy:.6f}, 成员数: {num_members}")
        
        # 收集本轮分簇结果
        current_clusters = []
        for cluster in clusters.values():
            cluster_data = {
                'head': cluster['head'].get_coordinates(),
                'members': [member.get_coordinates() for member in cluster['members']]
            }
            current_clusters.append(cluster_data)
        
        iter_cluster.append(current_clusters)
        
        # 检查网络状态
        alive_nodes = [node for node in nodes if node.energy > 0]
        dead_nodes = [node for node in nodes if node.energy <= 0]
        
        print(f"\n网络状态：")
        print(f"存活节点数：{len(alive_nodes)}/{len(nodes)}")
        print(f"死亡节点数：{len(dead_nodes)}")
        print(f"平均节点能量：{sum(node.energy for node in alive_nodes) / len(alive_nodes) if alive_nodes else 0:.6f}")
        
        # 如果所有节点都死亡，停止分簇
        if not alive_nodes:
            print("所有节点已死亡，停止分簇")
            break
    
    return iter_cluster

def run():
    """
    主函数，用于运行整个分簇过程并显示结果。
    """
    from .visualization import show_clusters
    
    N = int(input('请输入节点的个数：\n'))
    nodes, _ = generate_nodes(N)  # 忽略flag，因为现在使用Node类的状态
    
    # 运行分簇算法
    iter_cluster = clustering(nodes, k=20)
    
    # 提取可视化所需的簇结构
    visualization_clusters = []
    for clusters in iter_cluster:
        cluster_list = []
        for cluster in clusters:
            # 每个簇的结构：[簇首, 成员1, 成员2, ...]
            cluster_data = [cluster['head']] + cluster['members']
            cluster_list.append(cluster_data)
        visualization_clusters.append(cluster_list)
    
    # 显示分簇结果
    for i, cluster in enumerate(visualization_clusters):
        print(f"\n第 {i+1} 轮分簇可视化")
        show_clusters([cluster])
