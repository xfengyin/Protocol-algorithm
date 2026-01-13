# 演示脚本
"""自动运行LEACH算法的演示脚本，用于验证优化效果"""

from src.leach import generate_nodes, clustering, show_clusters

def demo():
    """
    运行LEACH算法演示，验证优化效果
    """
    # 生成节点
    N = 50  # 节点数量
    print(f"使用 {N} 个节点运行LEACH算法演示...")
    
    nodes, _ = generate_nodes(N)  # 忽略flag，因为现在使用Node类的状态
    
    # 运行分簇算法
    print("\n开始分簇...")
    iter_cluster = clustering(nodes, k=5)  # 运行5轮
    
    # 显示分簇结果
    print("\n分簇结果汇总：")
    for i, clusters in enumerate(iter_cluster):
        print(f"\n第 {i+1} 轮分簇结果：")
        for j, cluster in enumerate(clusters):
            print(f"  簇 {j+1}: 簇首位于 {cluster['head']}, 成员数: {len(cluster['members'])}")
    
    # 提取可视化所需的簇结构
    visualization_clusters = []
    for clusters in iter_cluster:
        cluster_list = []
        for cluster in clusters:
            # 每个簇的结构：[簇首, 成员1, 成员2, ...]
            cluster_data = [cluster['head']] + cluster['members']
            cluster_list.append(cluster_data)
        visualization_clusters.append(cluster_list)
    
    # 显示可视化结果
    print("\n生成可视化图表...")
    for i, cluster in enumerate(visualization_clusters):
        print(f"\n第 {i+1} 轮分簇可视化")
        show_clusters([cluster])

if __name__ == "__main__":
    demo()