# LEACH协议算法可视化模块
"""LEACH协议算法的可视化功能，用于显示分簇结果和网络拓扑。"""

from typing import List
import matplotlib.pyplot as plt

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
