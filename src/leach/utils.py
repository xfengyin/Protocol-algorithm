# LEACH协议算法工具函数
"""LEACH协议算法的工具函数，包括距离计算等辅助功能。"""

from typing import List
import numpy as np

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
