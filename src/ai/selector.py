"""AI 簇头选择器基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..models.cluster_head import ClusterHead
from ..models.network import Network
from ..models.node import Node

# 窄特征布局：Node.get_features 的顺序，逐位对应。examples/ 与 README 的入门路径依赖它。
NARROW_FEATURE_NAMES: tuple[str, ...] = (
    "x",
    "y",
    "energy",
    "energy_ratio",
    "dist_to_bs",
    "transmissions",
    "receptions",
)


def feature_names_for_width(width: int) -> List[str]:
    """按模型实际输入宽度给出特征名。

    名字必须与模型实际吃进去的列一一对应，否则 get_feature_importance 会把重要性
    贴到错误的名字上（静默错位，不报错）。宽度对不上任何已知布局时返回
    feature_00.. 占位名：宁可名字难看，不可名字说谎。

    Args:
        width: 模型输入的特征维度

    Returns:
        长度恰为 width 的特征名列表
    """
    if width == len(NARROW_FEATURE_NAMES):
        return list(NARROW_FEATURE_NAMES)

    try:
        from .feature_engineering import ADVANCED_FEATURE_NAMES

        wide_names = ADVANCED_FEATURE_NAMES
    except ImportError:  # pragma: no cover - 本仓内 feature_engineering 一定可导入
        wide_names = ()

    if width == len(wide_names):
        return list(wide_names)

    return [f"feature_{i:02d}" for i in range(width)]


class AIClusterSelector(ABC):
    """AI 簇头选择器基类"""

    def __init__(self, model_path: Optional[str] = None):
        """
        初始化

        Args:
            model_path: 模型保存路径
        """
        # 子类会分别在 _ensure_model/_load_model 里换成 sklearn / torch 的模型对象，
        # 所以这里必须声明成 Any —— 写 None 会让 mypy 把 self.model 定死为 NoneType。
        self.model: Any = None
        self.model_path = model_path
        self.is_trained = False
        # 未训练时的默认值（窄特征）。train() 会按实际训练宽度覆盖它。
        self.feature_names: List[str] = list(NARROW_FEATURE_NAMES)

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        预测簇头

        Args:
            features: 特征矩阵 (n_samples, n_features)

        Returns:
            预测概率或分数
        """

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        训练模型

        Args:
            X: 特征矩阵
            y: 标签 (1=簇头, 0=非簇头)
            **kwargs: 训练参数
        """

    def save(self, path: Optional[str] = None) -> None:
        """保存模型"""
        save_path = Path(path or self.model_path)
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_model(save_path)

    def load(self, path: Optional[str] = None) -> None:
        """加载模型"""
        load_path = Path(path or self.model_path)
        if load_path.exists():
            self._load_model(load_path)
            self.is_trained = True

    @abstractmethod
    def _save_model(self, path: Path) -> None:
        """保存模型实现"""

    @abstractmethod
    def _load_model(self, path: Path) -> None:
        """加载模型实现"""

    def get_feature_importance(self) -> Dict[str, float]:
        """
        特征重要性

        基类给出默认实现（不是每个选择器都能算重要性），sklearn / pytorch 两个子类各自覆盖。
        AITrainer.get_feature_importance 依赖它，所以它必须出现在基类契约里 ——
        之前只定义在子类，trainer 里那句调用在类型上就是「未声明的方法」。
        用默认实现而不是 abstractmethod：抽象方法会让外部已有的子类无法再实例化（公开 API 破坏）。
        """
        return {}

    def select_cluster_heads(
        self, network: Network, n_clusters: Optional[int] = None, **kwargs
    ) -> List[ClusterHead]:
        """
        基于 AI 模型选择簇头

        Args:
            network: 网络对象
            n_clusters: 目标簇数量

        Returns:
            簇头列表
        """
        alive_nodes = network.alive_nodes

        if n_clusters is None:
            n_clusters = max(1, int(len(alive_nodes) * 0.05))

        features = self._extract_features(network)

        scores = self.predict(features)

        cluster_id = 0
        cluster_heads = []
        selected_ids = set()

        sorted_indices = np.argsort(scores)[::-1]

        for idx in sorted_indices:
            if cluster_id >= n_clusters:
                break

            node = alive_nodes[idx]
            if node.id not in selected_ids:
                node.become_cluster_head(cluster_id)

                ch = ClusterHead(node=node, cluster_id=cluster_id)
                cluster_heads.append(ch)
                selected_ids.add(node.id)
                cluster_id += 1

        return cluster_heads

    def _extract_features(self, network: Network) -> np.ndarray:
        """
        提取节点特征

        Args:
            network: 网络对象

        Returns:
            特征矩阵
        """
        try:
            from ..ai.feature_engineering import AdvancedFeatureExtractor

            extractor = AdvancedFeatureExtractor(network)
            features = extractor.extract_batch(network.alive_nodes)
            return extractor.normalize_features(features, method="zscore")
        except ImportError:
            bs_pos = network.base_station.position
            features = []

            for node in network.alive_nodes:
                feat = node.get_features(bs_pos)

                neighbors = network.get_neighbors(node, 30.0)
                feat = np.append(feat, len(neighbors))
                feat = np.append(feat, node.distance_to(bs_pos))

                features.append(feat)

            return (
                np.array(features)
                if features
                else np.array([]).reshape(0, len(self.feature_names) + 2)
            )

    def _model_input_dim(self) -> Optional[int]:
        """模型实际期望的输入宽度；无法确定时返回 None。

        sklearn 模型有 n_features_in_；PyTorch 子类在 _ensure_model/_load_model 里
        记录 input_dim。两者都拿不到，说明模型状态不完整 —— 由调用方决定怎么报错。
        """
        n_features = getattr(self.model, "n_features_in_", None)
        if n_features is not None:
            return int(n_features)

        input_dim = getattr(self, "input_dim", None)
        if input_dim is not None:
            return int(input_dim)

        return None

    def _sync_feature_names(self, width: int) -> None:
        """训练后按实际宽度重设特征名（否则 get_feature_importance 会错位）。"""
        self.feature_names = feature_names_for_width(width)

    def _feature_row(self, node: Node, network: Network) -> np.ndarray:
        """按模型实际输入宽度取该节点的特征行（形状 (1, width)）。

        宽度不匹配时直接报错，而不是把一个宽度不对的向量丢给模型 ——
        后者只会以 "mat1 and mat2 shapes cannot be multiplied" 这类内部细节收场。
        """
        width = self._model_input_dim()
        if width is None:
            raise RuntimeError(
                "无法确定模型期望的特征宽度（模型既没有 n_features_in_，也没有记录 input_dim）。"
                "请用 train() 训练或加载本仓 _save_model 产出的模型文件。"
            )

        if len(self.feature_names) != width:
            raise ValueError(
                "特征名与模型输入宽度不一致：feature_names 有 %d 个，模型期望 %d 个。"
                "该模型要么由旧版本保存（那时 feature_names 不随训练宽度更新），"
                "要么是手工拼出来的。请用 train() 重新训练或重新导出模型。"
                % (len(self.feature_names), width)
            )

        if width == len(NARROW_FEATURE_NAMES):
            return np.asarray(node.get_features(network.base_station.position)).reshape(1, -1)

        matrix = self._extract_features(network)
        if matrix.ndim != 2 or matrix.shape[1] != width:
            raise RuntimeError(
                "特征提取器产出的宽度（%s）与模型期望的宽度（%d）不一致。"
                % (matrix.shape[1] if matrix.ndim == 2 else "?", width)
            )

        for index, alive in enumerate(network.alive_nodes):
            if alive.id == node.id:
                return matrix[index : index + 1]

        raise ValueError(
            "节点 %s 不在 alive_nodes 里（已死亡或不属于该网络），无法解释其预测。" % node.id
        )

    def explain_prediction(self, node: Node, network: Network) -> Dict[str, Any]:
        """
        解释为什么某个节点被选为簇头

        Args:
            node: 节点
            network: 网络对象

        Returns:
            解释字典
        """
        if not self.is_trained:
            return {"error": "Model not trained"}

        features = self._feature_row(node, network)
        prediction = self.predict(features)[0]

        # 解释项按特征名取，不再按位置硬编码：位置索引只在窄特征布局下成立，
        # 一旦模型按宽特征训练就会取错列（历史上的 shape 崩与错位都出在这里）。
        explained: Dict[str, float] = {}
        for name in ("energy_ratio", "dist_to_bs"):
            if name in self.feature_names:
                explained[name] = float(features[0][self.feature_names.index(name)])

        return {
            "node_id": node.id,
            "prediction_score": float(prediction),
            "features": explained,
        }

    def evaluate(
        self, X: np.ndarray, y_true: np.ndarray, threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        评估模型性能

        Args:
            X: 特征矩阵
            y_true: 真实标签
            threshold: 分类阈值

        Returns:
            评估指标字典
        """
        y_pred = (self.predict(X) >= threshold).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }


class EnsembleClusterSelector:
    """集成簇头选择器"""

    def __init__(self, selectors: Optional[List[AIClusterSelector]] = None):
        """
        初始化集成选择器

        Args:
            selectors: 选择器列表
        """
        self.selectors = selectors or []
        self.weights = [1.0] * len(self.selectors)

    def add_selector(self, selector: AIClusterSelector, weight: float = 1.0) -> None:
        """
        添加选择器

        Args:
            selector: 选择器实例
            weight: 权重
        """
        self.selectors.append(selector)
        self.weights.append(weight)

    def set_weights(self, weights: List[float]) -> None:
        """
        设置权重

        Args:
            weights: 权重列表
        """
        if len(weights) != len(self.selectors):
            raise ValueError("Weights length must match selectors length")
        self.weights = weights

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        """
        获取加权平均分数

        Args:
            features: 特征矩阵

        Returns:
            加权分数
        """
        if not self.selectors:
            raise ValueError("No selectors available")

        scores = np.zeros(len(features))
        total_weight = sum(self.weights)

        for selector, weight in zip(self.selectors, self.weights):
            scores += weight * selector.predict(features)

        return scores / total_weight

    def select_cluster_heads(
        self, network: Network, n_clusters: Optional[int] = None, **kwargs
    ) -> List[ClusterHead]:
        """
        集成选择簇头

        Args:
            network: 网络对象
            n_clusters: 目标簇数量
            **kwargs: 其他参数

        Returns:
            簇头列表
        """
        alive_nodes = network.alive_nodes

        if n_clusters is None:
            n_clusters = max(1, int(len(alive_nodes) * 0.05))

        if not self.selectors:
            raise ValueError("No selectors configured")

        features = self._extract_features(network)
        scores = self.predict_scores(features)

        cluster_id = 0
        cluster_heads = []
        selected_ids = set()

        sorted_indices = np.argsort(scores)[::-1]

        for idx in sorted_indices:
            if cluster_id >= n_clusters:
                break

            node = alive_nodes[idx]
            if node.id not in selected_ids:
                node.become_cluster_head(cluster_id)

                ch = ClusterHead(node=node, cluster_id=cluster_id)
                cluster_heads.append(ch)
                selected_ids.add(node.id)
                cluster_id += 1

        return cluster_heads

    def _extract_features(self, network: Network) -> np.ndarray:
        """提取特征"""
        if self.selectors:
            return self.selectors[0]._extract_features(network)

        bs_pos = network.base_station.position
        features = []

        for node in network.alive_nodes:
            feat = node.get_features(bs_pos)
            features.append(feat)

        return np.array(features)

    def get_selector_info(self) -> List[Dict[str, Any]]:
        """
        获取选择器信息

        Returns:
            选择器信息列表
        """
        return [
            {
                "type": selector.__class__.__name__,
                "weight": weight,
                "trained": selector.is_trained,
            }
            for selector, weight in zip(self.selectors, self.weights)
        ]
