"""sklearn 簇头选择器"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .selector import AIClusterSelector


class SklearnClusterSelector(AIClusterSelector):
    """基于 sklearn 的簇头选择器"""

    def __init__(
        self,
        estimator: str = "rf",
        model_path: Optional[str] = None,
        n_estimators: int = 100,
        **kwargs,
    ):
        """
        初始化

        Args:
            estimator: 估计器种类 ('rf' 随机森林 或 'gb' 梯度提升)。
                **不叫 `model_type`** —— 那是 `AITrainer` 的说法，它指的是
                `sklearn` / `pytorch`（框架）。同一个名字在两个模块里指两件事，
                是「同名不同物」，已在此统一：这里一律叫 `estimator`。
            model_path: 模型保存路径
            n_estimators: 决策树数量
            **kwargs: sklearn 参数
        """
        super().__init__(model_path)

        self.estimator = estimator
        self.n_estimators = n_estimators
        self.scaler = StandardScaler()

        # 创建模型
        if estimator == "rf":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                class_weight="balanced",  # 处理不平衡
                **kwargs,
            )
        elif estimator == "gb":
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators, random_state=42, **kwargs
            )
        else:
            raise ValueError(f"Unknown estimator: {estimator}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        预测簇头概率

        Args:
            features: 特征矩阵

        Returns:
            簇头概率
        """
        if not self.is_trained:
            # 未训练时返回均匀分布
            return np.ones(len(features)) / len(features)

        # 标准化
        features_scaled = self.scaler.transform(features)

        # 预测概率
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features_scaled)
            if probs.shape[1] == 2:
                return probs[:, 1]  # 正类概率
            return probs.ravel()
        else:
            return self.model.predict(features_scaled).astype(float)

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        训练模型

        Args:
            X: 特征矩阵
            y: 标签
            **kwargs: 训练参数
        """
        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 训练
        self.model.fit(X_scaled, y, **kwargs)
        self.is_trained = True
        # 名字必须跟着模型的实际输入宽度走，否则 get_feature_importance 会把重要性
        # 贴到别的特征名下（zip 长度不等时静默截断）。
        self._sync_feature_names(X.shape[1])

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性。

        名字数量与模型输入宽度不一致时**抛错**，不截断 —— 截断会把重要性静默贴到
        错误的名字下：调用方拿到一张看起来正常、实际错位的表，比抛错危险得多。
        """
        if not self.is_trained or not hasattr(self.model, "feature_importances_"):
            return {}

        importances = self.model.feature_importances_
        if len(importances) != len(self.feature_names):
            raise ValueError(
                "特征名与模型输入宽度不一致：模型（%s）有 %d 个特征，feature_names 有 %d 个"
                "（来源：AIClusterSelector.feature_names）。旧版本保存的模型文件会出现这种情况 —— "
                "那时 feature_names 不随训练宽度更新。"
                % (type(self.model).__name__, len(importances), len(self.feature_names))
            )

        return {name: float(imp) for name, imp in zip(self.feature_names, importances)}

    def _save_model(self, path: Path):
        """保存模型"""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "estimator": self.estimator,
            "n_estimators": self.n_estimators,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    def _load_model(self, path: Path):
        """
        加载模型

        安全边界（bandit B301 的存量登记，见下方 nosec）：pickle 反序列化等价于
        执行文件里的任意代码。这里只应加载本仓 `_save_model` 自己产出的模型文件；
        把外部来源的 .pkl 喂给这个方法，等于把那个文件的作者请进本进程。
        """
        # B301 的存量登记：只加载本方法 docstring 约定的自产文件。
        # （理由写在这里而不是标记后面 —— bandit 会把标记后的文本当测试名解析并告警。）
        with open(path, "rb") as f:
            model_data = pickle.load(f)  # nosec B301

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]

        # 加载即校验：旧文件的 feature_names 固定是 7 个，模型却可能按 19 列训练；
        # 这种文件继续用下去只会得到错位的重要性表，所以在这里就报错。
        width = getattr(self.model, "n_features_in_", None)
        if width is not None and int(width) != len(self.feature_names):
            raise ValueError(
                "模型文件不一致：模型按 %d 个特征训练，文件里记的 feature_names 却是 %d 个。"
                "该文件由旧版本保存（那时 feature_names 不随训练宽度更新），请重新训练并保存。"
                % (int(width), len(self.feature_names))
            )
        self._sync_feature_names(len(self.feature_names))
        # 兼容 2026-10-17 之前保存的模型文件：那时这个键叫 "model_type"。
        self.estimator = model_data.get("estimator", model_data.get("model_type"))
        if self.estimator is None:
            raise KeyError("模型文件里既没有 estimator 也没有 model_type")
        self.n_estimators = model_data["n_estimators"]
