"""AI 优化模块"""

from .feature_engineering import AdvancedFeatureExtractor, FeatureSelector
from .selector import AIClusterSelector, EnsembleClusterSelector
from .sklearn_selector import SklearnClusterSelector

try:
    from .pytorch_selector import PyTorchClusterSelector
except ImportError:
    PyTorchClusterSelector = None  # type: ignore[misc]  # 降级：mypy 把已 import 的名字当类型，赋 None 报 misc

try:
    from .trainer import AITrainer
except ImportError:
    AITrainer = None  # type: ignore[misc]  # 依赖缺失时的降级（同上）

__all__ = [
    "AIClusterSelector",
    "EnsembleClusterSelector",
    "SklearnClusterSelector",
    "PyTorchClusterSelector",
    "AdvancedFeatureExtractor",
    "FeatureSelector",
    "AITrainer",
]
