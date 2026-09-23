"""AI 优化模块"""

from .feature_engineering import AdvancedFeatureExtractor, FeatureSelector
from .selector import AIClusterSelector, EnsembleClusterSelector
from .sklearn_selector import SklearnClusterSelector

try:
    from .pytorch_selector import PyTorchClusterSelector
except ImportError:
    PyTorchClusterSelector = None  # type: ignore[misc]

try:
    from .trainer import AITrainer
except ImportError:
    AITrainer = None  # type: ignore[misc]

__all__ = [
    "AIClusterSelector",
    "EnsembleClusterSelector",
    "SklearnClusterSelector",
    "PyTorchClusterSelector",
    "AdvancedFeatureExtractor",
    "FeatureSelector",
    "AITrainer",
]
