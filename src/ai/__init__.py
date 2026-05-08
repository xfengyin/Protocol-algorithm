"""AI 优化模块"""

from .selector import AIClusterSelector, EnsembleClusterSelector
from .sklearn_selector import SklearnClusterSelector
from .feature_engineering import AdvancedFeatureExtractor, FeatureSelector

try:
    from .pytorch_selector import PyTorchClusterSelector
except ImportError:
    PyTorchClusterSelector = None

try:
    from .trainer import AITrainer
except ImportError:
    AITrainer = None

__all__ = [
    "AIClusterSelector",
    "EnsembleClusterSelector",
    "SklearnClusterSelector",
    "PyTorchClusterSelector",
    "AdvancedFeatureExtractor",
    "FeatureSelector",
    "AITrainer",
]
