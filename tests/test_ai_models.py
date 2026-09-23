"""AI 模型测试"""

import numpy as np
import pytest

from src.ai.feature_engineering import AdvancedFeatureExtractor
from src.ai.sklearn_selector import SklearnClusterSelector
from src.models.network import Network


class TestSklearnClusterSelector:
    """Sklearn 簇头选择器测试"""

    @pytest.fixture
    def selector(self):
        return SklearnClusterSelector(model_type="rf", n_estimators=10)

    @pytest.fixture
    def sample_data(self):
        """生成样本数据"""
        np.random.seed(42)

        n_samples = 100
        n_features = 7

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)

        return X, y

    def test_predict_untrained(self, selector):
        """测试未训练模型的预测"""
        X = np.random.randn(10, 7)

        probs = selector.predict(X)

        assert len(probs) == 10
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_train_and_predict(self, selector, sample_data):
        """测试训练和预测"""
        X, y = sample_data

        selector.train(X, y)

        assert selector.is_trained

        predictions = selector.predict(X)

        assert len(predictions) == len(y)

    def test_feature_importance(self, selector, sample_data):
        """测试特征重要性"""
        X, y = sample_data

        selector.train(X, y)

        importance = selector.get_feature_importance()

        assert len(importance) == len(selector.feature_names)
        assert all(v >= 0 for v in importance.values())


class TestAdvancedFeatureExtractor:
    """高级特征提取器测试"""

    @pytest.fixture
    def network(self):
        return Network(
            n_nodes=20,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            seed=42,
        )

    @pytest.fixture
    def extractor(self, network):
        return AdvancedFeatureExtractor(network)

    def test_extract_batch_shape(self, extractor, network):
        """测试批量特征提取的形状"""
        features = extractor.extract_batch(network.alive_nodes)

        assert features.shape[0] == len(network.alive_nodes)
        assert features.shape[1] == len(extractor.get_feature_names())

    def test_feature_names_include_core_fields(self, extractor):
        """测试特征名包含核心字段"""
        names = extractor.get_feature_names()

        assert "x" in names
        assert "y" in names
        assert "energy" in names
        assert "dist_to_bs" in names

    def test_compute_stats(self, extractor, network):
        """测试全局统计信息"""
        stats = extractor.compute_stats()

        assert stats.n_alive == len(network.alive_nodes)
        assert stats.mean_energy > 0
        assert stats.density > 0

    def test_normalize_features_zscore(self, extractor):
        """测试 zscore 标准化"""
        X = np.random.randn(100, 5)

        X_scaled = extractor.normalize_features(X, method="zscore")

        mean = np.mean(X_scaled, axis=0)
        std = np.std(X_scaled, axis=0)

        assert np.allclose(mean, 0, atol=1e-8)
        assert np.allclose(std, 1, atol=1e-8)

    def test_normalize_features_minmax(self, extractor):
        """测试 minmax 归一化"""
        X = np.random.randn(100, 5)

        X_scaled = extractor.normalize_features(X, method="minmax")

        min_vals = np.min(X_scaled, axis=0)
        max_vals = np.max(X_scaled, axis=0)

        assert np.allclose(min_vals, 0, atol=1e-8)
        assert np.allclose(max_vals, 1, atol=1e-8)

    def test_normalize_features_empty(self, extractor):
        """测试空输入原样返回"""
        empty = np.array([])

        assert extractor.normalize_features(empty, method="zscore").shape == (0,)
