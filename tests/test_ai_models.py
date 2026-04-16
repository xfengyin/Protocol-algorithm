"""AI 模型测试"""

import pytest
import numpy as np

from src.ai.sklearn_selector import SklearnClusterSelector
from src.ai.feature_engineering import FeatureEngineer


class TestSklearnClusterSelector:
    """Sklearn 簇头选择器测试"""
    
    @pytest.fixture
    def selector(self):
        return SklearnClusterSelector(model_type='rf', n_estimators=10)
    
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


class TestFeatureEngineer:
    """特征工程测试"""
    
    @pytest.fixture
    def engineer(self):
        return FeatureEngineer()
    
    def test_create_basic_features(self, engineer):
        """测试创建基础特征"""
        x = np.array([10, 20, 30])
        y = np.array([15, 25, 35])
        energy = np.array([0.5, 0.4, 0.3])
        dist_to_bs = np.array([50, 40, 30])
        
        features, names = engineer.create_basic_features(x, y, energy, dist_to_bs)
        
        assert features.shape[0] == 3
        assert len(names) == features.shape[1]
        assert 'x' in names
        assert 'energy' in names
    
    def test_scale_features_standard(self, engineer):
        """测试标准化"""
        X = np.random.randn(100, 5)
        
        X_scaled = engineer.scale_features(X, method='standard')
        
        mean = np.mean(X_scaled, axis=0)
        std = np.std(X_scaled, axis=0)
        
        assert np.allclose(mean, 0, atol=1e-10)
        assert np.allclose(std, 1, atol=1e-10)
    
    def test_scale_features_minmax(self, engineer):
        """测试归一化"""
        X = np.random.randn(100, 5)
        
        X_scaled = engineer.scale_features(X, method='minmax')
        
        min_vals = np.min(X_scaled, axis=0)
        max_vals = np.max(X_scaled, axis=0)
        
        assert np.allclose(min_vals, 0)
        assert np.allclose(max_vals, 1)
