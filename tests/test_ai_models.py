"""AI 模型测试"""

import pytest
import numpy as np

from src.ai.sklearn_selector import SklearnClusterSelector


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
