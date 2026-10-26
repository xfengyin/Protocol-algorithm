"""AI 模型测试"""

import numpy as np
import pytest

from src.ai.selector import NARROW_FEATURE_NAMES
from src.ai.sklearn_selector import SklearnClusterSelector
from src.models.network import Network


def make_dataset(n_features: int, n_samples: int = 60) -> tuple:
    """构造两类均衡的随机数据集。"""
    rng = np.random.RandomState(0)
    X = rng.rand(n_samples, n_features)
    y = np.array([i % 2 for i in range(n_samples)])
    return X, y


def build_network(seed: int) -> Network:
    """构造一个固定参数的测试网络。

    各节点能量刻意不同：网络刚建好时所有节点能量相同，归一化后 energy / energy_ratio
    两列都会变成常数 0.0，那样「取第 4 列还是第 3 列」根本区分不出来。
    """
    network = Network(n_nodes=20, area=(0, 100, 0, 100), base_station_pos=(50, 50), seed=seed)
    # initial_energy 也要逐节点不同：energy 与 energy_ratio 若只差一个常数倍，
    # z-score 之后两列会完全相同，就没法证明解释取的是第 4 列而不是第 3 列。
    for index, node in enumerate(network.alive_nodes):
        node.initial_energy = 0.3 + 0.03 * index
        node.energy = node.initial_energy * (0.4 + 0.01 * index)
    return network


def test_feature_names_match_extract_order():
    """19 维特征的名字与取值顺序必须同源（改一处漏一处会让名字和值错位）。

    断言里刻意用 names.index(...) 去取列：如果只改了名字顺序、没改取值顺序，
    这里会立刻红 —— 单看长度是发现不了的。
    """
    from src.ai.feature_engineering import ADVANCED_FEATURE_NAMES, AdvancedFeatureExtractor

    network = build_network(seed=3)
    extractor = AdvancedFeatureExtractor(network)
    node = network.alive_nodes[0]
    bs_pos = network.base_station.position

    names = extractor.get_feature_names()
    values = extractor.extract_features(node)

    assert names == list(ADVANCED_FEATURE_NAMES)
    assert len(names) == len(values) == 19
    assert values[names.index("energy_ratio")] == pytest.approx(node.energy / node.initial_energy)
    assert values[names.index("dist_to_bs")] == pytest.approx(node.distance_to(bs_pos))
    assert values[names.index("energy")] == pytest.approx(node.energy)


class TestSklearnClusterSelector:
    """Sklearn 簇头选择器测试"""

    @pytest.fixture
    def selector(self):
        return SklearnClusterSelector(estimator="rf", n_estimators=10)

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

    def test_feature_names_follow_training_width(self):
        """特征名跟着训练宽度走，不再固定 7 个。"""
        selector = SklearnClusterSelector(estimator="rf", n_estimators=5)
        X, y = make_dataset(n_features=19, n_samples=60)

        selector.train(X, y)

        assert len(selector.feature_names) == 19
        assert len(selector.get_feature_importance()) == 19

    def test_feature_importance_raises_on_name_width_mismatch(self):
        """19 宽的模型配上 7 个名字时必须抛错 —— 不许静默截断成错位的表。"""
        selector = SklearnClusterSelector(estimator="rf", n_estimators=5)
        X, y = make_dataset(n_features=19, n_samples=60)
        selector.train(X, y)

        # 模拟旧版本 checkpoint 的状态：模型 19 宽，feature_names 只有 7 个
        selector.feature_names = list(NARROW_FEATURE_NAMES)

        with pytest.raises(ValueError, match="特征名与模型输入宽度不一致"):
            selector.get_feature_importance()

    def test_explain_prediction_uses_wide_row(self):
        """宽特征模型上 explain_prediction 能跑，且 energy_ratio 取的是第 4 列。"""
        network = build_network(seed=1)
        selector = SklearnClusterSelector(estimator="rf", n_estimators=10)
        X = selector._extract_features(network)
        y = np.array([i % 2 for i in range(len(X))])
        selector.train(X, y)

        node = network.alive_nodes[0]
        result = selector.explain_prediction(node, network)

        row = X[0]
        energy_ratio_index = selector.feature_names.index("energy_ratio")
        assert energy_ratio_index == 4, "宽特征布局里 energy_ratio 应当在第 4 列"
        assert row[3] != row[4], "夹具失效：第 3/4 列同值，无法区分取错列"
        assert result["features"]["energy_ratio"] == pytest.approx(float(row[4]))
        assert result["features"]["energy_ratio"] != pytest.approx(float(row[3]))

    def test_explain_prediction_narrow_model_still_works(self):
        """窄特征（7 宽）模型上 explain_prediction 保持原行为。"""
        network = build_network(seed=2)
        bs_pos = network.base_station.position
        selector = SklearnClusterSelector(estimator="rf", n_estimators=10)
        X = np.array([n.get_features(bs_pos) for n in network.alive_nodes])
        y = np.array([i % 2 for i in range(len(X))])
        selector.train(X, y)

        node = network.alive_nodes[0]
        result = selector.explain_prediction(node, network)

        assert result["features"]["energy_ratio"] == pytest.approx(
            node.energy / node.initial_energy
        )
        assert result["features"]["dist_to_bs"] == pytest.approx(node.distance_to(bs_pos))

    def test_feature_importance(self, selector, sample_data):
        """测试特征重要性"""
        X, y = sample_data

        selector.train(X, y)

        importance = selector.get_feature_importance()

        # 断言「键数 == 模型实际输入宽度」，不是 == len(selector.feature_names)。
        # 旧版本里 feature_names 固定 7 个、模型却可能按更宽的列训练，用后者做断言
        # 等于给 zip 的静默截断发合格证（重要性贴到错误的名字上照样通过）。
        assert len(importance) == X.shape[1]
        assert list(importance) == selector.feature_names
        assert all(v >= 0 for v in importance.values())
