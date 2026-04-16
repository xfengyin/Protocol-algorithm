# Protocol-algorithm 使用教程

## 快速开始

### 1. 基本仿真

```python
from src.models.network import Network
from src.energy.radio_model import FirstOrderRadioModel
from src.leach.classic import ClassicLEACH

# 创建网络
network = Network(
    n_nodes=100,
    area=(0, 100, 0, 100),
    base_station_pos=(50, 50),
    initial_energy=0.5
)

# 运行仿真
results = network.simulate_network(rounds=5000)

print(f"Network Lifetime: {results['network_lifetime']} rounds")
```

### 2. 使用不同协议

```python
# LEACH-C
results = network.simulate_network(rounds=5000, protocol_name='leach-c')

# LEACH-EE
results = network.simulate_network(rounds=5000, protocol_name='leach-ee')

# LEACH-M
results = network.simulate_network(rounds=5000, protocol_name='leach-m')
```

### 3. 命令行使用

```bash
# 运行经典 LEACH
python cli.py --config config/config.yaml --output results/leach/

# 运行 AI 增强版本
python cli.py --config config/leach_ai.yaml --output results/ai/
```

### 4. 生成训练数据

```python
from src.data.generator import DataGenerator

generator = DataGenerator(n_nodes=100)

# 生成数据集
X, y = generator.generate_balanced_dataset(n_rounds=500)

# 保存
generator.save_dataset(X, y, 'data/training_set.csv')
```

### 5. 训练 AI 模型

```python
from src.ai.trainer import AITrainer

trainer = AITrainer(model_type='sklearn', n_estimators=100)

# 训练
trainer.train(X, y)

# 保存
trainer.save('models/ai_selector.pkl')
```

### 6. 可视化结果

```python
from src.visualization.metrics_plots import MetricsPlotter

plotter = MetricsPlotter()
plotter.plot_all_metrics(results, output_dir='results/plots')
```

## 配置说明

详见 `config/config.yaml`

## 扩展协议

```python
from src.leach.base import LEACHProtocol
from src.leach.variants import LEACHRegistry

class MyLEACH(LEACHProtocol):
    def select_cluster_heads(self, network, **kwargs):
        # 自定义簇头选择逻辑
        ...

# 注册
LEACHRegistry.register('my_leach', MyLEACH)
```
