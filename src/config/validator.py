"""配置验证器"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict, Union
from pathlib import Path
import yaml
import json


@dataclass
class NetworkConfig:
    """网络配置（带验证）"""
    n_nodes: int = 100
    area: Tuple[float, float, float, float] = (0, 100, 0, 100)
    base_station_pos: Tuple[float, float] = (50, 50)
    initial_energy: float = 0.5
    seed: Optional[int] = None

    def __post_init__(self):
        """验证配置"""
        self._validate()

    def _validate(self) -> None:
        """验证配置有效性"""
        if self.n_nodes < 1:
            raise ValueError(f"n_nodes must be >= 1, got {self.n_nodes}")

        if self.initial_energy <= 0:
            raise ValueError(f"initial_energy must be positive, got {self.initial_energy}")

        x_min, x_max, y_min, y_max = self.area
        if x_max <= x_min:
            raise ValueError(f"Invalid area x bounds: {self.area}")
        if y_max <= y_min:
            raise ValueError(f"Invalid area y bounds: {self.area}")

        bs_x, bs_y = self.base_station_pos
        if not (x_min <= bs_x <= x_max and y_min <= bs_y <= y_max):
            raise ValueError(
                f"Base station position {self.base_station_pos} must be within area {self.area}"
            )


@dataclass
class SimulationConfig:
    """仿真配置（带验证）"""
    rounds: int = 1000
    protocol_name: str = "leach"
    data_size: int = 4000
    stop_at_first_death: bool = False
    stop_at_half_death: bool = False
    checkpoint_interval: int = 0

    def __post_init__(self):
        """验证配置"""
        self._validate()

    def _validate(self) -> None:
        """验证配置有效性"""
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {self.rounds}")

        if self.data_size <= 0:
            raise ValueError(f"data_size must be positive, got {self.data_size}")

        if self.checkpoint_interval < 0:
            raise ValueError(f"checkpoint_interval must be >= 0, got {self.checkpoint_interval}")


@dataclass
class EnergyConfig:
    """能量模型配置"""
    model_name: str = "first_order"
    E_elec: float = 50e-9
    epsilon_fs: float = 10e-12
    epsilon_mp: float = 0.0013e-12
    d_threshold: float = 87.0
    E_da: float = 5e-9

    def __post_init__(self):
        """验证配置"""
        self._validate()

    def _validate(self) -> None:
        """验证配置有效性"""
        if self.model_name not in ['first_order', 'mica2', 'rssi', 'adaptive']:
            raise ValueError(f"Unknown energy model: {self.model_name}")

        if self.E_elec <= 0:
            raise ValueError(f"E_elec must be positive, got {self.E_elec}")

        if self.d_threshold <= 0:
            raise ValueError(f"d_threshold must be positive, got {self.d_threshold}")


@dataclass
class AIConfig:
    """AI 配置"""
    enabled: bool = False
    model_type: str = "sklearn"
    n_estimators: int = 100
    learning_rate: float = 0.1
    feature_normalization: str = "zscore"
    use_ensemble: bool = False

    def __post_init__(self):
        """验证配置"""
        self._validate()

    def _validate(self) -> None:
        """验证配置有效性"""
        if self.model_type not in ['sklearn', 'pytorch', 'lightgbm']:
            raise ValueError(f"Unknown model type: {self.model_type}")

        if self.feature_normalization not in ['zscore', 'minmax', 'robust', 'none']:
            raise ValueError(f"Unknown normalization: {self.feature_normalization}")

        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}")

        if self.learning_rate <= 0 or self.learning_rate > 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")


@dataclass
class VisualizationConfig:
    """可视化配置"""
    enabled: bool = True
    save_plots: bool = False
    save_animation: bool = False
    output_dir: str = "results"
    animation_interval: int = 200
    animation_fps: int = 10
    figsize: Tuple[int, int] = (12, 10)
    dpi: int = 100

    def __post_init__(self):
        """验证配置"""
        self._validate()

    def _validate(self) -> None:
        """验证配置有效性"""
        if self.animation_interval <= 0:
            raise ValueError(f"animation_interval must be positive, got {self.animation_interval}")

        if self.animation_fps <= 0:
            raise ValueError(f"animation_fps must be positive, got {self.animation_fps}")

        if self.dpi <= 0:
            raise ValueError(f"dpi must be positive, got {self.dpi}")


@dataclass
class FullConfig:
    """完整配置"""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "FullConfig":
        """
        从 YAML 文件加载配置

        Args:
            path: YAML 文件路径

        Returns:
            FullConfig 实例
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "FullConfig":
        """
        从 JSON 文件加载配置

        Args:
            path: JSON 文件路径

        Returns:
            FullConfig 实例
        """
        with open(path, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FullConfig":
        """
        从字典加载配置

        Args:
            data: 配置字典

        Returns:
            FullConfig 实例
        """
        network_data = data.get('network', {})
        simulation_data = data.get('simulation', {})
        energy_data = data.get('energy', {})
        ai_data = data.get('ai', {})
        visualization_data = data.get('visualization', {})

        return cls(
            network=NetworkConfig(**network_data),
            simulation=SimulationConfig(**simulation_data),
            energy=EnergyConfig(**energy_data),
            ai=AIConfig(**ai_data),
            visualization=VisualizationConfig(**visualization_data)
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典

        Returns:
            配置字典
        """
        return {
            'network': {
                'n_nodes': self.network.n_nodes,
                'area': list(self.network.area),
                'base_station_pos': list(self.network.base_station_pos),
                'initial_energy': self.network.initial_energy,
                'seed': self.network.seed,
            },
            'simulation': {
                'rounds': self.simulation.rounds,
                'protocol_name': self.simulation.protocol_name,
                'data_size': self.simulation.data_size,
                'stop_at_first_death': self.simulation.stop_at_first_death,
                'stop_at_half_death': self.simulation.stop_at_half_death,
                'checkpoint_interval': self.simulation.checkpoint_interval,
            },
            'energy': {
                'model_name': self.energy.model_name,
                'E_elec': self.energy.E_elec,
                'epsilon_fs': self.energy.epsilon_fs,
                'epsilon_mp': self.energy.epsilon_mp,
                'd_threshold': self.energy.d_threshold,
                'E_da': self.energy.E_da,
            },
            'ai': {
                'enabled': self.ai.enabled,
                'model_type': self.ai.model_type,
                'n_estimators': self.ai.n_estimators,
                'learning_rate': self.ai.learning_rate,
                'feature_normalization': self.ai.feature_normalization,
                'use_ensemble': self.ai.use_ensemble,
            },
            'visualization': {
                'enabled': self.visualization.enabled,
                'save_plots': self.visualization.save_plots,
                'save_animation': self.visualization.save_animation,
                'output_dir': self.visualization.output_dir,
                'animation_interval': self.visualization.animation_interval,
                'animation_fps': self.visualization.animation_fps,
                'figsize': list(self.visualization.figsize),
                'dpi': self.visualization.dpi,
            }
        }

    def save_yaml(self, path: str) -> None:
        """
        保存为 YAML 文件

        Args:
            path: 保存路径
        """
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save_json(self, path: str) -> None:
        """
        保存为 JSON 文件

        Args:
            path: 保存路径
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def validate_config_file(path: str) -> Tuple[bool, Optional[str]]:
    """
    验证配置文件

    Args:
        path: 配置文件路径

    Returns:
        (是否有效, 错误消息)
    """
    try:
        path_obj = Path(path)

        if not path_obj.exists():
            return False, f"File not found: {path}"

        if path_obj.suffix not in ['.yaml', '.yml', '.json']:
            return False, f"Unsupported file format: {path_obj.suffix}"

        if path_obj.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            with open(path, 'r') as f:
                data = json.load(f)

        if not isinstance(data, dict):
            return False, "Config must be a dictionary"

        FullConfig.from_dict(data)

        return True, None

    except yaml.YAMLError as e:
        return False, f"YAML parsing error: {e}"
    except json.JSONDecodeError as e:
        return False, f"JSON parsing error: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"
