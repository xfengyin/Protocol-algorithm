"""First Order Radio Model —— 向后兼容的导入路径。

实现已统一到 `src.energy.models.FirstOrderRadioModel`（EnergyModel 体系 + 工厂）。
这里保留同名导出，避免 engine / network / generator / tests 的既有 import 路径断掉，
同时消除「两个同名类、工厂产出的实例通不过 isinstance」的坑（2026-10-14 CI 实测）。
"""

from __future__ import annotations

from .models import FirstOrderRadioModel

__all__ = ["FirstOrderRadioModel"]
