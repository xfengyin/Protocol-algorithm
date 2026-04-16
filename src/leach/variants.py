"""LEACH 变体注册表"""

from typing import Dict, Type

from .base import LEACHProtocol
from .classic import ClassicLEACH
from .leach_c import LEACHC
from .leach_ee import LEACHEE
from .leach_m import LEACHM


class LEACHRegistry:
    """LEACH 协议注册表"""
    
    _protocols: Dict[str, Type[LEACHProtocol]] = {
        "leach": ClassicLEACH,
        "classic": ClassicLEACH,
        "leach-c": LEACHC,
        "leach_c": LEACHC,
        "leach-ee": LEACHEE,
        "leach_ee": LEACHEE,
        "leach-m": LEACHM,
        "leach_m": LEACHM,
    }
    
    @classmethod
    def register(cls, name: str, protocol_class: Type[LEACHProtocol]):
        """注册新协议"""
        cls._protocols[name.lower()] = protocol_class
    
    @classmethod
    def get(cls, name: str) -> LEACHProtocol:
        """获取协议实例"""
        name_lower = name.lower()
        
        if name_lower not in cls._protocols:
            available = ", ".join(cls._protocols.keys())
            raise ValueError(f"Unknown protocol: {name}. Available: {available}")
        
        return cls._protocols[name_lower]()
    
    @classmethod
    def list_protocols(cls) -> list:
        """列出所有注册的协议"""
        return list(cls._protocols.keys())
