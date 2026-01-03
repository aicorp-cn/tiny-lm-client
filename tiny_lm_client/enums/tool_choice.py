from enum import Enum

class ToolChoiceType(str, Enum):
    """工具调用策略枚举，控制模型何时使用工具"""
    
    NONE = "none"     # 不调用任何工具
    AUTO = "auto"     # 模型自主决定是否调用工具
    REQUIRED = "required" # 必须调用指定工具