from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ToolCall:
    """类型化的工具调用对象，表示模型决定调用的具体函数
    
    当模型根据对话内容决定调用某个工具时，会生成此对象。
    包含工具调用ID、类型和具体的函数调用参数。
    
    Attributes:
        id: 工具调用唯一标识符，用于匹配调用和结果
        type: 工具类型，通常为"function"
        function: 函数调用详情，包含函数名和参数(JSON字符串)
        extra: 扩展字段，保留API可能返回的未知参数
    """
    id: str
    type: str = "function"
    function: Dict[str, Any] = field(default_factory=dict)
    # 扩展字段，用于保留未知参数，确保API兼容性
    extra: Dict[str, Any] = field(default_factory=dict)