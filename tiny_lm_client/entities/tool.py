from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Tool:
    """工具定义类，描述模型可调用的函数工具
    
    遵循OpenAI Function Calling规范，定义函数的名称、描述和参数模式。
    用于enable模型在对话中识别和调用外部函数。
    
    Attributes:
        type: 工具类型，目前仅支持"function"
        function: 函数定义字典，包含name、description、parameters等
    """
    type: str = "function"
    function: Optional[Dict[str, Any]] = None