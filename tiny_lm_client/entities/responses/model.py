from dataclasses import dataclass
from typing import Optional


@dataclass
class Model:
    """模型信息类，表示可用的AI模型元数据
    
    用于/models端点返回的数据结构，提供模型的基本信息和归属。
    
    Attributes:
        id: 模型标识符，如"gpt-4"、"claude-3-opus"等
        created: 模型发布时间戳(Unix epoch seconds)
        object: 对象类型标识，固定为"model"
        owned_by: 模型所有者标识，如"openai"、"anthropic"等
    
    Note:
        可通过models_list()方法获取所有可用模型的列表
        owned_by字段有助于识别模型的提供方和可能的使用限制
    """
    id: str
    created: int
    object: str = "model"
    owned_by: Optional[str] = None