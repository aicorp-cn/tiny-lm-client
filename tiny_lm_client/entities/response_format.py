from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from tiny_lm_client.enums.response_format import ResponseFormatType


@dataclass
class ResponseFormat:
    """响应格式规范类，强制模型按指定格式输出
    
    用于约束模型的输出格式，特别是需要结构化数据时使用JSON模式。
    遵循OpenAI Response Format规范，确保模型输出可解析的结构化数据。
    
    Attributes:
        type: 响应格式类型(TEXT/JSON_OBJECT)
        json_schema: JSON模式定义，指定输出JSON的具体结构和字段约束
    """
    type: ResponseFormatType
    json_schema: Optional[Dict[str, Any]] = None