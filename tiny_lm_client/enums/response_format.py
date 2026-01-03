from enum import Enum

class ResponseFormatType(str, Enum):
    """响应格式类型，控制模型输出格式"""
    
    TEXT = "text"             # 纯文本格式
    JSON_OBJECT = "json_object" # 强制JSON对象格式