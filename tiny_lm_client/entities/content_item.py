from dataclasses import dataclass, field
from typing import Dict, Any, Union

from tiny_lm_client.enums.content_type import ContentType


@dataclass
class ContentItem:
    """多模态内容项数据类，用于表示支持多种媒体类型的消息内容
    
    Attributes:
        type: 内容类型，决定如何解析content字段
        content: 实际内容，字符串(文本)或字典(结构化数据如图像URL)
        extra: 扩展字段，保留API可能返回的未知参数，确保前向兼容
    """
    type: ContentType
    content: Union[str, Dict[str, Any]]
    # 扩展字段，用于保留未知参数，确保API兼容性
    extra: Dict[str, Any] = field(default_factory=dict)