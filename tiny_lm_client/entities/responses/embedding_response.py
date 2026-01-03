from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .embedding import Embedding
from .. import Usage


@dataclass
class EmbeddingResponse:
    """嵌入响应类，封装/embeddings端点的完整响应
    
    包含多个文本的嵌入向量和使用统计信息。
    
    Attributes:
        object: 对象类型标识，固定为"list"
        data: 嵌入向量列表，每个元素对应一个输入文本的嵌入
        model: 实际处理请求的嵌入模型名称
        usage: 令牌使用统计，仅包含prompt_tokens(无completion_tokens)
        extra: 扩展字段，保留API可能新增的响应字段
    
    Note:
        - data列表中元素的顺序与输入文本顺序一致
        - embedding主要用于语义检索，不直接涉及生成任务
        - usage只统计输入token，因为嵌入是无生成过程的编码操作
    """
    object: str = "list"
    data: List[Embedding] = field(default_factory=list)
    model: str = ""
    usage: Optional[Usage] = None
    # 扩展字段，用于保留未知参数，确保API向前兼容
    extra: Dict[str, Any] = field(default_factory=dict)