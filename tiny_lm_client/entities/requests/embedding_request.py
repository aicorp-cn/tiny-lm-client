from dataclasses import dataclass, field
from typing import Union, List, Optional, Dict, Any


@dataclass
class EmbeddingRequest:
    """文本嵌入请求配置类，封装/embeddings端点的请求参数
    
    用于将文本转换为高维向量表示，支持语义搜索、相似度计算等应用。
    支持单条和多条文本的批量嵌入。
    
    Attributes:
        model: 嵌入模型标识符，如"text-embedding-ada-002"
        input: 输入文本，可以是单个字符串或字符串列表
        encoding_format: 向量编码格式，如"float"、"base64"
        user: 用户标识，用于监控和配额管理
        extra: 扩展字段，保留API可能新增的未知参数
    
    Example:
        - 单文本: input="Hello world"
        - 多文本: input=["First text", "Second text"]
    """
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = None
    user: Optional[str] = None
    # 扩展字段，用于保留未知参数，确保API兼容性
    extra: Dict[str, Any] = field(default_factory=dict)