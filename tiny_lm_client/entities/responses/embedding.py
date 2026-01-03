from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Embedding:
    """单个文本嵌入向量数据类
    
    表示输入文本对应的高维向量表示。每个嵌入向量捕获了文本的语义信息。
    
    Attributes:
        index: 输入文本在请求中的索引位置，用于匹配输入输出
        vector: 嵌入向量数值列表，维度由模型决定(通常1536维)
        object: 对象类型标识，固定为"embedding"
        extra: 扩展字段，保留API可能返回的未知参数
    
    Note:
        向量维度因模型而异，使用时需注意模型兼容性
        index确保批量处理时能正确匹配输入和输出
    """
    index: int
    vector: List[float]  # 重命名为vector避免与类名混淆
    object: str = "embedding"
    # 扩展字段，用于保留未知参数，确保API兼容性
    extra: Dict[str, Any] = field(default_factory=dict)