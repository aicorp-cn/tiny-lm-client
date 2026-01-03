from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class CompletionChoice:
    """旧版文本补全选项，对应/v1/completions的单次生成结果
    
    与ChatCompletion的Choice不同，只包含纯文本生成结果，
    不包含角色信息和工具调用等复杂特性。
    
    Attributes:
        text: 生成的文本内容，纯字符串格式
        index: 选择项索引，当n>1时标识不同的生成结果
        logprobs: 可选的对数概率信息，包含top_logprobs个最可能的token及其概率
        finish_reason: 生成停止原因字符串(stop/length/content_filter等)
        extra: 扩展字段，保留API可能返回的未知参数
    
    Note:
        相比ChatCompletion的Choice更简单，适合传统文本生成场景
        finish_reason为字符串而非枚举，兼容性考虑
    """
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    # 扩展字段，用于保留未知参数，确保API兼容性
    extra: Dict[str, Any] = field(default_factory=dict)