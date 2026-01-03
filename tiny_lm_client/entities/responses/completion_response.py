from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .completion_choice import CompletionChoice
from .. import Usage


@dataclass
class CompletionResponse:
    """旧版文本补全响应类，对应/v1/completions端点的完整响应
    
    包含传统文本生成API的完整响应数据，结构与ChatCompletion类似
    但使用简化的Choice格式，专注于纯文本生成场景。
    
    Attributes:
        id: 响应唯一标识符
        object: 对象类型标识，固定为"text_completion"
        created: 响应创建时间戳
        model: 实际处理请求的模型名称
        choices: 文本生成结果列表，每个包含纯文本内容
        usage: 令牌使用统计(仅prompt_tokens和total_tokens)
        extra: 扩展字段，保留API可能新增的响应字段
    
    Note:
        这是传统completion API的响应格式，建议迁移到chat completion
        主要用于代码补全、文本续写等简单生成任务
    """
    id: str
    object: str = "text_completion"
    created: int = 0
    model: str = ""
    choices: List[CompletionChoice] = field(default_factory=list)
    usage: Optional[Usage] = None
    # 扩展字段，用于保留未知参数，确保API向前兼容
    extra: Dict[str, Any] = field(default_factory=dict)