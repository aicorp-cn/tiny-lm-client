from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from tiny_lm_client.enums.finish_reason import FinishReason
from .delta import Delta
from ..log_probs import LogProbs


@dataclass
class StreamChoice:
    """流式选择项类，表示ChatCompletion流式响应中的单个选择项的增量状态
    
    对应非流式响应中的Choice类，但只包含增量变化而非完整状态。
    在流式响应中，每个选择项的状态会随时间逐步更新。
    
    Attributes:
        index: 选择项索引，标识多个候选生成中的哪一个
        delta: 本次更新的增量内容，包含新增的文本、工具调用等
        finish_reason: 可选的完成原因，在流结束时出现
        logprobs: 可选的对数概率信息(流式模式下较少使用)
        extra: 扩展字段，保留API可能返回的未知参数
    
    Note:
        - delta字段可能只包含部分属性(如只有content，没有role)
        - 客户端需要累积所有delta才能获得完整消息
        - finish_reason只在最后一个块中出现
        - index确保在n>1时能正确组装多个候选流
    """
    index: int
    delta: Delta
    finish_reason: Optional[FinishReason] = None
    logprobs: Optional[LogProbs] = None
    # 扩展字段，用于保留未知参数，确保API兼容性
    extra: Dict[str, Any] = field(default_factory=dict)