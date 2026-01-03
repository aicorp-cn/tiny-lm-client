from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from tiny_lm_client.enums.finish_reason import FinishReason
from .chat_message import ChatMessage
from .log_probs import LogProbs

@dataclass
class Choice:
    """单次生成选择项，表示一个完整的模型回复
    
    当请求参数n>1时，响应会包含多个Choice，每个代表一个独立的生成结果。
    包含生成的消息内容、停止原因和可选的置信度信息。
    
    Attributes:
        index: 选择项索引(0到n-1)，标识多个生成结果中的序号
        message: 生成的消息内容，包含角色、文本、可能的工具调用等
        finish_reason: 生成停止的具体原因，解释为何在此处结束
        logprobs: 可选的对数概率信息，用于分析生成置信度
        extra: 扩展字段，保留API可能返回的未知参数
    
    Example:
        当n=2时，会得到两个Choice，index分别为0和1，
        分别对应第一次和第二次独立生成的结果
    """
    index: int
    message: ChatMessage
    finish_reason: FinishReason
    logprobs: Optional[LogProbs] = None
    # 扩展字段，用于保留未知参数，确保API兼容性
    extra: Dict[str, Any] = field(default_factory=dict)