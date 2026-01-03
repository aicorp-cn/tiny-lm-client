from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any

from tiny_lm_client.enums.role import Role
from ..tool_call import ToolCall


@dataclass
class Delta:
    """流式增量数据类，表示ChatCompletion流式响应中的单次增量更新
    
    在流式响应中，每个数据块只包含自上次更新以来的变化部分，
    而不是完整的消息状态。这减少了网络传输量并降低延迟。
    
    Attributes:
        role: 可选的角色信息，仅在消息开始时出现
        content: 新增的文本内容，流式响应的主要内容
        tool_calls: 新增的工具调用信息，当模型开始调用工具时出现
        refusal: 拒绝回复的原因，当内容被过滤时出现
        extra: 扩展字段，保留API可能返回的未知参数
    
    Stream Behavior:
        - 第一条消息块通常包含role字段
        - 后续块主要包含content增量
        - 工具调用会分多个块逐步返回
        - 最后可能有finish_reason字段
    
    Example:
        流式响应可能这样演进:
        chunk1: {"role": "assistant"}
        chunk2: {"content": "Hello"}
        chunk3: {"content": ", how"}
        chunk4: {"content": " are you?"}
        chunk5: {"finish_reason": "stop"}
    """
    role: Optional[Role] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Union['ToolCall', Dict[str, Any]]]] = None
    refusal: Optional[str] = None
    # 扩展字段，用于保留未知参数，确保API兼容性
    extra: Dict[str, Any] = field(default_factory=dict)