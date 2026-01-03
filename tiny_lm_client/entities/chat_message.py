from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any

from tiny_lm_client.enums.role import Role
from .tool_call import ToolCall


@dataclass
class ChatMessage:
    """增强消息类，支持工具调用、拒绝回复等多模态和交互特性
    
    这是核心消息类，用于复杂的对话场景。相比基础Message类，
    支持工具调用结果、函数调用、多模态内容和拒绝回复等新特性。
    
    Attributes:
        role: 消息发送者角色，包括TOOL和FUNCTION等扩展角色
        content: 消息文本内容，可为None(如纯工具调用消息)
        tool_calls: 此消息中模型发起的工具调用列表
        refusal: 模型拒绝回复的原因说明(安全或伦理限制)
        extra: 扩展字段，保留API可能返回的未知参数
    
    Note:
        - 当role为TOOL时，content应包含工具执行结果
        - 当model发起工具调用时，tool_calls字段包含调用详情
        - refusal字段用于处理内容过滤或安全限制情况
    """
    role: Role
    content: Optional[str] = None
    tool_calls: Optional[List[Union['ToolCall', Dict[str, Any]]]] = None
    refusal: Optional[str] = None
    # 扩展字段，用于保留未知参数，确保API向前兼容
    extra: Dict[str, Any] = field(default_factory=dict)