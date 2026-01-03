from dataclasses import dataclass

from tiny_lm_client.enums.role import Role


@dataclass
class Message:
    """基础消息类，表示简单的文本消息
    
    适用于不需要高级特性的简单对话场景，只包含角色和基础文本内容。
    对于需要工具调用、多模态内容等高级特性的场景，应使用ChatMessage类。
    
    Attributes:
        role: 消息发送者角色(SYSTEM/USER/ASSISTANT)
        content: 消息文本内容
    """
    role: Role
    content: str