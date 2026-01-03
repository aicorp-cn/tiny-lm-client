from enum import Enum

class Role(str, Enum):
    """对话消息角色枚举，定义消息发送者的身份"""
    SYSTEM = "system"     # 系统指令，设定AI行为准则
    USER = "user"         # 用户输入
    ASSISTANT = "assistant" # AI助手回复
    TOOL = "tool"         # 工具执行结果消息
    FUNCTION = "function" # 函数调用结果（旧版兼容