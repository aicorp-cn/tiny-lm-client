from enum import Enum

class FinishReason(str, Enum):
    """LLM生成停止原因枚举，对应OpenAI API规范"""
    
    STOP = "stop"           # 遇到停止标记或自然结束
    LENGTH = "length"       # 达到最大token长度限制
    TOOL_CALLS = "tool_calls" # 模型决定调用工具
    CONTENT_FILTER = "content_filter" # 内容被安全过滤器拦截
    FUNCTION_CALL = "function_call" # 模型决定调用函数（旧版）