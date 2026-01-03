from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

from tiny_lm_client.enums.tool_choice import ToolChoiceType
from tiny_lm_client.enums.response_format import ResponseFormatType
from ..message import Message
from ..chat_message import ChatMessage
from ..tool import Tool
from ..response_format import ResponseFormat


@dataclass
class ChatCompletionRequest:
    """聊天补全请求配置类，封装/chat/completions端点的所有请求参数
    
    此类提供类型安全的请求参数管理，替代直接使用字典参数的方式。
    支持OpenAI API规范的所有聊天补全参数，包括工具调用、响应格式等新特性。
    
    Attributes:
        model: 模型标识符，如"gpt-4"、"claude-3"等
        messages: 对话历史消息列表，每个消息包含角色和内容
        temperature: 采样温度，控制随机性(0-2)，越高越随机
        max_tokens: 生成的最大token数，限制响应长度
        stream: 是否启用流式响应，True时返回异步生成器
        top_p: 核采样概率阈值(0-1)，替代temperature的另一种随机性控制
        frequency_penalty: 频率惩罚系数(-2~2)，减少重复token
        presence_penalty: 存在惩罚系数(-2~2)，鼓励新话题
        stop: 停止序列，遇到这些字符串时停止生成
        n: 为每个提示生成的完成数量
        seed: 随机种子，确保结果可重现
        tools: 可用工具列表，定义模型可调用的函数
        tool_choice: 工具调用策略，控制模型使用工具的时机
        response_format: 响应格式要求，如强制JSON输出
        user: 用户标识，用于监控和配额管理
        parallel_tool_calls: 是否允许并行调用多个工具
        logit_bias: token级别的logits修正，影响特定token的生成概率
        logprobs: 是否返回token的对数概率
        top_logprobs: 返回每个位置前N个最可能token的对数概率
        extra: 扩展字段，保留API可能新增的未知参数
    """
    model: str
    messages: List[Union[Message, ChatMessage, Dict[str, Any]]]
    temperature: float = 1.0
    max_tokens: Optional[int] = None
    stream: bool = False
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    n: int = 1
    seed: Optional[int] = None
    tools: Optional[List[Union[Tool, Dict[str, Any]]]] = None
    tool_choice: Optional[Union[ToolChoiceType, Dict[str, Any]]] = None
    response_format: Optional[Union[ResponseFormatType, ResponseFormat, Dict[str, Any]]] = None
    user: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None
    logit_bias: Optional[Dict[int, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    # 扩展字段，用于保留未知参数，确保API向前兼容
    extra: Dict[str, Any] = field(default_factory=dict)