import httpx
import json
from typing import Dict, List, Optional, Union, AsyncGenerator, Any
from dataclasses import dataclass, field
from enum import Enum


class EncodingType(str, Enum):
    """HTTP响应内容编码类型，用于Accept-Encoding头部"""
    GZIP = "gzip"      # GZIP压缩，适用于文本数据压缩
    DEFLATE = "deflate" # DEFLATE压缩算法
    COMPRESS = "compress" # UNIX compress压缩


class FinishReason(str, Enum):
    """LLM生成停止原因枚举，对应OpenAI API规范"""
    STOP = "stop"           # 遇到停止标记或自然结束
    LENGTH = "length"       # 达到最大token长度限制
    TOOL_CALLS = "tool_calls" # 模型决定调用工具
    CONTENT_FILTER = "content_filter" # 内容被安全过滤器拦截
    FUNCTION_CALL = "function_call" # 模型决定调用函数（旧版）


class ToolChoiceType(str, Enum):
    """工具调用策略枚举，控制模型何时使用工具"""
    NONE = "none"     # 不调用任何工具
    AUTO = "auto"     # 模型自主决定是否调用工具
    REQUIRED = "required" # 必须调用指定工具


class ResponseFormatType(str, Enum):
    """响应格式类型，控制模型输出格式"""
    TEXT = "text"             # 纯文本格式
    JSON_OBJECT = "json_object" # 强制JSON对象格式


class Role(str, Enum):
    """对话消息角色枚举，定义消息发送者的身份"""
    SYSTEM = "system"     # 系统指令，设定AI行为准则
    USER = "user"         # 用户输入
    ASSISTANT = "assistant" # AI助手回复
    TOOL = "tool"         # 工具执行结果消息
    FUNCTION = "function" # 函数调用结果（旧版兼容）


class ContentType(str, Enum):
    """多模态内容类型枚举，支持文本、图像、音频等多种输入格式"""
    TEXT = "text"               # 文本内容
    IMAGE_URL = "image_url"     # 图片URL引用
    IMAGE_BASE64 = "image_base64" # Base64编码的图片数据
    AUDIO = "audio"             # 音频数据


@dataclass
class ContentItem:
    """多模态内容项数据类，用于表示支持多种媒体类型的消息内容
    
    Attributes:
        type: 内容类型，决定如何解析content字段
        content: 实际内容，字符串(文本)或字典(结构化数据如图像URL)
        extra: 扩展字段，保留API可能返回的未知参数，确保前向兼容
    """
    type: ContentType
    content: Union[str, Dict[str, Any]]
    # 扩展字段，用于保留未知参数，确保API兼容性
    extra: Dict[str, Any] = field(default_factory=dict)


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
    messages: List[Union['Message', 'ChatMessage', Dict[str, Any]]]
    temperature: float = 1.0
    max_tokens: Optional[int] = None
    stream: bool = False
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    n: int = 1
    seed: Optional[int] = None
    tools: Optional[List[Union['Tool', Dict[str, Any]]]] = None
    tool_choice: Optional[Union['ToolChoiceType', Dict[str, Any]]] = None
    response_format: Optional[Union['ResponseFormatType', 'ResponseFormat', Dict[str, Any]]] = None
    user: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None
    logit_bias: Optional[Dict[int, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    # 扩展字段，用于保留未知参数，确保API向前兼容
    extra: Dict[str, Any] = field(default_factory=dict)


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


@dataclass
class Tool:
    """工具定义类，描述模型可调用的函数工具
    
    遵循OpenAI Function Calling规范，定义函数的名称、描述和参数模式。
    用于enable模型在对话中识别和调用外部函数。
    
    Attributes:
        type: 工具类型，目前仅支持"function"
        function: 函数定义字典，包含name、description、parameters等
    """
    type: str = "function"
    function: Optional[Dict[str, Any]] = None


@dataclass
class ToolCall:
    """类型化的工具调用对象，表示模型决定调用的具体函数
    
    当模型根据对话内容决定调用某个工具时，会生成此对象。
    包含工具调用ID、类型和具体的函数调用参数。
    
    Attributes:
        id: 工具调用唯一标识符，用于匹配调用和结果
        type: 工具类型，通常为"function"
        function: 函数调用详情，包含函数名和参数(JSON字符串)
        extra: 扩展字段，保留API可能返回的未知参数
    """
    id: str
    type: str = "function"
    function: Dict[str, Any] = field(default_factory=dict)
    # 扩展字段，用于保留未知参数，确保API兼容性
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseFormat:
    """响应格式规范类，强制模型按指定格式输出
    
    用于约束模型的输出格式，特别是需要结构化数据时使用JSON模式。
    遵循OpenAI Response Format规范，确保模型输出可解析的结构化数据。
    
    Attributes:
        type: 响应格式类型(TEXT/JSON_OBJECT)
        json_schema: JSON模式定义，指定输出JSON的具体结构和字段约束
    """
    type: ResponseFormatType
    json_schema: Optional[Dict[str, Any]] = None


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


@dataclass
class LogProbs:
    """Token对数概率信息类，用于分析模型输出的置信度
    
    包含模型生成每个token时的对数概率值，用于调试、分析和置信度评估。
    仅在请求参数logprobs=True时返回。
    
    Attributes:
        content: Token概率列表，每个元素包含token、logprob、bytes等信息
                 格式: [{"token": "hello", "logprob": -0.5, "bytes": [104, 101, 108, 108, 111]}, ...]
    """
    content: Optional[List[Dict[str, Any]]] = None


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


@dataclass
class Usage:
    """令牌使用统计类，记录API调用的token消耗详情
    
    用于计费和性能监控，精确追踪输入、输出和总计的token数量。
    某些模型还提供更详细的分类统计，如提示缓存命中、推理加速等。
    
    Attributes:
        prompt_tokens: 输入提示消耗的token数量
        completion_tokens: 模型生成回复消耗的token数量
        total_tokens: 本次调用消耗的总token数量(prompt + completion)
        prompt_tokens_details: 提示token的详细分类，如{
            "cached_tokens": 0,      # 缓存命中的token数
            "audio_tokens": 0,       # 音频输入token数
            "text_tokens": 100       # 文本输入token数
        }
        completion_tokens_details: 生成token的详细分类，如{
            "reasoning_tokens": 0,   # 推理过程token数
            "accepted_prediction_tokens": 50,  # 接受的预测token数
            "rejected_prediction_tokens": 10   # 拒绝的预测token数
        }
        extra: 扩展字段，保留API可能新增的使用统计维度
    
    Note:
        不同模型提供商可能返回不同的details字段，通过extra保证兼容性
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[Dict[str, int]] = None
    completion_tokens_details: Optional[Dict[str, int]] = None
    # 扩展字段，用于保留未知参数，确保API向前兼容
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatCompletionResponse:
    """聊天补全响应类，封装/chat/completions端点的完整响应数据
    
    包含模型生成的回复内容、使用统计和系统信息等。
    支持流式和非流式两种响应模式的解析。
    
    Attributes:
        id: 响应唯一标识符，用于调试和日志记录
        object: 对象类型标识，固定为"chat.completion"
        created: 响应创建时间戳(Unix epoch seconds)
        model: 实际处理请求的模型名称(可能与请求model不同)
        choices: 生成结果列表，当n>1时包含多个独立回复
        usage: 令牌使用统计，用于计费和性能监控
        system_fingerprint: 系统指纹，标识后端配置版本
        service_tier: 服务等级，指示使用的计算资源级别
        extra: 扩展字段，保留API可能新增的响应字段
    
    Note:
        - 流式响应时，每个数据块也是ChatCompletionChunk类型
        - system_fingerprint可用于检测后端模型更新
        - service_tier反映响应速度和成本等级(如"default"、"premium")
    """
    id: str
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[Choice] = field(default_factory=list)
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None
    # 扩展字段，用于保留未知参数，确保API向前兼容
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingRequest:
    """文本嵌入请求配置类，封装/embeddings端点的请求参数
    
    用于将文本转换为高维向量表示，支持语义搜索、相似度计算等应用。
    支持单条和多条文本的批量嵌入。
    
    Attributes:
        model: 嵌入模型标识符，如"text-embedding-ada-002"
        input: 输入文本，可以是单个字符串或字符串列表
        encoding_format: 向量编码格式，如"float"、"base64"
        user: 用户标识，用于监控和配额管理
        extra: 扩展字段，保留API可能新增的未知参数
    
    Example:
        - 单文本: input="Hello world"
        - 多文本: input=["First text", "Second text"]
    """
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = None
    user: Optional[str] = None
    # 扩展字段，用于保留未知参数，确保API兼容性
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Embedding:
    """单个文本嵌入向量数据类
    
    表示输入文本对应的高维向量表示。每个嵌入向量捕获了文本的语义信息。
    
    Attributes:
        index: 输入文本在请求中的索引位置，用于匹配输入输出
        vector: 嵌入向量数值列表，维度由模型决定(通常1536维)
        object: 对象类型标识，固定为"embedding"
        extra: 扩展字段，保留API可能返回的未知参数
    
    Note:
        向量维度因模型而异，使用时需注意模型兼容性
        index确保批量处理时能正确匹配输入和输出
    """
    index: int
    vector: List[float]  # 重命名为vector避免与类名混淆
    object: str = "embedding"
    # 扩展字段，用于保留未知参数，确保API兼容性
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResponse:
    """嵌入响应类，封装/embeddings端点的完整响应
    
    包含多个文本的嵌入向量和使用统计信息。
    
    Attributes:
        object: 对象类型标识，固定为"list"
        data: 嵌入向量列表，每个元素对应一个输入文本的嵌入
        model: 实际处理请求的嵌入模型名称
        usage: 令牌使用统计，仅包含prompt_tokens(无completion_tokens)
        extra: 扩展字段，保留API可能新增的响应字段
    
    Note:
        - data列表中元素的顺序与输入文本顺序一致
        - embedding主要用于语义检索，不直接涉及生成任务
        - usage只统计输入token，因为嵌入是无生成过程的编码操作
    """
    object: str = "list"
    data: List[Embedding] = field(default_factory=list)
    model: str = ""
    usage: Optional[Usage] = None
    # 扩展字段，用于保留未知参数，确保API向前兼容
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionRequest:
    """旧版文本补全请求配置类，对应/v1/completions端点
    
    这是传统的文本生成API，适用于简单的文本续写、代码补全等场景。
    相比ChatCompletion，不支持多轮对话和工具调用，但支持更多传统生成参数。
    
    Attributes:
        model: 模型标识符，如"davinci"、"curie"等
        prompt: 输入提示，可以是字符串、字符串列表或token ID列表
        suffix: 追加到生成文本后的后缀内容
        max_tokens: 生成的最大token数
        temperature: 采样温度，控制随机性
        top_p: 核采样概率阈值
        n: 为每个提示生成的完成数量
        stream: 是否启用流式响应
        logprobs: 返回每个token的对数概率(top_logprobs个最可能的token)
        echo: 是否在响应中包含输入prompt
        stop: 停止序列数组
        presence_penalty: 存在惩罚系数
        frequency_penalty: 频率惩罚系数
        best_of: 从多个候选中选择最佳的生成数量(服务端)
        logit_bias: token级别的logits修正
        user: 用户标识
        extra: 扩展字段，保留API可能新增的未知参数
    
    Note:
        这是OpenAI兼容的旧版API，建议新项目使用ChatCompletion
        best_of与n参数配合使用时，服务端会生成best_of*n个候选
    """
    model: str
    prompt: Union[str, List[str], List[int], List[List[int]]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: bool = False
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    best_of: Optional[int] = None
    logit_bias: Optional[Dict[int, float]] = None
    user: Optional[str] = None
    # 扩展字段，用于保留未知参数，确保API兼容性
    extra: Dict[str, Any] = field(default_factory=dict)


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


@dataclass
class CompletionChunk:
    """旧版文本补全流式响应块，对应/v1/completions的流式数据块
    
    在stream=True模式下，服务端会分多次发送这种数据块，
    每个块包含增量生成的文本内容。客户端需要拼接所有块获得完整结果。
    
    Attributes:
        id: 响应唯一标识符(与完整响应相同)
        object: 对象类型标识，固定为"text_completion"
        created: 响应创建时间戳
        model: 实际处理请求的模型名称
        choices: 流式生成结果列表，每个包含增量文本内容
        extra: 扩展字段，保留API可能新增的响应字段
    
    Stream Format:
        数据流格式: data: {JSON_CHUNK}\n\n
        data: [DONE]\n\n 表示流结束
    
    Note:
        流式响应显著降低感知延迟，适合实时显示生成内容的场景
        每个chunk的choices只包含增量变化，不是完整状态
    """
    id: str
    object: str = "text_completion"
    created: int = 0
    model: str = ""
    choices: List[CompletionChoice] = field(default_factory=list)
    # 扩展字段，用于保留未知参数，确保API兼容性
    extra: Dict[str, Any] = field(default_factory=dict)


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


@dataclass
class ChatCompletionChunk:
    """聊天补全流式响应块类，对应/chat/completions的流式数据块
    
    在stream=True模式下，服务端分多次发送这种数据块，每个块包含
    一个或多个选择项的增量更新。客户端需要累积所有块来重建完整响应。
    
    Attributes:
        id: 响应唯一标识符(整个流式会话保持不变)
        object: 对象类型标识，固定为"chat.completion.chunk"
        created: 响应创建时间戳
        model: 实际处理请求的模型名称
        choices: 流式选择项列表，每个包含增量更新
        system_fingerprint: 系统指纹，标识后端配置版本
        service_tier: 服务等级信息
        extra: 扩展字段，保留API可能新增的响应字段
    
    Stream Format:
        数据流格式: data: {JSON_CHUNK}\n\n
        data: [DONE]\n\n 表示流结束
    
    Note:
        这是现代chat completion API的流式格式，比旧版completion更丰富
        支持工具调用的增量返回和多角色消息的流式构建
    """
    id: str
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: List[StreamChoice] = field(default_factory=list)
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None
    # 扩展字段，用于保留未知参数，确保API向前兼容
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Model:
    """模型信息类，表示可用的AI模型元数据
    
    用于/models端点返回的数据结构，提供模型的基本信息和归属。
    
    Attributes:
        id: 模型标识符，如"gpt-4"、"claude-3-opus"等
        created: 模型发布时间戳(Unix epoch seconds)
        object: 对象类型标识，固定为"model"
        owned_by: 模型所有者标识，如"openai"、"anthropic"等
    
    Note:
        可通过models_list()方法获取所有可用模型的列表
        owned_by字段有助于识别模型的提供方和可能的使用限制
    """
    id: str
    created: int
    object: str = "model"
    owned_by: Optional[str] = None


class OpenAIError(Exception):
    """OpenAI API错误异常类，封装API调用过程中的各种错误
    
    继承自Python标准Exception类，添加了OpenAI API特定的错误属性，
    便于错误处理和调试。支持结构化错误信息解析。
    
    Attributes:
        message: 人类可读的错误描述信息
        type: 错误类型，如"invalid_request_error"、"authentication_error"等
        code: 错误代码，具体的错误标识符
        param: 导致错误的参数名或相关信息
    
    Error Types:
        - invalid_request_error: 请求参数错误或缺失
        - authentication_error: API密钥无效或缺失
        - permission_error: 权限不足
        - rate_limit_error: 超出速率限制
        - server_error: 服务器内部错误
    
    Example:
        try:
            response = await client.chat_completion(request)
        except OpenAIError as e:
            print(f"Error: {e.message}, Type: {e.type}, Code: {e.code}")
    """
    def __init__(self, message: str, type: str = None, code: str = None, param: Any = None):
        self.message = message
        self.type = type
        self.code = code
        self.param = param
        super().__init__(message)


class TinyLMClient:
    """轻量级OpenAI兼容的大模型客户端类库
    
    基于httpx异步HTTP客户端构建，提供类型安全、功能完整的AI模型访问接口。
    支持现代OpenAI API规范的所有核心功能，包括聊天补全、文本嵌入、工具调用等。
    
    Key Features:
        - 完全兼容OpenAI API规范，支持GPT、Claude、本地模型等
        - 异步IO设计，高性能并发处理
        - 类型安全的数据类封装，编译期类型检查
        - 流式和非流式响应支持
        - 自动重试和指数退避机制
        - 内容编码支持(gzip/deflate/compress)
        - 扩展字段保留，确保API前向兼容
        - 统一的错误处理和异常体系
    
    Architecture:
        - 配置类驱动: 使用dataclass封装所有请求参数
        - 分层解析: 响应数据逐层解析为类型安全对象
        - 流式抽象: 统一流式和非流式接口设计
        - 错误恢复: 智能重试机制处理 transient failures
    
    Usage:
        async with TinyLMClient(base_url="https://api.openai.com/v1", 
                               api_key="your-key") as client:
            response = await client.chat_completion(request)
    
    Thread Safety:
        实例方法非线程安全，但每个异步上下文可安全使用
        建议在async with语句中使用以确保资源正确释放
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        encoding: Optional[EncodingType] = None,
        max_retries: int = 0,
        timeout: float = 60.0
    ):
        """初始化TinyLMClient实例
        
        设置客户端的基础配置，包括API端点、认证信息和HTTP客户端参数。
        支持自定义内容编码和重试策略，适应不同的部署环境。
        
        Args:
            base_url: API服务的基础URL，如"https://api.openai.com/v1"
                    支持本地部署的模型服务，如"http://localhost:8000/v1"
            api_key: API认证密钥，用于Bearer token认证
                    某些本地服务可能接受空字符串或dummy值
            encoding: HTTP响应内容编码类型，优化网络传输效率
                      None表示不指定，让服务器决定最佳编码
            max_retries: 最大重试次数，处理瞬时网络故障和速率限制
                        设置为0禁用重试，建议生产环境至少设置为3
            timeout: HTTP请求超时时间(秒)，防止长时间阻塞
                   复杂任务可适当增加，简单查询可减小以提高响应性
        
        Raises:
            ValueError: 当base_url为空或api_key为None时抛出
        
        Example:
            # OpenAI云服务
            client = TinyLMClient(
                base_url="https://api.openai.com/v1",
                api_key="sk-...",
                max_retries=3,
                timeout=30.0
            )
            
            # 本地模型服务
            client = TinyLMClient(
                base_url="http://localhost:8000/v1",
                api_key="",
                max_retries=1
            )
        """
        if not base_url:
            raise ValueError("base_url cannot be empty")
        if api_key is None:
            raise ValueError("api_key cannot be None")
            
        self.base_url = base_url.rstrip('/')  # 移除末尾斜杠，避免路径重复
        self.api_key = api_key
        self.encoding = encoding
        self.max_retries = max(max_retries, 0)  # 确保非负
        self.timeout = timeout
        
        # 构建HTTP请求头，遵循OpenAI API规范
        self.headers = {
            'Accept': 'application/json',           # 期望JSON响应
            "Content-Type": "application/json",   # 请求体为JSON格式
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            "Authorization": f"Bearer {self.api_key}",  # Bearer token认证
        }
        
        # 条件性添加内容编码头，优化传输效率
        if encoding:
            self.headers['Accept-Encoding'] = encoding.value
            
        # 初始化异步HTTP客户端，支持连接池和keep-alive
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=100,      # 最大并发连接数
                max_keepalive_connections=20  # 保持的连接池大小
            )
        )

    
    def _build_chat_completion_request(self, **kwargs) -> ChatCompletionRequest:
        """构建ChatCompletionRequest对象的工厂方法
        
        将分散的关键字参数封装为类型安全的ChatCompletionRequest对象。
        此方法作为completion方法的内部适配器，简化参数传递并避免参数爆炸。
        
        Args:
            **kwargs: 聊天补全请求的所有参数，对应ChatCompletionRequest的属性
                    包括model、messages、temperature等所有配置项
        
        Returns:
            ChatCompletionRequest: 类型安全的请求配置对象
        
        Note:
            - 自动清理extra字段，避免意外的额外参数污染
            - 保持与completion方法的向后兼容性
            - 集中参数验证逻辑，便于统一维护
        
        Example:
            request = self._build_chat_completion_request(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.7,
                stream=True
            )
        """
        # 过滤掉None值的extra参数，确保请求纯净性
        kwargs_clean = {k: v for k, v in kwargs.items() if not (k == "extra" and not v)}
        return ChatCompletionRequest(**{**kwargs_clean, "extra": kwargs.get("extra", {})})

    async def chat_completion(self, request: ChatCompletionRequest) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionChunk, None]]:
        """执行聊天补全请求，支持流式和非流式两种响应模式
        
        这是核心业务方法，处理所有聊天补全请求。根据stream参数决定返回完整响应
        还是流式响应生成器。内部自动处理请求构建、发送和响应解析。
        
        Args:
            request: ChatCompletionRequest配置对象，包含所有请求参数
                   推荐使用ChatCompletionRequest类创建，确保类型安全
        
        Returns:
            非流式模式: ChatCompletionResponse - 完整的响应对象
            流式模式: AsyncGenerator[ChatCompletionChunk, None] - 异步生成器
                      逐个产生响应块，适合实时显示
        
        Raises:
            OpenAIError: API调用失败时抛出，包含错误详情
            httpx.RequestError: 网络连接故障时抛出
            ValidationError: 请求参数验证失败时抛出
        
        Performance:
            - 流式模式显著降低感知延迟，首字节时间(TTFB)更短
            - 非流式模式适合需要完整响应的批处理场景
            - 大响应建议使用流式模式避免内存占用过高
        
        Example:
            # 非流式调用
            response = await client.chat_completion(request)
            print(response.choices[0].message.content)
            
            # 流式调用
            async for chunk in await client.chat_completion(stream_request):
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
        
        Note:
            推荐优先使用chat_completion而非completion方法，
            前者提供更好的类型安全和可维护性
        """
        # 构建符合API规范的请求数据字典
        request_data = self._build_chat_completion_request_data(request)
        
        # 根据stream参数选择响应模式
        if request.stream:
            # 返回流式响应生成器，支持异步迭代
            return self._stream_chat_completion(request_data)
        else:
            # 返回完整的响应对象
            return await self._non_stream_chat_completion(request_data)
    
    def _build_chat_completion_request_data(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """将ChatCompletionRequest对象转换为API所需的字典格式
        
        负责将类型安全的配置对象序列化为HTTP请求体。处理所有字段的转换逻辑，
        包括复杂枚举类型、嵌套对象和扩展字段的处理。
        
        Args:
            request: ChatCompletionRequest配置对象，包含完整的请求参数
        
        Returns:
            Dict[str, Any]: 符合OpenAI API规范的请求字典，可直接序列化为JSON
        
        Process Flow:
            1. 基础字段映射: model、messages、temperature等必需字段
            2. 可选字段处理: 使用_add_optional_field避免None值污染
            3. 复杂字段转换: tools、tool_choice、response_format的枚举处理
            4. 扩展字段合并: extra字段无条件合并，确保前向兼容
        
        Note:
            - 枚举值自动转换为字符串，避免序列化错误
            - 复杂对象(如Tool、ToolChoice)智能转换为字典格式
            - extra字段保持原样合并，支持API扩展功能
            - 此方法确保请求格式严格符合OpenAI API规范
        
        Example:
            request = ChatCompletionRequest(
                model="gpt-4",
                messages=[...],
                tools=[Tool(type="function", function={...})]
            )
            data = self._build_chat_completion_request_data(request)
            # data["tools"] 自动转换为 [{"type": "function", "function": {...}}]
        """
        # 构建基础请求数据，包含所有必需字段
        request_data = {
            "model": request.model,
            "messages": self._format_messages(request.messages),
            "temperature": request.temperature,
            "stream": request.stream,
            "n": request.n,
        }
        
        # 批量添加可选字段，使用统一方法避免重复代码
        optional_fields = [
            ("max_tokens", request.max_tokens),
            ("top_p", request.top_p),
            ("frequency_penalty", request.frequency_penalty),
            ("presence_penalty", request.presence_penalty),
            ("stop", request.stop),
            ("seed", request.seed),
            ("user", request.user),
            ("parallel_tool_calls", request.parallel_tool_calls),
            ("logit_bias", request.logit_bias),
            ("logprobs", request.logprobs),
            ("top_logprobs", request.top_logprobs),
        ]
        
        for field_name, field_value in optional_fields:
            self._add_optional_field(request_data, field_name, field_value)
        
        # 处理复杂字段：工具定义列表
        # 将Tool对象智能转换为API所需的字典格式
        if request.tools is not None:
            request_data["tools"] = [
                t if isinstance(t, dict) else {"type": t.type, "function": t.function}
                for t in request.tools
            ]
        
        # 处理工具调用策略：枚举值转字符串
        if request.tool_choice is not None:
            request_data["tool_choice"] = (
                request.tool_choice.value 
                if isinstance(request.tool_choice, ToolChoiceType) 
                else request.tool_choice
            )
        
        # 处理响应格式：枚举值转字符串
        if request.response_format is not None:
            request_data["response_format"] = (
                request.response_format.value 
                if isinstance(request.response_format, ResponseFormatType) 
                else request.response_format
            )
        
        # 合并扩展字段，确保API兼容性
        if request.extra:
            request_data.update(request.extra)
            
        return request_data
    
    def _add_optional_field(self, data: Dict[str, Any], key: str, value: Any):
        """条件性地向请求数据字典添加可选字段
        
        统一处理可选字段的添加逻辑，避免请求数据中包含None值字段。
        符合OpenAI API规范：省略None值字段而非显式传递null。
        
        Args:
            data: 目标请求数据字典，将被原地修改
            key: 字段名，将成为字典的键
            value: 字段值，只有当不为None时才添加到字典
        
        Note:
            - 值为None时静默跳过，不修改原字典
            - 支持任意类型的value，由调用方确保类型正确性
            - 此方法被多个请求构建方法复用，确保一致性
        
        Example:
            data = {"model": "gpt-4"}
            self._add_optional_field(data, "max_tokens", 100)  # 添加字段
            self._add_optional_field(data, "stop", None)       # 跳过添加
            # data => {"model": "gpt-4", "max_tokens": 100}
        """
        # 只有值非None时才添加到请求数据，避免API接收到null值
        if value is not None:
            data[key] = value

    def _format_messages(self, messages: List[Union[Message, ChatMessage, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """将混合类型的消息列表转换为API标准的字典格式
        
        统一处理不同类型的消息输入：Message对象、ChatMessage对象、原始字典。
        确保输出格式严格符合OpenAI API规范，支持工具调用等高级特性。
        
        Args:
            messages: 混合类型的消息列表，可包含Message、ChatMessage对象或预格式化字典
        
        Returns:
            List[Dict[str, Any]]: API标准的消息字典列表，每个字典包含role和content等字段
        
        Process:
            - 字典消息: 直接透传，假设已符合API格式
            - 对象消息: 调用_convert_single_message进行标准化转换
            - 自动处理tool_calls、refusal等高级字段
        
        Note:
            此方法确保无论输入类型如何，输出都符合API规范
            是消息预处理的关键步骤，被chat_completion等方法依赖
        """
        formatted_messages = []
        
        for msg in messages:
            if isinstance(msg, dict):
                # 已是字典格式，直接添加(假设调用方确保格式正确)
                formatted_messages.append(msg)
                continue
                
            # 对象类型消息，需要转换为字典格式
            msg_dict = self._convert_single_message(msg)
            formatted_messages.append(msg_dict)
            
        return formatted_messages
    
    def _convert_single_message(self, msg: Union[Message, ChatMessage]) -> Dict[str, Any]:
        """将单个消息对象转换为API标准的字典表示
        
        提取消息对象的核心属性和扩展字段，构建符合API规范的字典结构。
        处理不同类型消息对象的特殊字段，如ChatMessage的tool_calls和refusal。
        
        Args:
            msg: Message或ChatMessage对象实例
        
        Returns:
            Dict[str, Any]: API标准的消息字典，包含role、content等必要字段
        
        Field Mapping:
            - role: 消息发送者角色，枚举值自动转换为字符串
            - content: 消息文本内容，None值会被跳过
            - tool_calls: ChatMessage特有，工具调用详情列表
            - refusal: ChatMessage特有，拒绝回复原因
            - extra: 所有类型的扩展字段，无条件合并
        """
        # 基础消息字段：角色转换
        role_value = msg.role.value if isinstance(msg.role, Role) else msg.role
        msg_dict = {"role": role_value}
        
        # 添加content字段：仅当有实际内容时添加
        if hasattr(msg, 'content') and msg.content is not None:
            msg_dict["content"] = msg.content
            
        # 处理ChatMessage特有字段：工具调用和拒绝回复
        if isinstance(msg, ChatMessage):
            self._add_chat_message_fields(msg, msg_dict)
            
        # 添加extra扩展字段：保持API前向兼容性
        self._add_extra_fields(msg, msg_dict)
        
        return msg_dict
    
    def _add_chat_message_fields(self, msg: ChatMessage, msg_dict: Dict[str, Any]):
        """为ChatMessage对象添加特有字段到字典表示
        
        处理ChatMessage的高级特性字段，包括工具调用列表和拒绝回复。
        这些字段是基础Message类不具备的功能。
        
        Args:
            msg: ChatMessage对象实例
            msg_dict: 目标消息字典，将被原地修改添加特有字段
        """
        # 处理tool_calls：工具调用详情，模型发起的函数调用
        if msg.tool_calls is not None:
            msg_dict["tool_calls"] = self._format_tool_calls(msg.tool_calls)
            
        # 处理refusal：拒绝回复原因，内容过滤或安全限制触发
        if msg.refusal is not None:
            msg_dict["refusal"] = msg.refusal
            
    def _format_tool_calls(self, tool_calls: List[Union[ToolCall, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """将工具调用对象列表格式化为API标准字典格式
        
        统一处理ToolCall对象和预格式化字典，确保所有工具调用都符合API规范。
        保留tool_call的extra字段，支持API扩展功能。
        
        Args:
            tool_calls: ToolCall对象列表或字典列表
        
        Returns:
            List[Dict[str, Any]]: API标准的工具调用字典列表
        
        Note:
            - ToolCall对象自动转换为{"id", "type", "function"}结构
            - 预格式化字典直接透传，假设格式正确
            - extra字段无条件合并，确保兼容性
        """
        tool_calls_list = []
        for tc in tool_calls:
            if isinstance(tc, ToolCall):
                # ToolCall对象转换为标准字典格式
                tc_dict = {
                    "id": tc.id,           # 调用唯一标识符
                    "type": tc.type,       # 工具类型，通常为"function"
                    "function": tc.function  # 函数调用详情，包含name和arguments
                }
                # 合并extra字段：保留API可能新增的未知参数
                if tc.extra:
                    tc_dict.update(tc.extra)
                tool_calls_list.append(tc_dict)
            else:
                # 已是字典格式，直接添加
                tool_calls_list.append(tc)
        return tool_calls_list
    
    def _add_extra_fields(self, msg: Union[Message, ChatMessage], msg_dict: Dict[str, Any]):
        """将消息对象的extra扩展字段合并到字典表示
        
        无条件合并extra字段，确保API的前向兼容性。这些字段可能包含
        当前版本未知的API参数或自定义扩展。
        
        Args:
            msg: Message或ChatMessage对象实例
            msg_dict: 目标消息字典，extra字段将被合并到此字典
        """
        # 检查并合并extra字段：存在且非空时执行合并
        if hasattr(msg, 'extra') and msg.extra:
            msg_dict.update(msg.extra)

    async def _non_stream_chat_completion(self, request_data: Dict[str, Any]) -> ChatCompletionResponse:
        """非流式聊天补全"""
        response = await self._request_with_retry("POST", "/chat/completions", json=request_data)
        data = response.json()
        return self._parse_chat_completion_response(data)

    def _parse_chat_completion_response(self, data: Dict[str, Any]) -> ChatCompletionResponse:
        """解析聊天补全响应"""
        response_extra = self._extract_extra_fields(data, {"id", "object", "created", "model", "choices", "usage", "system_fingerprint", "service_tier"})
        
        choices = [self._parse_choice(choice_data) for choice_data in data.get("choices", [])]
        usage = self._parse_usage(data.get("usage"))
        
        return ChatCompletionResponse(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
            system_fingerprint=data.get("system_fingerprint"),
            service_tier=data.get("service_tier"),
            extra=response_extra
        )
    
    def _extract_extra_fields(self, data: Dict[str, Any], known_fields: set) -> Dict[str, Any]:
        """提取已知字段外的额外字段"""
        return {k: v for k, v in data.items() if k not in known_fields}
    
    def _parse_choice(self, choice_data: Dict[str, Any]) -> Choice:
        """解析单个选择项"""
        known_choice_fields = {"index", "message", "finish_reason", "logprobs"}
        choice_extra = self._extract_extra_fields(choice_data, known_choice_fields)
        
        message_data = choice_data.get("message", {})
        message = self._parse_chat_message(message_data)
        logprobs = self._parse_logprobs(choice_data.get("logprobs"))
        
        return Choice(
            index=choice_data.get("index", 0),
            message=message,
            finish_reason=FinishReason(choice_data.get("finish_reason", "stop")),
            logprobs=logprobs,
            extra=choice_extra
        )
    
    def _parse_chat_message(self, message_data: Dict[str, Any]) -> ChatMessage:
        """解析聊天消息"""
        known_message_fields = {"role", "content", "tool_calls", "refusal"}
        message_extra = self._extract_extra_fields(message_data, known_message_fields)
        
        tool_calls = self._parse_tool_calls(message_data.get("tool_calls"))
        
        return ChatMessage(
            role=Role(message_data.get("role", "")),
            content=message_data.get("content"),
            tool_calls=tool_calls,
            refusal=message_data.get("refusal"),
            extra=message_extra
        )
    
    def _parse_tool_calls(self, tool_calls_data: Optional[List[Dict[str, Any]]]) -> Optional[List[ToolCall]]:
        """解析工具调用列表"""
        if not tool_calls_data:
            return None
            
        tool_calls = []
        for tc_data in tool_calls_data:
            known_tc_fields = {"id", "type", "function"}
            tc_extra = self._extract_extra_fields(tc_data, known_tc_fields)
            
            tool_call = ToolCall(
                id=tc_data.get("id", ""),
                type=tc_data.get("type", "function"),
                function=tc_data.get("function", {}),
                extra=tc_extra
            )
            tool_calls.append(tool_call)
        return tool_calls
    
    def _parse_logprobs(self, logprobs_data: Optional[Dict[str, Any]]) -> Optional[LogProbs]:
        """解析日志概率"""
        if not logprobs_data:
            return None
        return LogProbs(content=logprobs_data.get("content"))
    
    def _parse_usage(self, usage_data: Optional[Dict[str, Any]]) -> Optional[Usage]:
        """解析使用统计"""
        if not usage_data:
            return None
            
        known_usage_fields = {"prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details", "completion_tokens_details"}
        usage_extra = self._extract_extra_fields(usage_data, known_usage_fields)
        
        return Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            prompt_tokens_details=usage_data.get("prompt_tokens_details"),
            completion_tokens_details=usage_data.get("completion_tokens_details"),
            extra=usage_extra
        )

    async def _stream_chat_completion(self, request_data: Dict[str, Any]) -> AsyncGenerator[ChatCompletionChunk, None]:
        """流式聊天补全"""
        async with self.client.stream("POST", "/chat/completions", json=request_data) as response:
            self._check_error(response)
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        yield self._parse_chat_completion_chunk(data)
                    except json.JSONDecodeError:
                        continue

    def _parse_chat_completion_chunk(self, data: Dict[str, Any]) -> ChatCompletionChunk:
        """解析流式聊天补全块"""
        known_chunk_fields = {"id", "object", "created", "model", "choices", "system_fingerprint", "service_tier"}
        chunk_extra = self._extract_extra_fields(data, known_chunk_fields)
        
        choices = [self._parse_stream_choice(choice_data) for choice_data in data.get("choices", [])]
        
        return ChatCompletionChunk(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion.chunk"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            system_fingerprint=data.get("system_fingerprint"),
            service_tier=data.get("service_tier"),
            extra=chunk_extra
        )
    
    def _parse_stream_choice(self, choice_data: Dict[str, Any]) -> StreamChoice:
        """解析流式选择项"""
        known_choice_fields = {"index", "delta", "finish_reason", "logprobs"}
        choice_extra = self._extract_extra_fields(choice_data, known_choice_fields)
        
        delta_data = choice_data.get("delta", {})
        delta = self._parse_delta(delta_data)
        logprobs = self._parse_logprobs(choice_data.get("logprobs"))
        finish_reason = FinishReason(choice_data["finish_reason"]) if choice_data.get("finish_reason") else None
        
        return StreamChoice(
            index=choice_data.get("index", 0),
            delta=delta,
            finish_reason=finish_reason,
            logprobs=logprobs,
            extra=choice_extra
        )
    
    def _parse_delta(self, delta_data: Dict[str, Any]) -> Delta:
        """解析增量数据"""
        known_delta_fields = {"role", "content", "tool_calls", "refusal"}
        delta_extra = self._extract_extra_fields(delta_data, known_delta_fields)
        
        tool_calls = self._parse_tool_calls(delta_data.get("tool_calls"))
        
        return Delta(
            role=Role(delta_data["role"]) if delta_data.get("role") else None,
            content=delta_data.get("content"),
            tool_calls=tool_calls,
            refusal=delta_data.get("refusal"),
            extra=delta_extra
        )

    async def models_list(self) -> List[Model]:
        """获取API服务支持的可用模型列表，对应/models端点
        
        查询后端服务当前可用的所有AI模型，包括模型标识符、发布时间、所有者等信息。
        用于动态发现服务能力和选择合适的模型进行推理。
        
        Returns:
            List[Model]: 模型信息列表，每个Model对象包含模型的元数据
                       包括id、created、owned_by等字段，按API返回顺序排列
        
        Raises:
            OpenAIError: API调用失败时抛出，包含错误详情
            httpx.RequestError: 网络连接故障时抛出
        
        Usage:
            - 启动时获取可用模型列表，构建模型选择菜单
            - 运行时检查特定模型是否可用
            - 监控模型服务的更新和变更
        
        Example:
            # 获取所有可用模型
            models = await client.models_list()
            
            # 打印模型信息
            for model in models:
                print(f"Model: {model.id}")
                print(f"  Created: {model.created}")
                print(f"  Owner: {model.owned_by}")
                print()
            
            # 查找特定模型
            gpt_models = [m for m in models if "gpt" in m.id.lower()]
            if gpt_models:
                selected_model = gpt_models[0].id
                print(f"Using model: {selected_model}")
            
            # 检查模型是否可用
            available_models = {m.id for m in models}
            if "gpt-4" in available_models:
                print("GPT-4 is available")
            else:
                print("GPT-4 is not available, using fallback model")
        
        Note:
            - 不同服务商返回的模型列表结构可能略有差异
            - owned_by字段有助于识别模型提供方(如"openai"、"anthropic")
            - created时间戳可用于判断模型新旧程度
            - 某些本地服务可能返回简化的模型列表
        """
        # 发送GET请求到/models端点，获取模型列表
        response = await self._request_with_retry("GET", "/models")
        data = response.json()
        
        # 解析响应数据，构建Model对象列表
        models = []
        for model_data in data.get("data", []):  # data字段包含模型数组
            model = Model(
                id=model_data.get("id", ""),           # 模型标识符，如"gpt-4"、"text-embedding-ada-002"
                created=model_data.get("created", 0),   # 模型发布时间戳
                object=model_data.get("object", "model"), # 对象类型，固定为"model"
                owned_by=model_data.get("owned_by")    # 模型所有者，如"openai"、"anthropic"
            )
            models.append(model)
        
        return models

    async def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """执行文本嵌入计算请求，对应/embeddings端点
        
        将文本转换为高维向量表示，支持语义搜索、相似度计算、聚类分析等应用。
        嵌入向量捕获文本的语义信息，可用于机器学习、信息检索等下游任务。
        
        Args:
            request: EmbeddingRequest配置对象，包含嵌入计算的所有参数
                   支持单条和多条文本的批量嵌入计算
        
        Returns:
            EmbeddingResponse: 嵌入响应对象，包含多个文本的嵌入向量和使用统计
                           data字段按输入顺序包含对应的嵌入向量
        
        Raises:
            OpenAIError: API调用失败时抛出，包含错误详情
            httpx.RequestError: 网络连接故障时抛出
        
        Performance:
            - 批量嵌入显著提高吞吐量，建议一次处理多条文本
            - 向量维度因模型而异(通常1536维)，使用时需注意兼容性
            - 嵌入计算无生成过程，响应速度通常较快
        
        Applications:
            - 语义搜索: 将文档和查询都嵌入，计算余弦相似度
            - 文本聚类: 基于嵌入向量进行K-means等聚类算法
            - 相似度匹配: 检测重复内容、推荐相关内容
            - 特征工程: 作为机器学习模型的特征输入
        
        Example:
            # 单文本嵌入
            request = EmbeddingRequest(
                model="text-embedding-ada-002",
                input="Hello world"
            )
            response = await client.embeddings(request)
            vector = response.data[0].vector  # 1536维向量
            
            # 批量嵌入 - 推荐用法
            request = EmbeddingRequest(
                model="text-embedding-ada-002",
                input=["First document", "Second document", "Third document"],
                encoding_format="float"
            )
            response = await client.embeddings(request)
            # response.data[0].vector 对应 "First document" 的嵌入
            # response.data[1].vector 对应 "Second document" 的嵌入
            
            # 计算两个文本的相似度
            import numpy as np
            vec1 = np.array(response.data[0].vector)
            vec2 = np.array(response.data[1].vector)
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        Note:
            - 输入文本长度受模型限制，超长文本需分段处理
            - 批量处理时注意内存使用，大量文本可分批次进行
            - embedding主要用于语义分析，不涉及文本生成任务
        """
        # 构建请求数据，包含必需的model和input字段
        request_data = {
            "model": request.model,              # 嵌入模型标识符，如"text-embedding-ada-002"
            "input": request.input,              # 输入文本，字符串或字符串列表
        }
        
        # 添加可选字段：编码格式和用户标识
        if request.encoding_format is not None:
            request_data["encoding_format"] = request.encoding_format  # 如"float"、"base64"
        if request.user is not None:
            request_data["user"] = request.user                      # 用户标识用于配额管理
        
        # 合并扩展字段：确保API兼容性，支持未来新增参数
        if request.extra:
            request_data.update(request.extra)
        
        # 发送请求并解析响应
        response = await self._request_with_retry("POST", "/embeddings", json=request_data)
        data = response.json()
        return self._parse_embedding_response(data)

    def _parse_embedding_response(self, data: Dict[str, Any]) -> EmbeddingResponse:
        """解析嵌入响应"""
        known_response_fields = {"object", "data", "model", "usage"}
        response_extra = {k: v for k, v in data.items() if k not in known_response_fields}
        
        embeddings = []
        for embedding_data in data.get("data", []):
            known_embedding_fields = {"index", "vector", "object"}  # 更新字段名
            embedding_extra = {k: v for k, v in embedding_data.items() if k not in known_embedding_fields}
            embedding = Embedding(
                index=embedding_data.get("index", 0),
                vector=embedding_data.get("vector", []),  # 使用新字段名
                object=embedding_data.get("object", "embedding"),
                extra=embedding_extra
            )
            embeddings.append(embedding)
        
        usage_data = data.get("usage")
        usage = None
        if usage_data:
            known_usage_fields = {"prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details", "completion_tokens_details"}
            usage_extra = {k: v for k, v in usage_data.items() if k not in known_usage_fields}
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                prompt_tokens_details=usage_data.get("prompt_tokens_details"),
                completion_tokens_details=usage_data.get("completion_tokens_details"),
                extra=usage_extra
            )
        
        return EmbeddingResponse(
            object=data.get("object", "list"),
            data=embeddings,
            model=data.get("model", ""),
            usage=usage,
            extra=response_extra
        )

    async def completions(self, request: CompletionRequest) -> Union[CompletionResponse, AsyncGenerator[CompletionChunk, None]]:
        """执行旧版文本补全请求，对应/v1/completions端点
        
        传统文本生成API，适用于简单的文本续写、代码补全等场景。
        相比chat_completion，不支持多轮对话和工具调用，但支持更多传统生成参数。
        建议新项目优先使用chat_completion方法。
        
        Args:
            request: CompletionRequest配置对象，包含传统补全的所有参数
                   包括prompt、suffix、echo、best_of等传统参数
        
        Returns:
            非流式模式: CompletionResponse - 完整的传统补全响应对象
            流式模式: AsyncGenerator[CompletionChunk, None] - 异步生成器
                      逐个产生文本补全块，适合实时显示
        
        Raises:
            OpenAIError: API调用失败时抛出，包含错误详情
            httpx.RequestError: 网络连接故障时抛出
        
        Deprecated:
            此为OpenAI兼容的旧版API，功能有限且不支持现代特性。
            新项目强烈建议使用chat_completion方法以获得更好的体验。
            
        Example:
            # 非流式调用 - 代码补全
            request = CompletionRequest(
                model="text-davinci-003",
                prompt="def fibonacci(n):",
                max_tokens=100,
                temperature=0.2
            )
            response = await client.completions(request)
            print(response.choices[0].text)
            
            # 流式调用 - 实时文本生成
            stream_request = CompletionRequest(
                model="text-davinci-003",
                prompt="Once upon a time",
                max_tokens=50,
                stream=True
            )
            async for chunk in await client.completions(stream_request):
                if chunk.choices[0].text:
                    print(chunk.choices[0].text, end="", flush=True)
        
        Note:
            - best_of参数在服务端生成多个候选，返回最佳的n个
            - echo=True时会在响应中包含输入prompt
            - 主要用于代码补全、文本续写等简单生成任务
        """
        request_data = self._build_completion_request_data(request)
        
        if request.stream:
            return self._stream_completion(request_data)
        else:
            response = await self._request_with_retry("POST", "/completions", json=request_data)
            data = response.json()
            return self._parse_completion_response(data)
    
    def _build_completion_request_data(self, request: CompletionRequest) -> Dict[str, Any]:
        """构建旧版补全请求数据，降低方法复杂度"""
        request_data = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": request.stream,
        }
        
        # 批量添加可选字段
        self._add_optional_field(request_data, "suffix", request.suffix)
        self._add_optional_field(request_data, "max_tokens", request.max_tokens)
        self._add_optional_field(request_data, "temperature", request.temperature)
        self._add_optional_field(request_data, "top_p", request.top_p)
        self._add_optional_field(request_data, "n", request.n)
        self._add_optional_field(request_data, "logprobs", request.logprobs)
        
        # echo字段特殊处理（默认False，只有显式设置时才添加）
        if request.echo is not False:
            request_data["echo"] = request.echo
            
        self._add_optional_field(request_data, "stop", request.stop)
        self._add_optional_field(request_data, "presence_penalty", request.presence_penalty)
        self._add_optional_field(request_data, "frequency_penalty", request.frequency_penalty)
        self._add_optional_field(request_data, "best_of", request.best_of)
        self._add_optional_field(request_data, "logit_bias", request.logit_bias)
        self._add_optional_field(request_data, "user", request.user)
        
        # 合并extra字段
        if request.extra:
            request_data.update(request.extra)
            
        return request_data

    def _parse_completion_response(self, data: Dict[str, Any]) -> CompletionResponse:
        """解析旧版补全响应"""
        known_response_fields = {"id", "object", "created", "model", "choices", "usage"}
        response_extra = {k: v for k, v in data.items() if k not in known_response_fields}
        
        choices = []
        for choice_data in data.get("choices", []):
            known_choice_fields = {"text", "index", "logprobs", "finish_reason"}
            choice_extra = {k: v for k, v in choice_data.items() if k not in known_choice_fields}
            
            choice = CompletionChoice(
                text=choice_data.get("text", ""),
                index=choice_data.get("index", 0),
                logprobs=choice_data.get("logprobs"),
                finish_reason=choice_data.get("finish_reason"),
                extra=choice_extra
            )
            choices.append(choice)
        
        usage_data = data.get("usage")
        usage = None
        if usage_data:
            known_usage_fields = {"prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details", "completion_tokens_details"}
            usage_extra = {k: v for k, v in usage_data.items() if k not in known_usage_fields}
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                prompt_tokens_details=usage_data.get("prompt_tokens_details"),
                completion_tokens_details=usage_data.get("completion_tokens_details"),
                extra=usage_extra
            )
        
        return CompletionResponse(
            id=data.get("id", ""),
            object=data.get("object", "text_completion"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
            extra=response_extra
        )

    async def _stream_completion(self, request_data: Dict[str, Any]) -> AsyncGenerator[CompletionChunk, None]:
        """流式补全"""
        async with self.client.stream("POST", "/completions", json=request_data) as response:
            self._check_error(response)
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        yield self._parse_completion_chunk(data)
                    except json.JSONDecodeError:
                        continue

    def _parse_completion_chunk(self, data: Dict[str, Any]) -> CompletionChunk:
        """解析流式补全块"""
        known_chunk_fields = {"id", "object", "created", "model", "choices"}
        chunk_extra = {k: v for k, v in data.items() if k not in known_chunk_fields}
        
        choices = []
        for choice_data in data.get("choices", []):
            known_choice_fields = {"text", "index", "logprobs", "finish_reason"}
            choice_extra = {k: v for k, v in choice_data.items() if k not in known_choice_fields}
            
            choice = CompletionChoice(
                text=choice_data.get("text", ""),
                index=choice_data.get("index", 0),
                logprobs=choice_data.get("logprobs"),
                finish_reason=choice_data.get("finish_reason"),
                extra=choice_extra
            )
            choices.append(choice)
        
        return CompletionChunk(
            id=data.get("id", ""),
            object=data.get("object", "text_completion"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            extra=chunk_extra
        )

    async def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """带自动重试机制的HTTP请求执行器，处理瞬时故障和速率限制
        
        为核心通信方法提供可靠性保障，自动重试失败的请求。特别针对
        429速率限制错误和各类网络故障实施智能重试策略。
        
        Args:
            method: HTTP方法字符串，如"GET"、"POST"、"PUT"、"DELETE"
            url: 请求路径，相对于base_url的相对路径，如"/chat/completions"
            **kwargs: 传递给httpx.request的其他参数，如json、params、headers等
        
        Returns:
            httpx.Response: 成功的HTTP响应对象，已通过错误检查
        
        Raises:
            OpenAIError: 经过所有重试后仍失败，且为API错误时抛出
            httpx.RequestError: 经过所有重试后仍失败，且为网络错误时抛出
            httpx.HTTPStatusError: 非429状态码的HTTP错误且重试耗尽时抛出
        
        Retry Strategy:
            - 总尝试次数: max_retries + 1 (首次尝试 + max_retries次重试)
            - 429速率限制: 无限重试直到成功或达到最大次数，使用指数退避
            - 其他HTTP错误: 立即抛出，不进行重试
            - 网络故障: 重试，使用指数退避策略
            - 退避时间: 2^attempt秒 (1s, 2s, 4s, 8s...)
        
        Reliability:
            - 指数退避避免加重服务器负载
            - 区分可重试错误(网络故障、429)和不可重试错误(4xx除429外)
            - 保留最后一次错误信息用于调试
        
        Example:
            # 内部调用示例，用户通常不直接使用此方法
            response = await self._request_with_retry(
                "POST", 
                "/chat/completions", 
                json={"model": "gpt-4", "messages": [...]}
            )
            data = response.json()
        
        Note:
            - 此为内部方法，用户不应直接调用
            - max_retries=0时禁用重试，适用于测试环境
            - 生产环境建议设置max_retries>=3以应对网络波动
        """
        # 保存最后一次错误，用于所有重试耗尽时的异常抛出
        last_error = None
        
        # 执行最多max_retries+1次尝试(首次+重试)
        for attempt in range(self.max_retries + 1):
            try:
                # 发送HTTP请求，使用配置的客户端和参数
                response = await self.client.request(method, url, **kwargs)
                
                # 检查响应是否包含API错误，如有则抛出异常
                self._check_error(response)
                
                # 成功响应，直接返回
                return response
                
            except httpx.HTTPStatusError as e:
                # HTTP状态码错误(4xx, 5xx)
                last_error = e
                
                # 专门处理429速率限制错误，进行重试
                if e.response.status_code == 429 and attempt < self.max_retries:
                    # 执行指数退避，等待后重试
                    await self._backoff(attempt)
                    continue
                
                # 其他HTTP错误或超过重试次数，直接抛出
                raise
                
            except httpx.RequestError as e:
                # 网络连接故障、超时等请求层面的错误
                last_error = e
                
                # 只要未达到最大重试次数，就进行重试
                if attempt < self.max_retries:
                    # 执行指数退避，等待后重试
                    await self._backoff(attempt)
                    continue
                
                # 超过重试次数，抛出最后的错误
                raise
        
        # 所有重试均失败，抛出最后一次错误
        raise last_error

    def _check_error(self, response: httpx.Response):
        """检查HTTP响应状态并转换API错误为结构化异常
        
        验证响应是否为错误状态，如果是则解析API返回的错误信息并抛出
        OpenAIError异常。提供统一的错误处理机制，便于上层代码捕获和处理。
        
        Args:
            response: httpx.Response对象，待检查的HTTP响应
        
        Raises:
            OpenAIError: 响应为错误状态时抛出，包含结构化的错误信息
                      包括错误消息、类型、代码和参数等详细信息
        
        Error Processing:
            - 检查response.is_error确定是否为错误状态(4xx, 5xx)
            - 尝试解析JSON格式的错误响应，提取error对象
            - 从error对象获取message、type、code、param等字段
            - JSON解析失败时回退到原始响应文本
        
        Error Types Handled:
            - invalid_request_error: 请求参数错误或缺失
            - authentication_error: API密钥无效或缺失  
            - permission_error: 权限不足
            - rate_limit_error: 超出速率限制(通常由_retry机制处理)
            - server_error: 服务器内部错误
        
        Example:
            # 内部调用流程示例
            try:
                response = await self.client.post("/chat/completions", json=payload)
                self._check_error(response)  # 检查并可能抛出异常
                data = response.json()  # 只有无错误时才解析
            except OpenAIError as e:
                print(f"API Error: {e.message}")
                print(f"Type: {e.type}, Code: {e.code}")
        
        Note:
            - 此方法应在每次API调用后立即调用
            - 成功响应(2xx状态码)不会抛出任何异常
            - 错误信息结构化便于程序化处理错误类型
            - 429错误通常由_retry机制在上层处理，此处主要处理其他错误
        """
        # 检查响应是否为错误状态(状态码4xx或5xx)
        if response.is_error:
            try:
                # 尝试解析JSON格式的错误响应
                error_data = response.json()
                
                # 提取error对象，API错误通常在error字段中
                error_info = error_data.get("error", {})
                
                # 抛出结构化的OpenAIError异常
                raise OpenAIError(
                    message=error_info.get("message", response.text),  # 错误消息，优先使用API提供的
                    type=error_info.get("type"),                       # 错误类型，如"invalid_request_error"
                    code=error_info.get("code"),                       # 错误代码，具体标识符
                    param=error_info.get("param")                      # 导致错误的参数或相关上下文
                )
                
            except (json.JSONDecodeError, ValueError):
                # JSON解析失败或数据格式错误，回退到原始响应文本
                raise OpenAIError(message=response.text)

    async def _backoff(self, attempt: int):
        """执行指数退避等待，避免重试时加重服务器负载
        
        实现经典的指数退避算法，在重试间增加等待时间，防止瞬时故障
        或速率限制情况下对服务器造成过大压力。
        
        Args:
            attempt: 当前重试尝试次数(从0开始计数)
                   第0次重试等待2^0=1秒，第1次等待2^1=2秒，依此类推
        
        Behavior:
            - 等待时间计算公式: 2^attempt 秒
            - attempt=0 → 1秒
            - attempt=1 → 2秒  
            - attempt=2 → 4秒
            - attempt=3 → 8秒
            - ...以此类推
        
        Purpose:
            - 给服务器时间恢复，特别是速率限制场景
            - 避免重试风暴，减少对下游服务的冲击
            - 符合分布式系统的最佳实践
        
        Example:
            # 内部调用示例，通常在重试循环中使用
            for attempt in range(max_retries):
                try:
                    response = await self.client.request(...)
                    return response
                except SomeTransientError:
                    await self._backoff(attempt)  # 等待后重试
            
            # 等待时间序列示例:
            # 第1次重试前等待: 1秒
            # 第2次重试前等待: 2秒  
            # 第3次重试前等待: 4秒
            # 第4次重试前等待: 8秒
        
        Note:
            - 此为内部方法，用户不应直接调用
            - 指数增长确保长时间重试不会导致过长等待
            - 对于频繁速率限制的场景，可能需要额外的抖动机制
            - 导入asyncio在函数内是为了避免模块加载时的依赖问题
        """
        # 动态导入asyncio(函数内导入避免循环依赖)
        import asyncio
        
        # 计算指数退避时间: 2^attempt秒
        backoff_time = 2 ** attempt
        
        # 异步睡眠指定时间
        await asyncio.sleep(backoff_time)

    async def close(self):
        """关闭HTTP客户端连接并释放资源
        
        优雅关闭底层的httpx.AsyncClient，清理连接池、取消挂起的请求、
        释放网络资源。必须在不再使用客户端时调用，避免资源泄漏。
        
        Behavior:
            - 关闭所有持久连接，返回连接池资源给操作系统
            - 取消所有正在进行的请求(如果有)
            - 清理内部缓冲区和状态
            - 标记客户端为已关闭状态，后续使用将抛出异常
        
        Usage Pattern:
            # 方式1: 使用async with语句(推荐)
            async with TinyLMClient(base_url, api_key) as client:
                response = await client.chat_completion(request)
            # 退出with块时自动调用close()
            
            # 方式2: 手动调用close()
            client = TinyLMClient(base_url, api_key)
            try:
                response = await client.chat_completion(request)
            finally:
                await client.close()  # 确保资源被释放
        
        Resource Management:
            - HTTP连接池: 关闭所有keep-alive连接
            - TCP套接字: 正常关闭，避免TIME_WAIT状态堆积
            - 内存缓冲: 清理读写缓冲区
            - 协程任务: 取消后台任务(如重定向跟随)
        
        Warning:
            - 未调用close()可能导致连接泄漏和资源耗尽
            - 特别是在高并发场景下，未关闭的客户端会占用大量socket
            - 推荐使用async with语句确保自动清理
        
        Example:
            # 正确用法 - 自动资源管理
            async with TinyLMClient(base_url, api_key) as client:
                models = await client.models_list()
                # 使用client进行各种操作...
            # 离开with块，连接自动关闭
            
            # 手动管理 - 需要确保调用close()
            client = TinyLMClient(base_url, api_key)
            try:
                response = await client.chat_completion(request)
                print(response.choices[0].message.content)
            finally:
                # 在finally块中确保关闭，即使发生异常
                await client.close()
        
        Note:
            - 多次调用close()是安全的，后续调用将是no-op
            - 关闭后客户端实例不应再被使用
            - 与__aexit__方法配合实现上下文管理器协议
        """
        # 调用httpx客户端的aclose方法，异步关闭所有连接和资源
        await self.client.aclose()

    async def __aenter__(self):
        """异步上下文管理器入口方法，支持async with语法
        
        实现异步上下文管理器协议，允许使用async with语句自动管理
        客户端生命周期。进入上下文时返回自身实例供使用。
        
        Returns:
            TinyLMClient: 返回自身实例，可在with块中使用
        
        Usage:
            # 推荐的客户端使用方式，确保资源自动清理
            async with TinyLMClient(base_url, api_key) as client:
                # 在with块内安全使用client
                response = await client.chat_completion(request)
                models = await client.models_list()
                embeddings = await client.embeddings(embedding_request)
            # 退出with块时自动调用__aexit__，关闭连接
        
        Benefits:
            - 自动资源管理: 无需手动调用close()
            - 异常安全: 即使发生异常也会清理资源
            - 代码简洁: 减少样板代码
            - 可读性: 明确表达资源的生命周期范围
        
        Example:
            # 典型使用模式
            async def process_with_llm():
                async with TinyLMClient(
                    base_url="https://api.openai.com/v1",
                    api_key="your-api-key",
                    max_retries=3
                ) as client:
                    # 执行多个API调用
                    chat_response = await client.chat_completion(chat_request)
                    embedding_response = await client.embeddings(embedding_request)
                    
                    # 处理结果
                    result = process_results(chat_response, embedding_response)
                    return result
            
            # 即使process_with_llm抛出异常，client也会被正确关闭
        
        Note:
            - 与__aexit__配合使用，形成完整的上下文管理器
            - 返回self使得可以在with语句中直接使用客户端方法
            - 这是使用TinyLMClient的首选方式
        """
        return self

    async def __aexit__(self, *_):
        """异步上下文管理器退出方法，自动清理资源
        
        实现异步上下文管理器协议的退出部分。无论with块如何退出
        (正常执行完毕或异常抛出)，都会调用此方法确保资源被正确释放。
        
        Args:
            *_: 可变位置参数，捕获exc_type, exc_val, exc_tb
                按照上下文管理器协议，但在本实现中不使用这些参数
        
        Behavior:
            - 无论with块是正常退出还是因异常退出，都会调用close()
            - 忽略传入的异常参数，不抑制异常传播
            - 异常会继续向上传播，但资源已确保被清理
        
        Exception Handling:
            - 不捕获或处理异常，让调用者感知原始错误
            - 即使在close()过程中发生异常，原始异常仍会传播
            - 资源清理优先，但不掩盖业务逻辑错误
        
        Example:
            try:
                async with TinyLMClient(base_url, api_key) as client:
                    response = await client.chat_completion(invalid_request)
                    # 如果发生异常，仍会调用__aexit__关闭连接
            except OpenAIError as e:
                print(f"API调用失败: {e.message}")
                # client已被自动关闭，无需手动清理
        
        Note:
            - 参数使用*_忽略，符合上下文管理器协议但不处理异常
            - 委托给close()方法执行实际的资源清理工作
            - 这是实现RAII(Resource Acquisition Is Initialization)模式的关键
        """
        # 调用close方法执行实际的资源清理工作
        await self.close()
