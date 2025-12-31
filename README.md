# TinyLMClient - 轻量级 OpenAI 兼容大模型客户端

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> 🚀 **一个功能完整、类型安全、高性能的 OpenAI 兼容大模型客户端库**

## 📋 概述

`TinyLMClient` 是一个基于 `httpx` 异步 HTTP 客户端构建的轻量级大模型访问库，提供**完全兼容 OpenAI API 规范**的类型安全接口。支持现代 AI 模型的所有核心功能，包括聊天补全、文本嵌入、工具调用、流式响应等。

### ✨ 核心特性

- **🎯 完全 OpenAI 兼容**: 支持 GPT、Claude、本地模型等所有 OpenAI 兼容服务
- **⚡ 异步高性能**: 基于 `httpx` 异步 IO，支持高并发请求处理
- **🔒 类型安全**: 全面的 `dataclass` 封装，编译期类型检查，消除运行时错误
- **🌊 流式响应**: 支持实时流式输出，显著降低感知延迟
- **🔄 智能重试**: 指数退避机制，自动处理网络故障和速率限制
- **📦 编码优化**: 支持 gzip/deflate/compress 内容编码，优化传输效率
- **🔮 前向兼容**: 扩展字段保留机制，确保 API 升级时的兼容性
- **🛡️ 统一错误处理**: 结构化异常体系，便于调试和错误恢复
- **🏗️ 现代化架构**: 配置驱动、分层解析、流式抽象的设计模式

## 🏗️ 架构设计

### 核心组件

```
TinyLMClient/
├── 枚举定义 (8个)          # 类型安全的常量定义
├── 数据类 (15个)          # 请求/响应对象封装
├── 异常体系 (1个)          # 结构化错误处理
├── HTTP客户端 (1个)        # 异步通信核心
└── 业务方法 (20+)         # 完整API覆盖
```

### 设计原则

- **配置类驱动**: 所有请求参数通过 `dataclass` 管理，避免字典参数错误
- **分层解析**: 响应数据逐层解析为类型安全对象，确保数据完整性
- **流式抽象**: 统一流式和非流式接口设计，简化调用逻辑
- **错误恢复**: 智能重试机制处理瞬时故障，提高系统可靠性

## 📦 安装依赖

```bash
pip install httpx
```

## 🚀 快速开始

### 基本用法

```python
import asyncio
from tiny_lm_client import TinyLMClient, ChatCompletionRequest, Message

async def main():
    # 初始化客户端
    async with TinyLMClient(
        base_url="https://api.openai.com/v1",
        api_key="your-api-key",
        max_retries=3,
        timeout=30.0
    ) as client:
        # 构建请求
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[
                Message(role="user", content="Hello, how are you?")
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        # 发送请求（非流式）
        response = await client.chat_completion(request)
        print(response.choices[0].message.content)
        
        # 发送请求（流式）
        stream_request = ChatCompletionRequest(
            model="gpt-4",
            messages=[Message(role="user", content="Tell me a story")],
            stream=True
        )
        
        async for chunk in await client.chat_completion(stream_request):
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)

asyncio.run(main())
```

### 本地模型服务

```python
# 连接本地部署的模型服务
client = TinyLMClient(
    base_url="http://localhost:8000/v1",
    api_key="",  # 本地服务可能不需要API密钥
    max_retries=1
)
```

## 📚 API 参考

### TinyLMClient 类

#### 构造函数

```python
TinyLMClient(
    base_url: str,                    # API服务基础URL
    api_key: str,                     # API认证密钥
    encoding: Optional[EncodingType], # 内容编码类型
    max_retries: int = 0,            # 最大重试次数
    timeout: float = 60.0            # 请求超时时间
)
```

#### 主要方法

##### 1. 聊天补全 (`chat_completion`)

**功能**: 执行聊天补全请求，支持流式和非流式响应

```python
async def chat_completion(
    request: ChatCompletionRequest
) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionChunk, None]]
```

**参数**:
- `request`: `ChatCompletionRequest` 配置对象

**返回**:
- 非流式: `ChatCompletionResponse` - 完整响应对象
- 流式: `AsyncGenerator[ChatCompletionChunk, None]` - 异步生成器

**示例**:
```python
# 非流式调用
response = await client.chat_completion(request)
print(response.choices[0].message.content)

# 流式调用
async for chunk in await client.chat_completion(stream_request):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

##### 2. 文本嵌入 (`embeddings`)

**功能**: 将文本转换为高维向量表示

```python
async def embeddings(
    request: EmbeddingRequest
) -> EmbeddingResponse
```

**应用场景**:
- 语义搜索
- 文本聚类
- 相似度计算
- 特征工程

**示例**:
```python
request = EmbeddingRequest(
    model="text-embedding-ada-002",
    input=["First document", "Second document"],
    encoding_format="float"
)
response = await client.embeddings(request)
vector = response.data[0].vector  # 1536维向量
```

##### 3. 模型列表 (`models_list`)

**功能**: 获取可用模型列表

```python
async def models_list() -> List[Model]
```

**示例**:
```python
models = await client.models_list()
for model in models:
    print(f"Model: {model.id}, Owner: {model.owned_by}")
```

##### 4. 传统补全 (`completions`)

**功能**: 旧版文本补全 API（已弃用）

```python
async def completions(
    request: CompletionRequest
) -> Union[CompletionResponse, AsyncGenerator[CompletionChunk, None]]
```

⚠️ **注意**: 建议使用 `chat_completion` 方法替代

### 数据类详解

#### 请求类

##### ChatCompletionRequest

完整的聊天补全请求配置，支持所有 OpenAI API 参数：

```python
@dataclass
class ChatCompletionRequest:
    model: str                                    # 模型标识符
    messages: List[Union[Message, ChatMessage, Dict]]  # 对话历史
    temperature: float = 1.0                     # 采样温度
    max_tokens: Optional[int] = None             # 最大token数
    stream: bool = False                         # 流式响应
    top_p: Optional[float] = None                # 核采样阈值
    frequency_penalty: Optional[float] = None    # 频率惩罚
    presence_penalty: Optional[float] = None     # 存在惩罚
    stop: Optional[Union[str, List[str]]] = None # 停止序列
    n: int = 1                                   # 生成数量
    seed: Optional[int] = None                   # 随机种子
    tools: Optional[List[Union[Tool, Dict]]] = None      # 工具定义
    tool_choice: Optional[Union[ToolChoiceType, Dict]] = None  # 工具策略
    response_format: Optional[Union[ResponseFormatType, ResponseFormat, Dict]] = None  # 响应格式
    user: Optional[str] = None                   # 用户标识
    parallel_tool_calls: Optional[bool] = None   # 并行工具调用
    logit_bias: Optional[Dict[int, float]] = None # token偏置
    logprobs: Optional[bool] = None              # 返回概率
    top_logprobs: Optional[int] = None           # 顶部概率数量
    extra: Dict[str, Any] = field(default_factory=dict)  # 扩展字段
```

##### EmbeddingRequest

文本嵌入请求配置：

```python
@dataclass
class EmbeddingRequest:
    model: str                              # 嵌入模型
    input: Union[str, List[str]]           # 输入文本
    encoding_format: Optional[str] = None  # 编码格式
    user: Optional[str] = None             # 用户标识
    extra: Dict[str, Any] = field(default_factory=dict)
```

##### CompletionRequest

传统文本补全请求（已弃用）：

```python
@dataclass
class CompletionRequest:
    model: str                                    # 模型标识符
    prompt: Union[str, List[str], List[int], List[List[int]]]  # 输入提示
    suffix: Optional[str] = None                 # 后缀内容
    max_tokens: Optional[int] = None             # 最大token数
    temperature: Optional[float] = None          # 采样温度
    top_p: Optional[float] = None                # 核采样阈值
    n: Optional[int] = None                      # 生成数量
    stream: bool = False                         # 流式响应
    logprobs: Optional[int] = None              # 返回概率
    echo: bool = False                           # 回显输入
    stop: Optional[Union[str, List[str]]] = None # 停止序列
    presence_penalty: Optional[float] = None     # 存在惩罚
    frequency_penalty: Optional[float] = None    # 频率惩罚
    best_of: Optional[int] = None                # 最佳候选数
    logit_bias: Optional[Dict[int, float]] = None # token偏置
    user: Optional[str] = None                   # 用户标识
    extra: Dict[str, Any] = field(default_factory=dict)
```

#### 响应类

##### ChatCompletionResponse

聊天补全完整响应：

```python
@dataclass
class ChatCompletionResponse:
    id: str                                      # 响应ID
    object: str = "chat.completion"              # 对象类型
    created: int = 0                             # 创建时间戳
    model: str = ""                              # 模型名称
    choices: List[Choice] = field(default_factory=list)  # 生成结果
    usage: Optional[Usage] = None                # 使用统计
    system_fingerprint: Optional[str] = None     # 系统指纹
    service_tier: Optional[str] = None           # 服务等级
    extra: Dict[str, Any] = field(default_factory=dict)  # 扩展字段
```

##### EmbeddingResponse

文本嵌入响应：

```python
@dataclass
class EmbeddingResponse:
    object: str = "list"                         # 对象类型
    data: List[Embedding] = field(default_factory=list)  # 嵌入向量
    model: str = ""                              # 模型名称
    usage: Optional[Usage] = None                # 使用统计
    extra: Dict[str, Any] = field(default_factory=dict)
```

#### 消息类

##### Message & ChatMessage

支持基础消息和增强消息：

```python
@dataclass
class Message:
    role: Role          # 角色 (SYSTEM/USER/ASSISTANT)
    content: str        # 文本内容

@dataclass
class ChatMessage:
    role: Role                                  # 角色 (包含TOOL/FUNCTION)
    content: Optional[str] = None               # 文本内容
    tool_calls: Optional[List[ToolCall]] = None # 工具调用
    refusal: Optional[str] = None               # 拒绝回复
    extra: Dict[str, Any] = field(default_factory=dict)
```

#### 工具调用类

##### Tool & ToolCall

支持函数调用功能：

```python
@dataclass
class Tool:
    type: str = "function"                      # 工具类型
    function: Optional[Dict[str, Any]] = None   # 函数定义

@dataclass
class ToolCall:
    id: str                                     # 调用ID
    type: str = "function"                      # 工具类型
    function: Dict[str, Any] = field(default_factory=dict)  # 函数调用详情
    extra: Dict[str, Any] = field(default_factory=dict)
```

### 枚举类型

#### 核心枚举

```python
class Role(str, Enum):
    SYSTEM = "system"         # 系统指令
    USER = "user"             # 用户输入
    ASSISTANT = "assistant"   # AI助手回复
    TOOL = "tool"             # 工具执行结果
    FUNCTION = "function"     # 函数调用结果

class FinishReason(str, Enum):
    STOP = "stop"             # 自然结束
    LENGTH = "length"         # 长度限制
    TOOL_CALLS = "tool_calls" # 工具调用
    CONTENT_FILTER = "content_filter" # 内容过滤
    FUNCTION_CALL = "function_call"   # 函数调用

class ToolChoiceType(str, Enum):
    NONE = "none"             # 不调用工具
    AUTO = "auto"             # 自动选择
    REQUIRED = "required"     # 必须调用

class ResponseFormatType(str, Enum):
    TEXT = "text"             # 纯文本
    JSON_OBJECT = "json_object" # JSON格式

class EncodingType(str, Enum):
    GZIP = "gzip"            # GZIP压缩
    DEFLATE = "deflate"      # DEFLATE压缩
    COMPRESS = "compress"    # UNIX压缩
```

### 异常处理

#### OpenAIError

统一的 API 错误异常类：

```python
class OpenAIError(Exception):
    def __init__(self, message: str, type: str = None, code: str = None, param: Any = None):
        self.message = message    # 错误描述
        self.type = type          # 错误类型
        self.code = code          # 错误代码
        self.param = param        # 相关参数
```

**错误类型**:
- `invalid_request_error`: 请求参数错误
- `authentication_error`: 认证失败
- `permission_error`: 权限不足
- `rate_limit_error`: 速率限制
- `server_error`: 服务器错误

**使用示例**:
```python
try:
    response = await client.chat_completion(request)
except OpenAIError as e:
    print(f"Error: {e.message}")
    print(f"Type: {e.type}, Code: {e.code}")
```

## 🔧 高级特性

### 1. 流式响应处理

```python
# 实时显示生成内容
stream_request = ChatCompletionRequest(
    model="gpt-4",
    messages=[Message(role="user", content="Write a poem")],
    stream=True
)

async for chunk in await client.chat_completion(stream_request):
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)  # 实时输出
```

### 2. 工具调用支持

```python
# 定义可用工具
tools = [
    Tool(
        type="function",
        function={
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    )
]

request = ChatCompletionRequest(
    model="gpt-4",
    messages=[Message(role="user", content="What's the weather in Beijing?")],
    tools=tools,
    tool_choice="auto"
)
```

### 3. JSON 格式强制输出

```python
request = ChatCompletionRequest(
    model="gpt-4",
    messages=[Message(role="user", content="Return user data as JSON")],
    response_format=ResponseFormatType.JSON_OBJECT
)
```

### 4. 批量嵌入计算

```python
# 高效批量处理
request = EmbeddingRequest(
    model="text-embedding-ada-002",
    input=[
        "Document 1 content",
        "Document 2 content", 
        "Document 3 content"
    ]
)
response = await client.embeddings(request)
# response.data[i].vector 对应 input[i] 的嵌入向量
```

### 5. 自定义重试策略

```python
client = TinyLMClient(
    base_url="https://api.openai.com/v1",
    api_key="your-key",
    max_retries=5,    # 最多重试5次
    timeout=120.0     # 超时2分钟
)
```

## ⚙️ 配置选项

### HTTP 客户端配置

```python
# 连接池配置
client = TinyLMClient(
    base_url="https://api.openai.com/v1",
    api_key="your-key",
    timeout=60.0
)

# 底层 httpx 配置
# - 最大连接数: 100
# - 保持连接数: 20
# - 超时设置: 可配置
# - 编码支持: gzip/deflate/compress
```

### 内容编码

```python
# 启用 GZIP 压缩
client = TinyLMClient(
    base_url="https://api.openai.com/v1",
    api_key="your-key",
    encoding=EncodingType.GZIP
)
```

## 📊 性能优化

### 最佳实践

1. **流式响应**: 大响应建议使用流式模式，降低内存占用和感知延迟
2. **批量处理**: 嵌入计算支持批量处理，显著提高吞吐量
3. **连接复用**: 客户端自动管理连接池，避免重复建立连接
4. **智能重试**: 合理配置重试次数，平衡可靠性和响应速度
5. **超时设置**: 根据任务复杂度调整超时时间

### 性能指标

- **流式首字节时间**: < 100ms（取决于网络和服务端）
- **批量嵌入吞吐**: 1000+ 文本/秒（取决于模型和服务）
- **并发支持**: 100+ 并发连接
- **内存效率**: 流式模式常数级内存占用

## 🛠️ 错误处理

### 自动重试场景

- **429 速率限制**: 无限重试，指数退避
- **网络故障**: 最多 `max_retries` 次重试
- **服务器错误**: 5xx 状态码自动重试

### 手动错误处理

```python
try:
    response = await client.chat_completion(request)
except OpenAIError as e:
    if e.type == "authentication_error":
        # 处理认证错误
        refresh_api_key()
    elif e.type == "rate_limit_error":
        # 处理速率限制
        await asyncio.sleep(60)
    elif e.type == "invalid_request_error":
        # 处理参数错误
        validate_request()
except httpx.RequestError as e:
    # 处理网络错误
    logger.error(f"Network error: {e}")
```

## 🔍 调试和监控

### 请求追踪

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 系统指纹追踪
response = await client.chat_completion(request)
print(f"System fingerprint: {response.system_fingerprint}")
```

### 使用统计

```python
response = await client.chat_completion(request)
usage = response.usage
print(f"Prompt tokens: {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total tokens: {usage.total_tokens}")
```

## 🚨 注意事项

### 重要提醒

1. **线程安全**: 实例方法非线程安全，每个异步上下文单独使用
2. **资源管理**: 务必使用 `async with` 语句或手动调用 `close()`
3. **API 版本**: 遵循 OpenAI API 规范，注意版本兼容性
4. **成本控制**: 监控 token 使用情况，合理设置 `max_tokens`
5. **错误处理**: 生产环境必须实现完整的错误处理逻辑

### 弃用警告

- `completions()` 方法已弃用，建议使用 `chat_completion()`
- 传统 `Message` 类功能有限，推荐使用 `ChatMessage`
- `Function` 相关字段建议使用 `Tool` 替代

### 兼容性说明

- **Python 版本**: >= 3.8
- **HTTPX 版本**: >= 0.24.0
- **API 兼容**: OpenAI API v1
- **模型支持**: GPT系列、Claude、本地模型等兼容服务

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发环境

```bash
pip install -r requirements-dev.txt
```

### 代码规范

- 遵循 PEP 8 代码风格
- 使用类型注解
- 编写详细的 docstring
- 包含单元测试

## 📄 许可证

MIT License

## 🙏 致谢

- 基于 `httpx` 异步 HTTP 客户端
- 兼容 OpenAI API 规范
- 灵感来源于现代 Python 异步编程最佳实践

---

**版本**: 1.0.0  
**更新日期**: 2025-12-31  
**维护者**: AI-Corp
