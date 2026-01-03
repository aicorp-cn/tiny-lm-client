import asyncio
import json
from typing import Dict, List, Optional, Union, AsyncGenerator, Any

from ..entities.requests import (
    ChatCompletionRequest,
    EmbeddingRequest,
    CompletionRequest
)
from ..entities.responses import (
    ChatCompletionResponse,
    ChatCompletionChunk,
    EmbeddingResponse,
    CompletionResponse,
    CompletionChunk,
    Model
)
from ..entities import Message, ChatMessage, ToolCall

from tiny_lm_client.enums import (
    ToolChoiceType,
    ResponseFormatType,
    Role,
    EncodingType
)

from tiny_lm_client.errors import (
    BaseError,
    RequestValidationError,
    ChatCompletionError,
    EmbeddingError,
    ModelListError,
    CompletionError
)
from tiny_lm_client.validators import (
    ParamsValidator,
    RequestValidator
)
from tiny_lm_client.http.http_client import HTTPClient
from tiny_lm_client.parsers.response_parser import ResponseParser


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
        ParamsValidator.validate_base_url(base_url)
        if api_key is None:
            raise ValueError("api_key cannot be None")
        ParamsValidator.validate_api_key(api_key)
                
        self.base_url = base_url.rstrip('/')  # 移除末尾斜杠，避免路径重复
        self._api_key = api_key
        self._masked_api_key = ParamsValidator.mask_sensitive_data(api_key)
        self.encoding = encoding
        self.max_retries = max(max_retries, 0)  # 确保非负
        self.timeout = timeout
                
        # 构建HTTP请求头，遵循OpenAI API规范
        self.headers = {
            'Accept': 'application/json',           # 期望JSON响应
            "Content-Type": "application/json",   # 请求体为JSON格式
            'User-Agent': 'TinyLMClient/1.0',
            "Authorization": f"Bearer {self._api_key}",  # Bearer token认证
        }
        
        # 条件性添加内容编码头，优化传输效率
        if encoding:
            self.headers['Accept-Encoding'] = encoding.value
            
        # 初始化HTTP客户端组件
        self.http_client = HTTPClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # 初始化解析器组件
        self.parser = ResponseParser()

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
            if isinstance(tc, (ToolCall, type(None))):  # Check if it's a ToolCall instance
                if tc is None:
                    continue
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
            APIError: API调用失败时抛出，包含错误详情
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
        
        # 验证请求数据
        validation_errors = RequestValidator.validate_chat_completion_request(request_data)
        if validation_errors:
            raise RequestValidationError(
                message=f"Request validation failed: {'; '.join(validation_errors)}"
            )
        
        # 根据stream参数选择响应模式
        if request.stream:
            # 返回流式响应生成器，支持异步迭代
            return self._stream_chat_completion(request_data)
        else:
            # 返回完整的响应对象
            return await self._non_stream_chat_completion(request_data)
    
    async def _non_stream_chat_completion(self, request_data: Dict[str, Any]) -> ChatCompletionResponse:
        """非流式聊天补全"""
        response = await self.http_client.request_with_retry("POST", "/chat/completions", endpoint="/chat/completions", json=request_data)
        data = response.json()
        return self.parser.parse_chat_completion_response(data)

    async def _stream_chat_completion(self, request_data: Dict[str, Any]) -> AsyncGenerator[ChatCompletionChunk, None]:
        """带背压控制的流式聊天补全处理
        
        使用Server-Sent Events(SSE)协议处理流式响应，逐行读取
        服务器发送的数据块。实现流式响应的实时处理和
        传输，显著降低首字节时间(TTFB)和感知延迟。添加
        背压控制机制防止内存积压。

        Args:
            request_data: 已构建的请求数据字典

        Yields:
            ChatCompletionChunk: 解析后的响应块对象
            
        Performance:
            - 流式传输显著降低感知延迟，用户可实时看到输出
            - 背压控制防止内存积压
            - 适合实时对话和长文本生成场景

        Protocol:
            - 遵循SSE (Server-Sent Events) 协议格式
            - 数据行以"data: "开头
            - 结束标识为"[DONE]"
            - 每个数据块为JSON格式

        Error Handling:
            - JSON解析失败时跳过该数据块
            - 保持流的连续性
            - 传递底层HTTP错误

        Example:
            async for chunk in _stream_chat_completion(request_data):
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)

        Note:
            - 此为内部方法，用户不应直接调用
            - 通过chat_completion方法的stream=True参数间接使用
            - 与非流式响应使用相同的解析逻辑
        """
        chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=10)  # 限制缓冲区大小
        endpoint = "/chat/completions"  # 确定端点路径
        
        # 启动生产者任务
        async def producer():
            try:
                async with self.http_client.client.stream("POST", endpoint, json=request_data) as response:
                    self.http_client.check_error(response, endpoint)
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                chunk = self.parser.parse_chat_completion_chunk(data)
                                
                                # 使用put_nowait避免阻塞生产者
                                try:
                                    await chunk_queue.put(chunk)
                                except asyncio.QueueFull:
                                    # 如果队列满，等待消费者处理
                                    await chunk_queue.put(chunk)
                                    
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                # 将异常放入队列，让消费者知道出错
                await chunk_queue.put(e)
            finally:
                # 发送结束信号
                await chunk_queue.put(None)
        
        # 启动生产者任务
        producer_task = asyncio.create_task(producer())
        
        # 消费者循环 - 从队列中获取数据并yield
        try:
            while True:
                item = await chunk_queue.get()
                
                # 检查是否是结束信号
                if item is None:
                    break
                # 检查是否是异常
                elif isinstance(item, Exception):
                    raise item
                else:
                    yield item
                    chunk_queue.task_done()
        finally:
            # 确保生产者任务完成
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass

    async def models_list(self) -> List[Model]:
        """获取API服务支持的可用模型列表，对应/models端点
        
        查询后端服务当前可用的所有AI模型，包括模型标识符、发布时间、所有者等信息。
        用于动态发现服务能力和选择合适的模型进行推理。

        Returns:
            List[Model]: 模型信息列表，每个Model对象包含模型的元数据
                       包括id、created、owned_by等字段，按API返回顺序排列

        Raises:
            APIError: API调用失败时抛出，包含错误详情
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
        response = await self.http_client.request_with_retry("GET", "/models", endpoint="/models")
        data = response.json()
        
        # 解析响应数据，构建Model对象列表
        return self.parser.parse_models_list(data)

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
            APIError: API调用失败时抛出，包含错误详情
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
        
        # 验证请求数据
        validation_errors = RequestValidator.validate_embedding_request(request_data)
        if validation_errors:
            raise RequestValidationError(
                message=f"Request validation failed: {'; '.join(validation_errors)}"
            )
        
        # 发送请求并解析响应
        response = await self.http_client.request_with_retry("POST", "/embeddings", endpoint="/embeddings", json=request_data)
        data = response.json()
        return self.parser.parse_embedding_response(data)

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
            APIError: API调用失败时抛出，包含错误详情
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
        
        # 验证请求数据
        validation_errors = RequestValidator.validate_completion_request(request_data)
        if validation_errors:
            raise RequestValidationError(
                message=f"Request validation failed: {'; '.join(validation_errors)}"
            )
        
        if request.stream:
            return self._stream_completion(request_data)
        else:
            response = await self.http_client.request_with_retry("POST", "/completions", endpoint="/completions", json=request_data)
            data = response.json()
            return self.parser.parse_completion_response(data)
    
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

    async def _stream_completion(self, request_data: Dict[str, Any]) -> AsyncGenerator[CompletionChunk, None]:
        """流式补全"""
        endpoint = "/completions"  # 确定端点路径
        async with self.http_client.client.stream("POST", endpoint, json=request_data) as response:
            self.http_client.check_error(response, endpoint)
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        yield self.parser.parse_completion_chunk(data)
                    except json.JSONDecodeError:
                        continue

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
        await self.http_client.close()

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
            except BaseError as e:
                print(f"API调用失败: {e.message}")
                # client已被自动关闭，无需手动清理

        Note:
            - 参数使用*_忽略，符合上下文管理器协议但不处理异常
            - 委托给close()方法执行实际的资源清理工作
            - 这是实现RAII(Resource Acquisition Is Initialization)模式的关键
        """
        # 调用close方法执行实际的资源清理工作
        await self.close()