import asyncio
import httpx
import json
import random
from typing import Dict, Any, Optional
from tiny_lm_client.errors import (APIError, ChatCompletionError, EmbeddingError, ModelListError, CompletionError, RequestValidationError)


class HTTPClient:
    """HTTP客户端类，专门处理HTTP请求和响应
    
    遵循单一职责原则，专门处理HTTP通信逻辑，包括连接管理、
    请求发送、错误处理、重试策略等，与业务逻辑完全分离。
    """
    
    def __init__(
        self,
        base_url: str,
        headers: Dict[str, str],
        timeout: float,
        max_retries: int
    ):
        """初始化HTTP客户端
        
        Args:
            base_url: API服务的基础URL
            headers: HTTP请求头字典
            timeout: HTTP请求超时时间(秒)
            max_retries: 最大重试次数
        """
        self.base_url = base_url
        self.headers = headers
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 初始化异步HTTP客户端，支持连接池和keep-alive
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=100,      # 最大并发连接数
                max_keepalive_connections=20  # 保持的连接池大小
            )
        )
    
    async def request_with_retry(self, method: str, url: str, endpoint: str = None, **kwargs) -> httpx.Response:
        """带自动重试机制的HTTP请求执行器，处理瞬时故障和速率限制
            
        为核心通信方法提供可靠性保障，自动重试失败的请求。特别针对
        429速率限制错误和各类网络故障实施智能重试策略。
    
        Args:
            method: HTTP方法字符串，如"GET"、"POST"、"PUT"、"DELETE"
            url: 请求路径，相对于base_url的相对路径，如"/chat/completions"
            endpoint: API端点路径，用于确定错误类型
            **kwargs: 传递给httpx.request的其他参数，如json、params、headers等
            
        Returns:
            httpx.Response: 成功的HTTP响应对象，已通过错误检查
            
        Raises:
            APIError: 经过所有重试后仍失败，且为API错误时抛出
            httpx.RequestError: 经过所有重试后仍失败，且为网络错误时抛出
            httpx.HTTPStatusError: 非429状态码的HTTP错误且重试耗尽时抛出
            
        Retry Strategy:
            - 总尝试次数: max_retries + 1 (首次尝试 + max_retries次重试)
            - 429速率限制: 无限重试直到成功或达到最大次数，使用指数退避
            - 其他HTTP错误: 立即抛出，不进行重试
            - 网络故障: 重试，使用指数退避策略
            - 退避时间: 2^attempt秒 (1s, 2s, 4s, 8s...)
        """
        # 保存最后一次错误，用于所有重试耗尽时的异常抛出
        last_error = None
            
        # 如果未提供endpoint，则使用url作为endpoint
        if endpoint is None:
            endpoint = url
            
        # 执行最多max_retries+1次尝试(首次+重试)
        for attempt in range(self.max_retries + 1):
            try:
                # 发送HTTP请求，使用配置的客户端和参数
                response = await self.client.request(method, url, **kwargs)
                    
                # 检查响应是否包含API错误，如有则抛出异常
                self.check_error(response, endpoint)
                    
                # 成功响应，直接返回
                return response
                    
            except httpx.HTTPStatusError as e:
                # HTTP状态码错误(4xx, 5xx)
                last_error = e
                    
                # 专门处理429速率限制错误，进行重试
                if e.response.status_code == 429 and attempt < self.max_retries:
                    # 执行指数退避，等待后重试
                    await self.backoff(attempt)
                    continue
                    
                # 其他HTTP错误或超过重试次数，直接抛出
                raise
                    
            except httpx.RequestError as e:
                # 网络连接故障、超时等请求层面的错误
                last_error = e
                    
                # 只要未达到最大重试次数，就进行重试
                if attempt < self.max_retries:
                    # 执行指数退避，等待后重试
                    await self.backoff(attempt)
                    continue
                    
                # 超过重试次数，抛出最后的错误
                raise
            
        # 所有重试均失败，抛出最后一次错误
        raise last_error

    def check_error(self, response: httpx.Response, endpoint: str = None):
        """检查HTTP响应错误，安全处理错误信息
        
        验证HTTP响应状态码，对于4xx/5xx错误状态码，解析错误
        响应内容并转换为适当的APIError异常。确保错误信息结构
        化，便于调用方处理和调试。错误响应遵循OpenAI API
        规范的error对象格式。安全处理错误信息，防止敏感
        数据泄露。根据端点路径选择适当的错误类型。
        
        Args:
            response: HTTP响应对象
            endpoint: API端点路径，用于确定错误类型
        
        Raises:
            APIError: 当响应状态码为4xx/5xx时抛出
        
        Error Format:
            - message: 错误描述信息
            - type: 错误类型标识
            - code: 错误代码
        
        Example:
            try:
                response = await self.client.request(...)
                self.check_error(response, "/chat/completions")
            except APIError as e:
                print(f"API Error: {e.message}, Code: {e.code}")
        """
        if response.is_error:
            try:
                error_data = response.json()
                error_info = error_data.get("error", {})
                
                # 安全提取错误信息，避免泄露原始响应
                message = error_info.get("message", f"API request failed with status {response.status_code}")
                error_type = error_info.get("type", "unknown_error")
                error_code = error_info.get("code")
                
                # 根据端点路径选择适当的错误类型
                if endpoint:
                    if "/chat/completions" in endpoint:
                        raise ChatCompletionError(message=message, type=error_type, code=error_code)
                    elif "/embeddings" in endpoint:
                        raise EmbeddingError(message=message, type=error_type, code=error_code)
                    elif "/models" in endpoint:
                        raise ModelListError(message=message, type=error_type, code=error_code)
                    elif "/completions" in endpoint:
                        raise CompletionError(message=message, type=error_type, code=error_code)
                
                # 默认使用通用APIError
                raise APIError(message=message, type=error_type, code=error_code)
            except (json.JSONDecodeError, ValueError):
                # 避免直接返回原始响应内容
                raise APIError(
                    message=f"API request failed with status {response.status_code}",
                    type="api_error",
                    code=response.status_code
                )

    async def backoff(self, attempt: int):
        """指数退避策略，添加抖动避免请求雪崩
        
        实现指数退避算法，每次重试的等待时间按指数增长，避免
        瞬时故障时的频繁重试对服务造成过大压力。退避时间按
        2^attempt秒增长，添加随机抖动避免请求雪崩。
        
        Args:
            attempt: 当前重试尝试次数(从0开始)
        
        Example:
            # 第1次重试前等待: 1秒
            # 第2次重试前等待: 2秒  
            # 第3次重试前等待: 4秒
            # 第4次重试前等待: 8秒
        
        Note:
            - 此为内部方法，用户不应直接调用
            - 指数增长确保长时间重试不会导致过长等待
            - 抖动机制避免请求雪崩
        """
        # 计算基础退避时间
        base_backoff = 2 ** attempt
        # 添加抖动 (减少25%到增加25%之间)
        jitter = random.uniform(0.75, 1.25)
        backoff_time = base_backoff * jitter
        
        # 确保不超过最大退避时间
        max_backoff = 60.0  # 最大60秒
        backoff_time = min(backoff_time, max_backoff)
        
        await asyncio.sleep(backoff_time)

    async def close(self):
        """关闭HTTP客户端连接并释放资源"""
        await self.client.aclose()