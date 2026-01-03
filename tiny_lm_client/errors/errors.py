from typing import Any, Optional


class BaseError(Exception):
    """OpenAI API基础异常类，所有其他异常的基类
    
    提供统一的异常处理接口，同时支持类型安全的错误处理。
    """
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ValidationError(BaseError):
    """验证错误基类
    
    用于处理请求和响应验证失败的情况。
    """
    def __init__(self, message: str):
        super().__init__(message)


class RequestValidationError(ValidationError):
    """请求验证错误
    
    当请求参数验证失败时抛出。
    """
    def __init__(self, message: str):
        super().__init__(message)


class ResponseValidationError(ValidationError):
    """响应验证错误
    
    当响应数据验证失败时抛出。
    """
    def __init__(self, message: str):
        super().__init__(message)


class APIError(BaseError):
    """API错误基类
    
    用于处理API调用层面的错误。
    
    Attributes:
        type: 错误类型标识符
        code: 错误代码
    """
    def __init__(self, message: str, type: str, code: Optional[str] = None):
        super().__init__(message)
        self.type = type
        self.code = code


class ChatCompletionError(APIError):
    """聊天补全错误
    
    聊天补全API操作相关的错误。
    """
    pass


class EmbeddingError(APIError):
    """嵌入错误
    
    嵌入API操作相关的错误。
    """
    pass


class ModelListError(APIError):
    """模型列表错误
    
    模型列表API操作相关的错误。
    """
    pass


class CompletionError(APIError):
    """传统补全错误
    
    传统补全API操作相关的错误。
    """
    pass