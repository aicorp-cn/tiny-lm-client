"""请求实体包 - 包含所有API请求数据类"""

from .chat_completion_request import ChatCompletionRequest
from .embedding_request import EmbeddingRequest
from .completion_request import CompletionRequest

__all__ = [
    'ChatCompletionRequest',
    'EmbeddingRequest',
    'CompletionRequest'
]