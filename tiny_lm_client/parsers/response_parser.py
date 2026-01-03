from typing import Dict, Any, Optional, List, TYPE_CHECKING
from ..entities import (
    ChatCompletionResponse, Choice, ChatMessage, ToolCall, LogProbs,
    Usage, ChatCompletionChunk, StreamChoice, Delta, EmbeddingResponse,
    Embedding, CompletionResponse, CompletionChoice, Model
)
from tiny_lm_client.enums import FinishReason, Role
from tiny_lm_client.errors import ResponseValidationError
from tiny_lm_client.validators import ResponseValidator

if TYPE_CHECKING:
    from ..entities import CompletionChunk


class ResponseParser:
    """响应解析器类，专门负责API响应数据的解析工作
    
    遵循单一职责原则，专门处理响应解析逻辑，将API返回的
    JSON数据转换为类型安全的数据类对象。
    """
    
    @staticmethod
    def extract_extra_fields(data: Dict[str, Any], known_fields: set) -> Dict[str, Any]:
        """提取已知字段外的额外字段"""
        return {k: v for k, v in data.items() if k not in known_fields}
    
    def parse_chat_completion_response(self, data: Dict[str, Any]) -> ChatCompletionResponse:
        """解析聊天补全响应 - 增强验证机制"""
        # 运行时验证响应结构
        validation_errors = ResponseValidator.validate_chat_completion_response(data)
        if validation_errors:
            raise ResponseValidationError(
                message=f"Response validation failed: {'; '.join(validation_errors)}"
            )
        
        response_extra = self.extract_extra_fields(data, {"id", "object", "created", "model", "choices", "usage", "system_fingerprint", "service_tier"})
        
        choices = [self.parse_choice(choice_data) for choice_data in data.get("choices", [])]
        usage = self.parse_usage(data.get("usage"))
        
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
    
    def parse_choice(self, choice_data: Dict[str, Any]) -> Choice:
        """解析单个选择项"""
        if not isinstance(choice_data, dict):
            raise ResponseValidationError("Choice must be a dictionary")
        
        known_choice_fields = {"index", "message", "finish_reason", "logprobs"}
        choice_extra = self.extract_extra_fields(choice_data, known_choice_fields)
        
        message_data = choice_data.get("message", {})
        message = self.parse_chat_message(message_data)
        logprobs = self.parse_logprobs(choice_data.get("logprobs"))
        
        return Choice(
            index=choice_data.get("index", 0),
            message=message,
            finish_reason=FinishReason(choice_data.get("finish_reason", "stop")),
            logprobs=logprobs,
            extra=choice_extra
        )
    
    def parse_chat_message(self, message_data: Dict[str, Any]) -> ChatMessage:
        """解析聊天消息"""
        if not isinstance(message_data, dict):
            raise ResponseValidationError("Message must be a dictionary")
        
        known_message_fields = {"role", "content", "tool_calls", "refusal"}
        message_extra = self.extract_extra_fields(message_data, known_message_fields)
        
        tool_calls = self.parse_tool_calls(message_data.get("tool_calls"))
        
        return ChatMessage(
            role=Role(message_data.get("role", "")),
            content=message_data.get("content"),
            tool_calls=tool_calls,
            refusal=message_data.get("refusal"),
            extra=message_extra
        )
    
    def parse_tool_calls(self, tool_calls_data: Optional[List[Dict[str, Any]]]) -> Optional[List[ToolCall]]:
        """解析工具调用列表"""
        if not tool_calls_data:
            return None
            
        tool_calls = []
        for tc_data in tool_calls_data:
            if not isinstance(tc_data, dict):
                raise ResponseValidationError("Tool call must be a dictionary")
            
            known_tc_fields = {"id", "type", "function"}
            tc_extra = self.extract_extra_fields(tc_data, known_tc_fields)
            
            tool_call = ToolCall(
                id=tc_data.get("id", ""),
                type=tc_data.get("type", "function"),
                function=tc_data.get("function", {}),
                extra=tc_extra
            )
            tool_calls.append(tool_call)
        return tool_calls
    
    def parse_logprobs(self, logprobs_data: Optional[Dict[str, Any]]) -> Optional[LogProbs]:
        """解析日志概率"""
        if not logprobs_data:
            return None
        return LogProbs(content=logprobs_data.get("content"))
    
    def parse_usage(self, usage_data: Optional[Dict[str, Any]]) -> Optional[Usage]:
        """解析使用统计"""
        if not usage_data:
            return None
            
        validation_errors = ResponseValidator.validate_usage_response(usage_data)
        if validation_errors:
            raise ResponseValidationError(
                message=f"Usage validation failed: {'; '.join(validation_errors)}"
            )
        
        known_usage_fields = {"prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details", "completion_tokens_details"}
        usage_extra = self.extract_extra_fields(usage_data, known_usage_fields)
        
        return Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            prompt_tokens_details=usage_data.get("prompt_tokens_details"),
            completion_tokens_details=usage_data.get("completion_tokens_details"),
            extra=usage_extra
        )
    
    def parse_chat_completion_chunk(self, data: Dict[str, Any]) -> ChatCompletionChunk:
        """解析流式聊天补全块"""
        known_chunk_fields = {"id", "object", "created", "model", "choices", "system_fingerprint", "service_tier"}
        chunk_extra = self.extract_extra_fields(data, known_chunk_fields)
        
        choices = [self.parse_stream_choice(choice_data) for choice_data in data.get("choices", [])]
        
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
    
    def parse_stream_choice(self, choice_data: Dict[str, Any]) -> StreamChoice:
        """解析流式选择项"""
        known_choice_fields = {"index", "delta", "finish_reason", "logprobs"}
        choice_extra = self.extract_extra_fields(choice_data, known_choice_fields)
        
        delta_data = choice_data.get("delta", {})
        delta = self.parse_delta(delta_data)
        logprobs = self.parse_logprobs(choice_data.get("logprobs"))
        finish_reason = FinishReason(choice_data["finish_reason"]) if choice_data.get("finish_reason") else None
        
        return StreamChoice(
            index=choice_data.get("index", 0),
            delta=delta,
            finish_reason=finish_reason,
            logprobs=logprobs,
            extra=choice_extra
        )
    
    def parse_delta(self, delta_data: Dict[str, Any]) -> Delta:
        """解析增量数据"""
        known_delta_fields = {"role", "content", "tool_calls", "refusal"}
        delta_extra = self.extract_extra_fields(delta_data, known_delta_fields)
        
        tool_calls = self.parse_tool_calls(delta_data.get("tool_calls"))
        
        return Delta(
            role=Role(delta_data["role"]) if delta_data.get("role") else None,
            content=delta_data.get("content"),
            tool_calls=tool_calls,
            refusal=delta_data.get("refusal"),
            extra=delta_extra
        )
    
    def parse_embedding_response(self, data: Dict[str, Any]) -> EmbeddingResponse:
        """解析嵌入响应 - 增强验证机制"""
        # 运行时验证响应结构
        validation_errors = ResponseValidator.validate_embedding_response(data)
        if validation_errors:
            raise ResponseValidationError(
                message=f"Embedding response validation failed: {'; '.join(validation_errors)}"
            )
        
        known_response_fields = {"object", "data", "model", "usage"}
        response_extra = {k: v for k, v in data.items() if k not in known_response_fields}
        
        embeddings = []
        for embedding_data in data.get("data", []):
            if not isinstance(embedding_data, dict):
                raise ResponseValidationError("Embedding must be a dictionary")
            
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
    
    def parse_completion_response(self, data: Dict[str, Any]) -> CompletionResponse:
        """解析旧版补全响应 - 增强验证机制"""
        # 运行时验证响应结构
        validation_errors = ResponseValidator.validate_completion_response(data)
        if validation_errors:
            raise ResponseValidationError(
                message=f"Completion response validation failed: {'; '.join(validation_errors)}"
            )
        
        known_response_fields = {"id", "object", "created", "model", "choices", "usage"}
        response_extra = {k: v for k, v in data.items() if k not in known_response_fields}
        
        choices = []
        for choice_data in data.get("choices", []):
            if not isinstance(choice_data, dict):
                raise ResponseValidationError("Completion choice must be a dictionary")
            
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
    
    def parse_completion_chunk(self, data: Dict[str, Any]) -> 'CompletionChunk':
        """解析流式补全块"""
        from ..entities import CompletionChunk
        
        known_chunk_fields = {"id", "object", "created", "model", "choices"}
        chunk_extra = {k: v for k, v in data.items() if k not in known_chunk_fields}
        
        choices = []
        for choice_data in data.get("choices", []):
            if not isinstance(choice_data, dict):
                raise ResponseValidationError("Completion choice must be a dictionary")
            
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
    
    def parse_models_list(self, data: Dict[str, Any]) -> List[Model]:
        """解析模型列表响应"""
        models = []
        for model_data in data.get("data", []):  # data字段包含模型数组
            if not isinstance(model_data, dict):
                raise ResponseValidationError("Model must be a dictionary")
            
            model = Model(
                id=model_data.get("id", ""),           # 模型标识符，如"gpt-4"、"text-embedding-ada-002"
                created=model_data.get("created", 0),   # 模型发布时间戳
                object=model_data.get("object", "model"), # 对象类型，固定为"model"
                owned_by=model_data.get("owned_by")    # 模型所有者，如"openai"、"anthropic"
            )
            models.append(model)
        
        return models