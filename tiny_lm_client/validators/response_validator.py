from typing import Any, Dict, List
from tiny_lm_client.validators.validation_utils import ValidationUtils


class ResponseValidator:
    """响应验证器 - 专门验证API响应数据"""
    
    @staticmethod
    def validate_chat_completion_response(data: Dict[str, Any]) -> List[str]:
        """验证聊天补全响应"""
        errors = []
        
        # 验证必需字段
        errors.extend(ValidationUtils.validate_required_fields(data, ["id", "object", "created", "model", "choices"]))
        
        # 验证字段类型
        errors.extend(ValidationUtils.validate_field_type(data, "id", str))
        errors.extend(ValidationUtils.validate_field_type(data, "object", str))
        errors.extend(ValidationUtils.validate_field_type(data, "created", int))
        errors.extend(ValidationUtils.validate_field_type(data, "model", str))
        errors.extend(ValidationUtils.validate_field_type(data, "choices", list))
        
        # 验证choices数组
        choices = data.get("choices", [])
        if isinstance(choices, list):
            for i, choice in enumerate(choices):
                if isinstance(choice, dict):
                    errors.extend(ValidationUtils.validate_field_type(choice, "index", int))
                    errors.extend(ValidationUtils.validate_field_type(choice, "message", dict))
        
        # 验证usage字段（如果存在）
        if "usage" in data:
            usage_errors = ResponseValidator.validate_usage_response(data["usage"])
            errors.extend(usage_errors)
        
        return errors
    
    @staticmethod
    def validate_embedding_response(data: Dict[str, Any]) -> List[str]:
        """验证嵌入响应"""
        errors = []
        
        # 验证必需字段
        errors.extend(ValidationUtils.validate_required_fields(data, ["object", "data", "model"]))
        
        # 验证字段类型
        errors.extend(ValidationUtils.validate_field_type(data, "object", str))
        errors.extend(ValidationUtils.validate_field_type(data, "data", list))
        errors.extend(ValidationUtils.validate_field_type(data, "model", str))
        
        # 验证data数组中的每个嵌入
        embeddings = data.get("data", [])
        if isinstance(embeddings, list):
            for i, embedding in enumerate(embeddings):
                if isinstance(embedding, dict):
                    errors.extend(ValidationUtils.validate_field_type(embedding, "index", int))
                    errors.extend(ValidationUtils.validate_field_type(embedding, "vector", list))
        
        return errors
    
    @staticmethod
    def validate_completion_response(data: Dict[str, Any]) -> List[str]:
        """验证补全响应"""
        errors = []
        
        # 验证必需字段
        errors.extend(ValidationUtils.validate_required_fields(data, ["id", "object", "created", "model", "choices"]))
        
        # 验证字段类型
        errors.extend(ValidationUtils.validate_field_type(data, "id", str))
        errors.extend(ValidationUtils.validate_field_type(data, "object", str))
        errors.extend(ValidationUtils.validate_field_type(data, "created", int))
        errors.extend(ValidationUtils.validate_field_type(data, "model", str))
        errors.extend(ValidationUtils.validate_field_type(data, "choices", list))
        
        return errors
    
    @staticmethod
    def validate_usage_response(usage_data: Dict[str, Any]) -> List[str]:
        """验证使用统计响应"""
        errors = []
        
        if isinstance(usage_data, dict):
            errors.extend(ValidationUtils.validate_field_type(usage_data, "prompt_tokens", int))
            errors.extend(ValidationUtils.validate_field_type(usage_data, "completion_tokens", int))
            errors.extend(ValidationUtils.validate_field_type(usage_data, "total_tokens", int))
        
        return errors