from typing import Any, Dict, List
from tiny_lm_client.validators.validation_utils import ValidationUtils


class RequestValidator:
    """请求验证器 - 专门验证API请求参数"""
    
    @staticmethod
    def validate_chat_completion_request(request_data: Dict[str, Any]) -> List[str]:
        """验证聊天补全请求"""
        errors = []
        
        # 验证必需字段
        errors.extend(ValidationUtils.validate_required_fields(request_data, ["model", "messages"]))
        
        # 验证字段类型
        errors.extend(ValidationUtils.validate_field_type(request_data, "model", str))
        errors.extend(ValidationUtils.validate_field_type(request_data, "messages", list))
        
        # 验证数值字段范围
        errors.extend(ValidationUtils.validate_range(
            request_data.get("temperature"), min_val=0.0, max_val=2.0, field_name="temperature"
        ))
        errors.extend(ValidationUtils.validate_range(
            request_data.get("top_p"), min_val=0.0, max_val=1.0, field_name="top_p"
        ))
        errors.extend(ValidationUtils.validate_range(
            request_data.get("frequency_penalty"), min_val=-2.0, max_val=2.0, field_name="frequency_penalty"
        ))
        errors.extend(ValidationUtils.validate_range(
            request_data.get("presence_penalty"), min_val=-2.0, max_val=2.0, field_name="presence_penalty"
        ))
        
        return errors
    
    @staticmethod
    def validate_embedding_request(request_data: Dict[str, Any]) -> List[str]:
        """验证嵌入请求"""
        errors = []
        
        # 验证必需字段
        errors.extend(ValidationUtils.validate_required_fields(request_data, ["model", "input"]))
        
        # 验证字段类型
        errors.extend(ValidationUtils.validate_field_type(request_data, "model", str))
        errors.extend(ValidationUtils.validate_field_type(request_data, "input", (str, list)))
        
        return errors
    
    @staticmethod
    def validate_completion_request(request_data: Dict[str, Any]) -> List[str]:
        """验证补全请求"""
        errors = []
        
        # 验证必需字段
        errors.extend(ValidationUtils.validate_required_fields(request_data, ["model", "prompt"]))
        
        # 验证字段类型
        errors.extend(ValidationUtils.validate_field_type(request_data, "model", str))
        errors.extend(ValidationUtils.validate_field_type(request_data, "prompt", (str, list)))
        
        # 验证数值字段范围
        errors.extend(ValidationUtils.validate_range(
            request_data.get("temperature"), min_val=0.0, max_val=2.0, field_name="temperature"
        ))
        
        return errors