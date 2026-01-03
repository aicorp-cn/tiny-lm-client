from typing import Any, Dict, List, Optional

class ValidationUtils:
    """验证工具类 - 提供可复用的验证组件"""
    
    @staticmethod
    def validate_type(value: Any, expected_type: type, field_name: str) -> List[str]:
        """验证值的类型"""
        errors = []
        if value is not None and not isinstance(value, expected_type):
            errors.append(f"Field '{field_name}' expected {expected_type.__name__}, got {type(value).__name__}")
        return errors
    
    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
        """验证必需字段"""
        errors = []
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        return errors
    
    @staticmethod
    def validate_field_type(data: Dict[str, Any], field: str, expected_type: type) -> List[str]:
        """验证字段类型"""
        if field in data:
            return ValidationUtils.validate_type(data[field], expected_type, field)
        return []
    
    @staticmethod
    def validate_range(value: Any, min_val: Optional[float] = None, max_val: Optional[float] = None, field_name: str = "") -> List[str]:
        """验证数值范围"""
        errors = []
        if value is not None and isinstance(value, (int, float)):
            if min_val is not None and value < min_val:
                errors.append(f"Field '{field_name}' value {value} is less than minimum {min_val}")
            if max_val is not None and value > max_val:
                errors.append(f"Field '{fieldname}' value {value} is greater than maximum {max_val}")
        return errors
    
    @staticmethod
    def validate_string_length(value: Any, min_length: int = 0, max_length: Optional[int] = None, field_name: str = "") -> List[str]:
        """验证字符串长度"""
        errors = []
        if isinstance(value, str):
            if len(value) < min_length:
                errors.append(f"Field '{field_name}' length {len(value)} is less than minimum {min_length}")
            if max_length is not None and len(value) > max_length:
                errors.append(f"Field '{field_name}' length {len(value)} is greater than maximum {max_length}")
        return errors