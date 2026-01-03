import re
import socket
from urllib.parse import urlparse
from tiny_lm_client.errors import BaseError, RequestValidationError


class ParamsValidator:
    """参数验证器 - 专门处理参数验证，遵循单一职责原则"""
    
    @staticmethod
    def validate_base_url(base_url: str) -> None:
        """验证base_url参数的安全性"""
        # 基本URL格式验证
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|/\S*)$', re.IGNORECASE)  # path

        parsed = urlparse(base_url)
        if not url_pattern.match(base_url) or not parsed.netloc:
            raise RequestValidationError(f"Invalid base_url format: {base_url}")
            
        # 防止SSRF: 检查是否为内网地址
        hostname = parsed.hostname or parsed.netloc.split(':')[0]
        if ParamsValidator.is_internal_address(hostname):
            raise RequestValidationError(f"SSRF protection: Internal addresses not allowed: {hostname}")
    
    @staticmethod
    def is_internal_address(hostname: str) -> bool:
        """检查是否为内网地址以防止SSRF，但允许localhost用于本地开发"""
        # 允许localhost用于本地开发场景
        if hostname == 'localhost':
            return False
        try:
            ip = socket.gethostbyname(hostname)
            return ip.startswith(('10.', '172.', '192.168.', '127.', '0.')) or ip == '::1'
        except socket.gaierror:
            return hostname.startswith(('10.', '172.', '192.168.', '127.', '0.'))
        
    @staticmethod
    def mask_sensitive_data(data: str) -> str:
        """敏感数据掩码处理"""
        if len(data) <= 8:
            return "***"
        return data[:3] + "***" + data[-3:]
    
    @staticmethod
    def validate_api_key(api_key: str) -> None:
        """验证API密钥"""
        if not api_key or not isinstance(api_key, str):
            raise RequestValidationError("API key cannot be empty or None")
        
        # 可选：验证API密钥格式（根据实际API提供商的要求）
        # 对于Ollama等本地服务，允许较短的API密钥
        if len(api_key) < 10 and api_key != "ollama":
            raise RequestValidationError("API key appears to be too short")