from enum import Enum

class ContentType(str, Enum):
    """多模态内容类型枚举，支持文本、图像、音频等多种输入格式"""
    
    TEXT = "text"               # 文本内容
    IMAGE_URL = "image_url"     # 图片URL引用
    IMAGE_BASE64 = "image_base64" # Base64编码的图片数据
    AUDIO = "audio"             # 音频数据