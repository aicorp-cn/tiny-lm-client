from enum import Enum

class EncodingType(str, Enum):
    """HTTP响应内容编码类型，用于Accept-Encoding头部"""
    
    GZIP = "gzip"      # GZIP压缩，适用于文本数据压缩
    DEFLATE = "deflate" # DEFLATE压缩算法
    COMPRESS = "compress" # UNIX compress压缩