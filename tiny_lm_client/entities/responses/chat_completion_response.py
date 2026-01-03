from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .. import Choice, Usage


@dataclass
class ChatCompletionResponse:
    """聊天补全响应类，封装/chat/completions端点的完整响应数据
    
    包含模型生成的回复内容、使用统计和系统信息等。
    支持流式和非流式两种响应模式的解析。
    
    Attributes:
        id: 响应唯一标识符，用于调试和日志记录
        object: 对象类型标识，固定为"chat.completion"
        created: 响应创建时间戳(Unix epoch seconds)
        model: 实际处理请求的模型名称(可能与请求model不同)
        choices: 生成结果列表，当n>1时包含多个独立回复
        usage: 令牌使用统计，用于计费和性能监控
        system_fingerprint: 系统指纹，标识后端配置版本
        service_tier: 服务等级，指示使用的计算资源级别
        extra: 扩展字段，保留API可能新增的响应字段
    
    Note:
        - 流式响应时，每个数据块也是ChatCompletionChunk类型
        - system_fingerprint可用于检测后端模型更新
        - service_tier反映响应速度和成本等级(如"default"、"premium")
    """
    id: str
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[Choice] = field(default_factory=list)
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None
    # 扩展字段，用于保留未知参数，确保API向前兼容
    extra: Dict[str, Any] = field(default_factory=dict)