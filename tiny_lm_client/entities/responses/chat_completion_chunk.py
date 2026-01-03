from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .stream_choice import StreamChoice


@dataclass
class ChatCompletionChunk:
    """聊天补全流式响应块类，对应/chat/completions的流式数据块
    
    在stream=True模式下，服务端分多次发送这种数据块，每个块包含
    一个或多个选择项的增量更新。客户端需要累积所有块来重建完整响应。
    
    Attributes:
        id: 响应唯一标识符(整个流式会话保持不变)
        object: 对象类型标识，固定为"chat.completion.chunk"
        created: 响应创建时间戳
        model: 实际处理请求的模型名称
        choices: 流式选择项列表，每个包含增量更新
        system_fingerprint: 系统指纹，标识后端配置版本
        service_tier: 服务等级信息
        extra: 扩展字段，保留API可能新增的响应字段
    
    Stream Format:
        数据流格式: data: {JSON_CHUNK}\n\n
        data: [DONE]\n\n 表示流结束
    
    Note:
        这是现代chat completion API的流式格式，比旧版completion更丰富
        支持工具调用的增量返回和多角色消息的流式构建
    """
    id: str
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: List[StreamChoice] = field(default_factory=list)
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None
    # 扩展字段，用于保留未知参数，确保API向前兼容
    extra: Dict[str, Any] = field(default_factory=dict)