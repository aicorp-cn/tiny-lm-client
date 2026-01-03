from dataclasses import dataclass, field
from typing import List, Dict, Any

from .completion_choice import CompletionChoice


@dataclass
class CompletionChunk:
    """旧版文本补全流式响应块，对应/v1/completions的流式数据块
    
    在stream=True模式下，服务端会分多次发送这种数据块，
    每个块包含增量生成的文本内容。客户端需要拼接所有块获得完整结果。
    
    Attributes:
        id: 响应唯一标识符(与完整响应相同)
        object: 对象类型标识，固定为"text_completion"
        created: 响应创建时间戳
        model: 实际处理请求的模型名称
        choices: 流式生成结果列表，每个包含增量文本内容
        extra: 扩展字段，保留API可能新增的响应字段
    
    Stream Format:
        数据流格式: data: {JSON_CHUNK}\n\n
        data: [DONE]\n\n 表示流结束
    
    Note:
        流式响应显著降低感知延迟，适合实时显示生成内容的场景
        每个chunk的choices只包含增量变化，不是完整状态
    """
    id: str
    object: str = "text_completion"
    created: int = 0
    model: str = ""
    choices: List[CompletionChoice] = field(default_factory=list)
    # 扩展字段，用于保留未知参数，确保API兼容性
    extra: Dict[str, Any] = field(default_factory=dict)