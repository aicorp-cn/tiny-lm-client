from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class LogProbs:
    """Token对数概率信息类，用于分析模型输出的置信度
    
    包含模型生成每个token时的对数概率值，用于调试、分析和置信度评估。
    仅在请求参数logprobs=True时返回。
    
    Attributes:
        content: Token概率列表，每个元素包含token、logprob、bytes等信息
                 格式: [{"token": "hello", "logprob": -0.5, "bytes": [104, 101, 108, 108, 111]}, ...]
    """
    content: Optional[List[Dict[str, Any]]] = None