from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Usage:
    """令牌使用统计类，记录API调用的token消耗详情
    
    用于计费和性能监控，精确追踪输入、输出和总计的token数量。
    某些模型还提供更详细的分类统计，如提示缓存命中、推理加速等。
    
    Attributes:
        prompt_tokens: 输入提示消耗的token数量
        completion_tokens: 模型生成回复消耗的token数量
        total_tokens: 本次调用消耗的总token数量(prompt + completion)
        prompt_tokens_details: 提示token的详细分类，如{
            "cached_tokens": 0,      # 缓存命中的token数
            "audio_tokens": 0,       # 音频输入token数
            "text_tokens": 100       # 文本输入token数
        }
        completion_tokens_details: 生成token的详细分类，如{
            "reasoning_tokens": 0,   # 推理过程token数
            "accepted_prediction_tokens": 50,  # 接受的预测token数
            "rejected_prediction_tokens": 10   # 拒绝的预测token数
        }
        extra: 扩展字段，保留API可能新增的使用统计维度
    
    Note:
        不同模型提供商可能返回不同的details字段，通过extra保证兼容性
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[Dict[str, int]] = None
    completion_tokens_details: Optional[Dict[str, int]] = None
    # 扩展字段，用于保留未知参数，确保API向前兼容
    extra: Dict[str, Any] = field(default_factory=dict)