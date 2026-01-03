from dataclasses import dataclass, field
from typing import Union, List, Optional, Dict, Any


@dataclass
class CompletionRequest:
    """旧版文本补全请求配置类，对应/v1/completions端点
    
    这是传统的文本生成API，适用于简单的文本续写、代码补全等场景。
    相比ChatCompletion，不支持多轮对话和工具调用，但支持更多传统生成参数。
    
    Attributes:
        model: 模型标识符，如"davinci"、"curie"等
        prompt: 输入提示，可以是字符串、字符串列表或token ID列表
        suffix: 追加到生成文本后的后缀内容
        max_tokens: 生成的最大token数
        temperature: 采样温度，控制随机性
        top_p: 核采样概率阈值
        n: 为每个提示生成的完成数量
        stream: 是否启用流式响应
        logprobs: 返回每个token的对数概率(top_logprobs个最可能的token)
        echo: 是否在响应中包含输入prompt
        stop: 停止序列数组
        presence_penalty: 存在惩罚系数
        frequency_penalty: 频率惩罚系数
        best_of: 从多个候选中选择最佳的生成数量(服务端)
        logit_bias: token级别的logits修正
        user: 用户标识
        extra: 扩展字段，保留API可能新增的未知参数
    
    Note:
        这是OpenAI兼容的旧版API，建议新项目使用ChatCompletion
        best_of与n参数配合使用时，服务端会生成best_of*n个候选
    """
    model: str
    prompt: Union[str, List[str], List[int], List[List[int]]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: bool = False
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    best_of: Optional[int] = None
    logit_bias: Optional[Dict[int, float]] = None
    user: Optional[str] = None
    # 扩展字段，用于保留未知参数，确保API兼容性
    extra: Dict[str, Any] = field(default_factory=dict)