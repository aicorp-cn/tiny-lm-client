# TinyLMClient 示例代码

本目录包含 TinyLMClient 的完整示例代码，展示如何使用项目的核心功能。

## 前提条件

确保已安装 Ollama 并下载所需模型：

```bash
# 安装 Ollama（如果尚未安装）
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh
# Windows: 从 https://ollama.com 下载安装包

# 启动 Ollama 服务
ollama serve

# 下载基础聊天模型
ollama pull deepseek-r1:1.5b

# 可选：下载嵌入模型（用于嵌入示例）
ollama pull nomic-embed-text
```

## 示例列表

### 1. `basic_usage.py` - 基础用法

展示 TinyLMClient 的核心功能：

- **基础聊天补全**：发送简单问题并获得回答
- **多轮对话**：进行上下文相关的连续对话
- **流式聊天**：实时显示模型生成的内容
- **获取模型列表**：查询可用的模型

**运行**：
```bash
cd examples
python basic_usage.py
```

**适用场景**：新手入门、快速体验核心功能

---

### 2. `advanced_features.py` - 高级特性

展示 TinyLMClient 的高级功能：

- **工具调用**：定义工具并让模型决定何时调用
- **JSON 格式输出**：强制模型返回结构化 JSON 数据
- **错误处理**：处理各种 API 错误和网络异常
- **高级参数配置**：使用 temperature、top_p、penalty 等参数
- **多个输出候选**：生成多个不同的候选回答

**运行**：
```bash
python advanced_features.py
```

**适用场景**：需要更精细控制的开发场景

---

### 3. `embeddings_example.py` - 文本嵌入

展示文本嵌入功能的应用：

- **基础文本嵌入**：将文本转换为向量表示
- **批量嵌入**：高效处理多个文本
- **相似度计算**：计算文本之间的语义相似度
- **语义搜索**：基于语义相似度的文档检索
- **文本聚类**：使用嵌入向量对文本进行聚类

**运行**：
```bash
python embeddings_example.py
```

**注意**：需要 `nomic-embed-text` 模型：
```bash
ollama pull nomic-embed-text
```

**适用场景**：搜索、推荐、聚类、语义分析

---

### 4. `real_world_scenarios.py` - 实际应用场景

展示真实项目中的应用：

- **代码助手**：代码生成和解释
- **文本摘要**：长文本自动摘要
- **问答系统**：基于知识库的问答
- **翻译助手**：多语言翻译
- **创意写作**：故事、诗歌、剧本创作
- **对话机器人**：自然对话交互
- **数据格式化**：非结构化数据转 JSON

**运行**：
```bash
python real_world_scenarios.py
```

**适用场景**：学习如何在实际项目中集成 TinyLMClient

---

## 项目结构说明

```
examples/
├── README.md                      # 本文档
├── basic_usage.py                 # 基础用法示例
├── advanced_features.py           # 高级特性示例
├── embeddings_example.py          # 文本嵌入示例
└── real_world_scenarios.py       # 实际应用场景示例
```

---

## 代码模式

### 推荐的使用模式

```python
import asyncio
from tiny_lm_client import TinyLMClient, ChatCompletionRequest, Message

async def example():
    # 使用 async with 确保资源正确释放
    async with TinyLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        max_retries=3,
        timeout=60.0
    ) as client:
        # 构建请求
        request = ChatCompletionRequest(
            model="deepseek-r1:1.5b",
            messages=[
                Message(role="user", content="你好！")
            ],
            temperature=0.7,
            max_tokens=150
        )

        # 发送请求
        response = await client.chat_completion(request)

        # 处理响应
        print(response.choices[0].message.content)

asyncio.run(example())
```

### 流式输出模式

```python
# 启用流式输出
request = ChatCompletionRequest(
    model="deepseek-r1:1.5b",
    messages=[Message(role="user", content="写一个故事")],
    stream=True
)

# 逐个处理流式响应块
async for chunk in await client.chat_completion(request):
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

---

## 核心概念

### 1. 聊天补全 (Chat Completion)
- 主要的文本生成接口
- 支持多轮对话和工具调用
- 可流式或非流式输出

### 2. 文本嵌入 (Embeddings)
- 将文本转换为高维向量
- 用于语义搜索、相似度计算
- 支持批量处理

### 3. 模型列表 (Models List)
- 获取可用模型信息
- 动态发现服务能力

### 4. 错误处理
- `BaseError`：所有错误的基类
- `APIError`：API 调用错误
- `ValidationError`：验证错误

---

## 参数说明

### 常用请求参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | str | 必需 | 模型名称 |
| `messages` | List[Message] | 必需 | 对话历史 |
| `temperature` | float | 1.0 | 采样温度 (0-2) |
| `max_tokens` | int | None | 最大生成 token 数 |
| `stream` | bool | False | 是否流式输出 |
| `top_p` | float | None | 核采样阈值 |
| `tools` | List[Tool] | None | 工具定义列表 |

### 高级参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `frequency_penalty` | float | 频率惩罚，减少重复 |
| `presence_penalty` | float | 存在惩罚，鼓励新话题 |
| `stop` | Union[str, List[str]] | 停止序列 |
| `seed` | int | 随机种子，便于复现 |
| `response_format` | ResponseFormatType | 响应格式 |

---

## 故障排除

### 问题 1: 连接超时

```
连接错误: Cannot connect to host localhost:11434
```

**解决方法**：确保 Ollama 正在运行
```bash
ollama serve
```

### 问题 2: 模型不存在

```
API 错误: model 'xxx' not found
```

**解决方法**：下载所需模型
```bash
ollama pull deepseek-r1:1.5b
```

### 问题 3: 嵌入功能不支持

```
该模型不支持嵌入功能
```

**解决方法**：使用支持嵌入的模型
```bash
ollama pull nomic-embed-text
```

---

## 进阶学习

1. **深入学习 API**：阅读项目根目录的 `README.md`
2. **查看源码**：了解核心实现细节
3. **自定义扩展**：根据需求修改示例代码

---

## 贡献

欢迎提交 PR 添加更多示例！

---

**更新日期**：2026-01-03
