"""
TinyLMClient 高级特性示例

展示工具调用、JSON格式输出、错误处理等高级功能
前提：已安装 Ollama 并下载 deepseek-r1:1.5b 模型
"""

import asyncio
import json
from typing import Dict, Any
from tiny_lm_client import (
    TinyLMClient,
    ChatCompletionRequest,
    Message,
    Tool,
    ToolChoiceType,
    ResponseFormatType,
    BaseError,
    APIError
)


async def tool_calling_example():
    """工具调用示例"""
    print("=" * 50)
    print("示例 1: 工具调用")
    print("=" * 50)

    # 定义可用工具
    tools = [
        Tool(
            type="function",
            function={
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "温度单位"
                        }
                    },
                    "required": ["city"]
                }
            }
        ),
        Tool(
            type="function",
            function={
                "name": "calculate",
                "description": "执行数学计算",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "数学表达式，如 '2+2*3'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        )
    ]

    async with TinyLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    ) as client:

        # 构建请求，提示模型调用工具
        request = ChatCompletionRequest(
            model="deepseek-r1:1.5b",
            messages=[
                Message(role="user", content="北京现在的天气怎么样？另外，计算一下 3+5*2 等于多少？")
            ],
            tools=tools,
            tool_choice=ToolChoiceType.AUTO,  # 让模型自动决定是否调用工具
            temperature=0.3
        )

        response = await client.chat_completion(request)

        print(f"\n完整响应: {json.dumps(response.choices[0].message.__dict__, default=str, indent=2, ensure_ascii=False)}")

        # 检查是否有工具调用
        if response.choices[0].message.tool_calls:
            print(f"\n模型请求调用工具:")
            for tool_call in response.choices[0].message.tool_calls:
                print(f"  - 工具: {tool_call.function.get('name')}")
                print(f"    参数: {tool_call.function.get('arguments')}")
        else:
            print(f"\n回答: {response.choices[0].message.content}")


async def json_output_example():
    """JSON 格式输出示例"""
    print("\n" + "=" * 50)
    print("示例 2: JSON 格式强制输出")
    print("=" * 50)

    async with TinyLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    ) as client:

        request = ChatCompletionRequest(
            model="deepseek-r1:1.5b",
            messages=[
                Message(
                    role="user",
                    content="请以 JSON 格式返回一个用户信息对象，包含 name、age、city 三个字段。"
                )
            ],
            response_format=ResponseFormatType.JSON_OBJECT,
            temperature=0.5
        )

        response = await client.chat_completion(request)

        print(f"\nJSON 输出:")
        try:
            json_data = json.loads(response.choices[0].message.content)
            print(json.dumps(json_data, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(response.choices[0].message.content)


async def error_handling_example():
    """错误处理示例"""
    print("\n" + "=" * 50)
    print("示例 3: 错误处理")
    print("=" * 50)

    async with TinyLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        max_retries=2,
        timeout=10.0
    ) as client:

        # 示例 1: 处理无效模型错误
        print("\n测试 1: 无效模型名称")
        request = ChatCompletionRequest(
            model="non-existent-model",
            messages=[Message(role="user", content="Hello")],
            max_tokens=50
        )

        try:
            response = await client.chat_completion(request)
            print(f"响应: {response.choices[0].message.content}")
        except APIError as e:
            print(f"API 错误:")
            print(f"  - 类型: {e.type}")
            print(f"  - 代码: {e.code}")
            print(f"  - 消息: {e.message}")
        except BaseError as e:
            print(f"错误: {e.message}")

        # 示例 2: 处理网络错误
        print("\n测试 2: 超时设置")
        long_request = ChatCompletionRequest(
            model="deepseek-r1:1.5b",
            messages=[Message(role="user", content="请详细解释量子计算原理")],
            max_tokens=1000
        )

        try:
            # 使用较短的超时时间模拟超时场景
            response = await client.chat_completion(long_request)
            print(f"响应获取成功")
        except Exception as e:
            print(f"请求错误: {e}")


async def advanced_parameters_example():
    """高级参数配置示例"""
    print("\n" + "=" * 50)
    print("示例 4: 高级参数配置")
    print("=" * 50)

    async with TinyLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    ) as client:

        # 使用多种高级参数
        request = ChatCompletionRequest(
            model="deepseek-r1:1.5b",
            messages=[
                Message(role="system", content="你是一个创意写作助手。"),
                Message(role="user", content="写一个关于人工智能的短故事。")
            ],
            # 高级参数
            temperature=0.9,           # 较高的温度，增加创造性
            top_p=0.95,               # 核采样，控制多样性
            max_tokens=200,           # 限制输出长度
            frequency_penalty=0.5,     # 频率惩罚，减少重复
            presence_penalty=0.3,     # 存在惩罚，鼓励新话题
            stop=["\n\n", "故事结束"],  # 停止序列
            n=1,                      # 生成1个候选
            seed=42                   # 固定随机种子，便于复现
        )

        response = await client.chat_completion(request)
        print(f"\n生成的故事:\n{response.choices[0].message.content}")


async def multiple_outputs_example():
    """多个输出候选示例"""
    print("\n" + "=" * 50)
    print("示例 5: 多个输出候选")
    print("=" * 50)

    async with TinyLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    ) as client:

        request = ChatCompletionRequest(
            model="deepseek-r1:1.5b",
            messages=[
                Message(role="user", content="用一句话描述夏天。")
            ],
            temperature=1.0,
            n=3,  # 生成3个不同的候选
            max_tokens=50
        )

        response = await client.chat_completion(request)

        print(f"\n生成了 {len(response.choices)} 个候选:\n")
        for i, choice in enumerate(response.choices, 1):
            print(f"候选 {i}: {choice.message.content}")
            print()


async def main():
    """运行所有高级示例"""
    print("\nTinyLMClient 高级特性示例\n")
    print("请确保 Ollama 已运行并下载了 deepseek-r1:1.5b 模型\n")

    try:
        await tool_calling_example()
        await json_output_example()
        await error_handling_example()
        await advanced_parameters_example()
        await multiple_outputs_example()

        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        print("=" * 50)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
