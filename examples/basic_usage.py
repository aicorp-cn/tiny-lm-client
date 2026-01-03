"""
TinyLMClient 基础用法示例

展示如何使用 TinyLMClient 与本地 Ollama 模型进行交互
前提：已安装 Ollama 并下载 deepseek-r1:1.5b 模型
"""

import asyncio
from tiny_lm_client import TinyLMClient, ChatCompletionRequest, Message


async def basic_chat_example():
    """基础聊天补全示例"""
    print("=" * 50)
    print("示例 1: 基础聊天补全")
    print("=" * 50)

    # 初始化客户端（连接本地 Ollama）
    async with TinyLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        max_retries=2,
        timeout=60.0
    ) as client:

        # 构建聊天请求
        request = ChatCompletionRequest(
            model="deepseek-r1:1.5b",
            messages=[
                Message(role="user", content="你好！请简单介绍一下你自己。")
            ],
            temperature=0.7,
            max_tokens=200
        )

        # 发送请求并获取响应
        response = await client.chat_completion(request)

        # 打印响应
        print(f"\n模型: {response.model}")
        print(f"回答: {response.choices[0].message.content}")

        # 打印使用统计
        if response.usage:
            print(f"\nToken 使用统计:")
            print(f"  - 输入 tokens: {response.usage.prompt_tokens}")
            print(f"  - 输出 tokens: {response.usage.completion_tokens}")
            print(f"  - 总计 tokens: {response.usage.total_tokens}")


async def multi_turn_chat_example():
    """多轮对话示例"""
    print("\n" + "=" * 50)
    print("示例 2: 多轮对话")
    print("=" * 50)

    async with TinyLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    ) as client:

        # 构建多轮对话
        conversation = [
            Message(role="system", content="你是一个乐于助人的AI助手。"),
            Message(role="user", content="什么是Python？"),
            Message(role="assistant", content="Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。"),
            Message(role="user", content="Python有哪些主要特点？")
        ]

        request = ChatCompletionRequest(
            model="deepseek-r1:1.5b",
            messages=conversation,
            temperature=0.6,
            max_tokens=150
        )

        response = await client.chat_completion(request)
        print(f"\n回答: {response.choices[0].message.content}")


async def stream_chat_example():
    """流式聊天示例"""
    print("\n" + "=" * 50)
    print("示例 3: 流式聊天")
    print("=" * 50)

    async with TinyLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    ) as client:

        request = ChatCompletionRequest(
            model="deepseek-r1:1.5b",
            messages=[
                Message(role="user", content="请写一首关于春天的短诗。")
            ],
            stream=True,  # 启用流式输出
            temperature=0.8
        )

        print("\n流式输出: ", end="", flush=True)

        # 逐个处理流式响应块
        async for chunk in await client.chat_completion(request):
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)

        print("\n\n流式输出完成！")


async def get_models_example():
    """获取可用模型列表"""
    print("\n" + "=" * 50)
    print("示例 4: 获取可用模型列表")
    print("=" * 50)

    async with TinyLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    ) as client:

        models = await client.models_list()

        print(f"\n可用模型数量: {len(models)}")
        print("\n模型列表:")
        for model in models:
            print(f"  - {model.id} (所有者: {model.owned_by})")


async def main():
    """运行所有示例"""
    print("\nTinyLMClient 基础用法示例\n")
    print("请确保 Ollama 已运行并下载了 deepseek-r1:1.5b 模型\n")

    try:
        await basic_chat_example()
        await multi_turn_chat_example()
        await stream_chat_example()
        await get_models_example()

        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        print("=" * 50)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
