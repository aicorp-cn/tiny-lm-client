#!/usr/bin/env python3
"""
TinyLMClient 快速测试脚本

快速验证 TinyLMClient 和 Ollama 环境是否正常工作
"""

import asyncio
import sys
from tiny_lm_client import TinyLMClient, ChatCompletionRequest, Message


async def test_basic_connection():
    """测试基本连接和聊天功能"""
    print("=" * 60)
    print("TinyLMClient 快速测试")
    print("=" * 60)
    print()

    try:
        print("1. 正在连接到 Ollama 服务...")

        async with TinyLMClient(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            timeout=30.0
        ) as client:

            print("✓ 连接成功！\n")

            print("2. 获取可用模型列表...")
            models = await client.models_list()
            print(f"✓ 找到 {len(models)} 个模型:")
            for model in models:
                print(f"  - {model.id}")
            print()

            # 检查 deepseek-r1:1.5b 模型
            model_id = "deepseek-r1:1.5b"
            has_model = any(model.id == model_id for model in models)

            if not has_model:
                print(f"⚠️  警告: 未找到模型 '{model_id}'")
                print(f"请运行: ollama pull {model_id}\n")
                return False

            print(f"3. 使用模型 '{model_id}' 进行聊天测试...\n")

            request = ChatCompletionRequest(
                model=model_id,
                messages=[
                    Message(role="user", content="你好！请用一句话介绍你自己。")
                ],
                temperature=0.7,
                max_tokens=100
            )

            print("等待回复...")
            response = await client.chat_completion(request)

            print("\n" + "=" * 60)
            print("模型回复:")
            print("=" * 60)
            print(response.choices[0].message.content)
            print("=" * 60)
            print()

            if response.usage:
                print("Token 使用:")
                print(f"  输入: {response.usage.prompt_tokens}")
                print(f"  输出: {response.usage.completion_tokens}")
                print(f"  总计: {response.usage.total_tokens}")
                print()

            print("=" * 60)
            print("✓ 所有测试通过！")
            print("=" * 60)
            return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        print()

        print("故障排除:")
        print("-" * 60)
        print("1. 确保 Ollama 正在运行:")
        print("   ollama serve")
        print()
        print("2. 确保已下载模型:")
        print("   ollama pull deepseek-r1:1.5b")
        print()
        print("3. 检查 Ollama 是否正常响应:")
        print("   curl http://localhost:11434/api/tags")
        print()
        print("4. 确保已安装依赖:")
        print("   pip install httpx")
        print("-" * 60)

        import traceback
        print("\n详细错误信息:")
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print()

    result = asyncio.run(test_basic_connection())

    if result:
        print("\n✓ 环境配置正确，可以开始使用 TinyLMClient！")
        print()
        print("下一步:")
        print("  - 运行示例: python examples/run_examples.py")
        print("  - 查看文档: examples/README.md")
        sys.exit(0)
    else:
        print("\n✗ 环境配置有问题，请按照上面的提示进行修复")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已中断测试")
        sys.exit(0)
