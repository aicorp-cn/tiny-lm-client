#!/usr/bin/env python3
"""
TinyLMClient 示例启动器

快速运行各种示例的便捷脚本
"""

import sys
import asyncio
from pathlib import Path


def print_banner():
    """打印横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║         TinyLMClient 示例启动器                              ║
║                                                              ║
║   轻量级 OpenAI 兼容大模型客户端示例演示                       ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_menu():
    """打印菜单"""
    menu = """
请选择要运行的示例：

  1. 基础用法 (basic_usage.py)
     - 基础聊天补全
     - 多轮对话
     - 流式聊天
     - 获取模型列表

  2. 高级特性 (advanced_features.py)
     - 工具调用
     - JSON 格式输出
     - 错误处理
     - 高级参数配置
     - 多个输出候选

  3. 文本嵌入 (embeddings_example.py)
     - 基础文本嵌入
     - 批量嵌入
     - 相似度计算
     - 语义搜索
     - 文本聚类
     ⚠️  需要 nomic-embed-text 模型

  4. 实际应用场景 (real_world_scenarios.py)
     - 代码助手
     - 文本摘要
     - 问答系统
     - 翻译助手
     - 创意写作
     - 对话机器人
     - 数据格式化

  5. 运行所有示例

  0. 退出

"""
    print(menu, end="")


async def run_example(example_file: str, example_name: str):
    """运行指定的示例"""
    print(f"\n{'=' * 60}")
    print(f"正在运行: {example_name}")
    print(f"文件: {example_file}")
    print('=' * 60, end="\n\n")

    try:
        # 动态导入示例模块
        spec = __import__(example_file).__spec__
        module = __import__(spec.name)

        # 运行示例的 main 函数
        await module.main()

    except ImportError:
        print(f"错误: 无法导入 {example_file}")
        print(f"请确保文件存在于 examples 目录中")

    except Exception as e:
        print(f"\n运行示例时出错: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"{example_name} 运行完成")
    print('=' * 60, end="\n\n")


def check_ollama():
    """检查 Ollama 是否已安装"""
    import subprocess

    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✓ Ollama 已安装: {version}")
            return True
        else:
            print("✗ Ollama 命令执行失败")
            return False
    except FileNotFoundError:
        print("✗ Ollama 未安装")
        return False
    except subprocess.TimeoutExpired:
        print("✗ Ollama 命令超时")
        return False
    except Exception as e:
        print(f"✗ 检查 Ollama 时出错: {e}")
        return False


def check_model(model_name: str):
    """检查模型是否已下载"""
    import subprocess

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            models = result.stdout
            if model_name in models:
                print(f"✓ 模型 '{model_name}' 已下载")
                return True
            else:
                print(f"✗ 模型 '{model_name}' 未下载")
                return False
        else:
            print("✗ 无法获取模型列表")
            return False
    except Exception as e:
        print(f"✗ 检查模型时出错: {e}")
        return False


def print_system_check():
    """打印系统检查结果"""
    print("\n系统检查:")
    print("-" * 60)

    ollama_ok = check_ollama()
    if ollama_ok:
        check_model("deepseek-r1:1.5b")
        check_model("nomic-embed-text")

    print("-" * 60)
    print()


def print_setup_instructions():
    """打印设置说明"""
    instructions = """
如果未安装 Ollama 或缺少模型，请按照以下步骤设置：

1. 安装 Ollama:
   - macOS:   brew install ollama
   - Linux:   curl -fsSL https://ollama.com/install.sh | sh
   - Windows: 从 https://ollama.com 下载安装包

2. 启动 Ollama 服务:
   ollama serve

3. 下载模型:
   ollama pull deepseek-r1:1.5b
   ollama pull nomic-embed-text   # 用于嵌入示例

更多信息请参考: https://ollama.com
"""
    print(instructions)


async def run_all_examples():
    """运行所有示例"""
    examples = [
        ("examples.basic_usage", "基础用法"),
        ("examples.advanced_features", "高级特性"),
        ("examples.real_world_scenarios", "实际应用场景"),
    ]

    print("\n准备运行所有示例...\n")

    for module_name, title in examples:
        await run_example(module_name, title)

        print("\n按 Enter 继续下一个示例，或输入 'q' 退出...")
        user_input = input().strip().lower()
        if user_input == 'q':
            print("\n已取消运行剩余示例")
            break


def main():
    """主函数"""
    print_banner()
    print_system_check()

    while True:
        print_menu()
        choice = input("请输入选项 (0-5): ").strip()

        if choice == '0':
            print("\n感谢使用 TinyLMClient 示例！")
            break

        elif choice == '1':
            asyncio.run(run_example("examples.basic_usage", "基础用法示例"))

        elif choice == '2':
            asyncio.run(run_example("examples.advanced_features", "高级特性示例"))

        elif choice == '3':
            # 嵌入示例需要额外检查
            print("\n提示: 嵌入示例需要 nomic-embed-text 模型")
            confirm = input("是否继续? (y/n): ").strip().lower()
            if confirm == 'y':
                asyncio.run(run_example("examples.embeddings_example", "文本嵌入示例"))
            else:
                print("\n已取消")

        elif choice == '4':
            asyncio.run(run_example("examples.real_world_scenarios", "实际应用场景示例"))

        elif choice == '5':
            print("\n注意: 运行所有示例可能需要较长时间")
            confirm = input("是否继续? (y/n): ").strip().lower()
            if confirm == 'y':
                asyncio.run(run_all_examples())
            else:
                print("\n已取消")

        else:
            print(f"\n无效选项: {choice}")
            print("请输入 0-5 之间的数字")

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已中断执行")
        sys.exit(0)
