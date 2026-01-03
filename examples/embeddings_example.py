"""
TinyLMClient 文本嵌入示例

展示如何使用文本嵌入功能进行语义搜索、相似度计算等
前提：已安装 Ollama 并下载支持嵌入的模型（如 nomic-embed-text）
"""

import asyncio
import numpy as np
from tiny_lm_client import (
    TinyLMClient,
    EmbeddingRequest
)


async def basic_embedding_example():
    """基础文本嵌入示例"""
    print("=" * 50)
    print("示例 1: 基础文本嵌入")
    print("=" * 50)

    async with TinyLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    ) as client:

        # 单文本嵌入
        request = EmbeddingRequest(
            model="nomic-embed-text",  # 使用支持嵌入的模型
            input="人工智能是计算机科学的一个分支"
        )

        response = await client.embeddings(request)

        embedding = response.data[0].vector
        print(f"\n嵌入向量维度: {len(embedding)}")
        print(f"前10个维度的值: {embedding[:10]}")
        print(f"\n向量统计:")
        print(f"  - 最小值: {min(embedding):.4f}")
        print(f"  - 最大值: {max(embedding):.4f}")
        print(f"  - 平均值: {np.mean(embedding):.4f}")

        if response.usage:
            print(f"\nToken 使用: {response.usage.total_tokens}")


async def batch_embedding_example():
    """批量文本嵌入示例"""
    print("\n" + "=" * 50)
    print("示例 2: 批量文本嵌入")
    print("=" * 50)

    documents = [
        "机器学习是人工智能的核心技术之一",
        "深度学习使用神经网络进行学习",
        "自然语言处理让计算机理解人类语言",
        "计算机视觉让机器能够识别图像",
        "强化学习通过奖励机制训练智能体"
    ]

    async with TinyLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    ) as client:

        request = EmbeddingRequest(
            model="nomic-embed-text",
            input=documents,
            encoding_format="float"
        )

        response = await client.embeddings(request)

        print(f"\n处理了 {len(response.data)} 个文档的嵌入")
        print(f"每个文档的向量维度: {len(response.data[0].vector)}")

        if response.usage:
            print(f"总 Token 使用: {response.usage.total_tokens}")
            print(f"平均每文档 Token: {response.usage.total_tokens / len(documents):.1f}")


async def similarity_calculation_example():
    """文本相似度计算示例"""
    print("\n" + "=" * 50)
    print("示例 3: 文本相似度计算")
    print("=" * 50)

    # 定义查询和候选文档
    query = "如何训练神经网络？"
    documents = [
        "神经网络需要通过反向传播算法进行训练",
        "今天天气很好，适合去公园散步",
        "深度学习模型通常使用GPU加速训练",
        "机器学习包括监督学习、无监督学习等多种方法"
    ]

    async with TinyLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    ) as client:

        # 计算查询和所有文档的嵌入
        all_texts = [query] + documents
        request = EmbeddingRequest(
            model="nomic-embed-text",
            input=all_texts
        )

        response = await client.embeddings(request)

        # 提取嵌入向量
        query_embedding = np.array(response.data[0].vector)
        doc_embeddings = [np.array(d.vector) for d in response.data[1:]]

        # 计算余弦相似度
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            # 余弦相似度 = (A · B) / (||A|| * ||B||)
            similarity = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            similarities.append((i, similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        print(f"\n查询: {query}\n")
        print("最相似的文档:")
        for rank, (idx, sim) in enumerate(similarities, 1):
            print(f"{rank}. 相似度: {sim:.4f}")
            print(f"   文档: {documents[idx]}\n")


async def semantic_search_example():
    """语义搜索示例"""
    print("\n" + "=" * 50)
    print("示例 4: 语义搜索")
    print("=" * 50)

    # 构建文档库
    knowledge_base = [
        "Python 是一种广泛使用的高级编程语言",
        "JavaScript 主要用于网页开发",
        "Java 是一种面向对象的编程语言",
        "C++ 以其高性能和底层控制而闻名",
        "Go 语言由 Google 开发，适合并发编程",
        "Rust 是一种系统编程语言，注重内存安全",
        "Swift 是 Apple 开发的编程语言",
        "Kotlin 运行在 JVM 上，与 Java 互操作"
    ]

    async with TinyLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    ) as client:

        # 计算所有文档的嵌入（一次性批量处理）
        request = EmbeddingRequest(
            model="nomic-embed-text",
            input=knowledge_base
        )

        response = await client.embeddings(request)
        doc_embeddings = [np.array(d.vector) for d in response.data]

        # 定义搜索函数
        async def search(query_text: str, top_k: int = 3):
            query_req = EmbeddingRequest(
                model="nomic-embed-text",
                input=query_text
            )
            query_resp = await client.embeddings(query_req)
            query_embedding = np.array(query_resp.data[0].vector)

            # 计算所有相似度
            scores = []
            for idx, doc_emb in enumerate(doc_embeddings):
                sim = np.dot(query_embedding, doc_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
                )
                scores.append((idx, sim))

            # 返回前 k 个结果
            scores.sort(key=lambda x: x[1], reverse=True)
            return [(knowledge_base[idx], score) for idx, score in scores[:top_k]]

        # 执行搜索查询
        queries = [
            "适合做网页的语言",
            "高性能的编程语言",
            "内存安全的语言"
        ]

        for query in queries:
            print(f"\n查询: {query}")
            results = await search(query)
            print("匹配结果:")
            for i, (doc, score) in enumerate(results, 1):
                print(f"  {i}. [{score:.4f}] {doc}")


async def text_clustering_example():
    """文本聚类示例"""
    print("\n" + "=" * 50)
    print("示例 5: 文本聚类")
    print("=" * 50)

    # 准备文本
    texts = [
        "机器学习是AI的重要分支",
        "深度学习基于神经网络",
        "自然语言处理理解文本",
        "图像识别属于计算机视觉",
        "数据挖掘发现数据中的模式",
        "强化学习通过奖励学习",
        "算法优化提高程序效率",
        "数据库管理系统存储数据",
        "网络协议管理数据传输"
    ]

    async with TinyLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    ) as client:

        # 获取所有嵌入
        request = EmbeddingRequest(
            model="nomic-embed-text",
            input=texts
        )

        response = await client.embeddings(request)
        embeddings = np.array([d.vector for d in response.data])

        # 简单聚类：计算与质心的距离
        centroids = {
            "AI相关": [0, 1, 2, 3, 4, 5],
            "计算机基础": [6, 7, 8]
        }

        print("\n文本聚类结果:\n")

        for category, indices in centroids.items():
            category_embeddings = embeddings[indices]
            centroid = np.mean(category_embeddings, axis=0)

            print(f"类别: {category}")
            for idx in indices:
                dist = np.linalg.norm(embeddings[idx] - centroid)
                print(f"  - {texts[idx]} (距离质心: {dist:.4f})")
            print()


async def main():
    """运行所有嵌入示例"""
    print("\nTinyLMClient 文本嵌入示例\n")
    print("请确保 Ollama 已运行")
    print("注意: 需要支持嵌入的模型（如 nomic-embed-text）")
    print("可以使用以下命令下载: ollama pull nomic-embed-text\n")

    try:
        await basic_embedding_example()
        await batch_embedding_example()
        await similarity_calculation_example()
        await semantic_search_example()
        await text_clustering_example()

        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        print("=" * 50)

    except Exception as e:
        print(f"\n错误: {e}")
        print("\n如果遇到模型不支持嵌入的错误，请确保:")
        print("1. 已下载支持嵌入的模型: ollama pull nomic-embed-text")
        print("2. 模型名称与代码中一致")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
