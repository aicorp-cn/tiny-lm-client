from setuptools import setup, find_packages

setup(
    name="tiny-lm-client",
    version="1.0.0",
    description="轻量级OpenAI兼容大模型客户端",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="AI-Corp",
    author_email="our-aicorp@hotmail.com",
    url="https://github.com/aicorp-cn/tiny-lm-client",
    project_urls={
        "Bug Tracker": "https://github.com/aicorp-cn/tiny-lm-client/issues",
        "Source Code": "https://github.com/aicorp-cn/tiny-lm-client",
    },
    packages=find_packages(exclude=["test_env", "prompts"]),
    install_requires=[
        "httpx>=0.24.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    keywords=["openai", "llm", "client", "ai", "nlp"],
)