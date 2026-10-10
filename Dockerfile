# SemiForge 运行时镜像 (作者: 晨星)
FROM python:3.13-slim

WORKDIR /app

# 安装运行时依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝源码
COPY . .

# 运行端到端基准 (落盘 benchmark.json)
CMD ["python", "cli.py", "demo"]
