# .env 配置详细指南

本指南将帮助您正确配置PaperAgent的环境变量。

## 📋 快速配置步骤

### 1. 复制配置文件

```bash
cp .env.example .env
```

### 2. 根据您的使用场景选择配置方案

---

## 🎯 配置方案选择

### 方案A: 本地免费使用（推荐新手）

**特点**: 完全免费，无需API密钥，使用本地LLM

**配置步骤**:

1. **安装Ollama** (如果还没安装)
   ```bash
   # Windows: 下载安装
   # https://ollama.ai/download

   # Linux/Mac
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **拉取模型**
   ```bash
   # 推荐: Llama 3 (7B)
   ollama pull llama3

   # 或者使用中文优化模型
   ollama pull qwen2
   ```

3. **编辑.env文件**
   ```bash
   # LLM配置
   DEFAULT_LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3

   # 可以不填写以下API密钥
   OPENAI_API_KEY=
   ANTHROPIC_API_KEY=

   # 数据库 (使用Docker)
   DATABASE_URL=postgresql://paperagent:paperagent_password@postgres:5432/paperagent
   REDIS_URL=redis://redis:6379/0
   ```

4. **启动服务**
   ```bash
   docker-compose up -d
   ```

✅ **完成！现在可以免费使用PaperAgent了！**

---

### 方案B: 使用OpenAI GPT-4（推荐效果最好）

**特点**: 效果最好，需要API密钥，按使用付费

**配置步骤**:

1. **获取OpenAI API密钥**
   - 访问: https://platform.openai.com/api-keys
   - 创建新的API密钥
   - 复制密钥（以 `sk-` 开头）

2. **编辑.env文件**
   ```bash
   # LLM配置
   DEFAULT_LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   OPENAI_MODEL=gpt-4o

   # 如果使用GPT-4 Turbo
   # OPENAI_MODEL=gpt-4-turbo-preview

   # 如果使用GPT-3.5 (更便宜)
   # OPENAI_MODEL=gpt-3.5-turbo

   # 数据库配置
   DATABASE_URL=postgresql://paperagent:paperagent_password@postgres:5432/paperagent
   REDIS_URL=redis://redis:6379/0
   ```

3. **启动服务**
   ```bash
   docker-compose up -d
   ```

💡 **价格参考** (2024年1月):
- GPT-4: $0.03/1K tokens (输入), $0.06/1K tokens (输出)
- GPT-3.5 Turbo: $0.0005/1K tokens (输入), $0.0015/1K tokens (输出)

---

### 方案C: 使用Anthropic Claude（推荐平衡）

**特点**: 效果优秀，速度快，上下文长，价格适中

**配置步骤**:

1. **获取Anthropic API密钥**
   - 访问: https://console.anthropic.com/
   - 创建API密钥
   - 复制密钥

2. **编辑.env文件**
   ```bash
   # LLM配置
   DEFAULT_LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

   # 其他Claude模型选项
   # ANTHROPIC_MODEL=claude-3-opus-20240229  # 最强但最贵
   # ANTHROPIC_MODEL=claude-3-haiku-20240307 # 最快最便宜

   # 数据库配置
   DATABASE_URL=postgresql://paperagent:paperagent_password@postgres:5432/paperagent
   REDIS_URL=redis://redis:6379/0
   ```

3. **启动服务**
   ```bash
   docker-compose up -d
   ```

💡 **Claude优势**:
- 200K tokens上下文窗口
- 优秀的代码生成能力
- 快速响应
- 价格合理

---

## 📝 完整配置说明

### 数据库配置

```bash
# 使用Docker (推荐)
DATABASE_URL=postgresql://paperagent:paperagent_password@postgres:5432/paperagent

# 使用本地PostgreSQL
# DATABASE_URL=postgresql://username:password@localhost:5432/paperagent

# 使用SQLite (仅开发测试)
# DATABASE_URL=sqlite:///./paperagent.db
```

### Redis配置

```bash
# 使用Docker
REDIS_URL=redis://redis:6379/0

# 使用本地Redis
# REDIS_URL=redis://localhost:6379/0

# 带密码的Redis
# REDIS_URL=redis://:password@redis:6379/0
```

### LLM详细配置

```bash
# OpenAI配置
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o                    # 推荐
# OPENAI_MODEL=gpt-4-turbo-preview    # 更快
# OPENAI_MODEL=gpt-3.5-turbo          # 更便宜

# Anthropic配置
ANTHROPIC_API_KEY=sk-ant-your-api-key
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022  # 推荐
# ANTHROPIC_MODEL=claude-3-opus-20240229    # 最强
# ANTHROPIC_MODEL=claude-3-haiku-20240307   # 最快

# Ollama配置
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3                    # 推荐
# OLLAMA_MODEL=qwen2                   # 中文优化
# OLLAMA_MODEL=mistral                 # 小而强
# OLLAMA_MODEL=codellama               # 代码优化

# 选择默认提供商
DEFAULT_LLM_PROVIDER=ollama            # ollama, openai, 或 anthropic
```

### 文献检索配置

```bash
# arXiv配置
ARXIV_MAX_RESULTS=50                   # 每次搜索最多返回论文数

# Google Scholar (可选)
USE_PROXY=false                        # 是否使用代理
PROXY_URL=                             # 代理URL (如需要)
# PROXY_URL=http://proxy.example.com:8080
```

### 应用设置

```bash
# 基本信息
APP_NAME=PaperAgent
APP_VERSION=1.0.0
DEBUG=true                             # 生产环境设为false
LOG_LEVEL=INFO                         # DEBUG, INFO, WARNING, ERROR

# 文件存储路径
DATA_DIR=./data
PAPERS_DIR=./data/papers
EXPERIMENTS_DIR=./data/experiments
LITERATURE_DIR=./data/literature
OUTPUTS_DIR=./data/outputs
```

### 安全配置

```bash
# 密钥 (生产环境必须修改！)
SECRET_KEY=your-secret-key-change-this-in-production

# 生成随机密钥:
# python -c "import secrets; print(secrets.token_urlsafe(32))"

# 允许的主机
ALLOWED_HOSTS=localhost,127.0.0.1
# 生产环境添加实际域名:
# ALLOWED_HOSTS=localhost,127.0.0.1,yourdomain.com
```

### 性能配置

```bash
# 任务队列
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# 速率限制
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60               # 每分钟请求限制

# 并发配置
MAX_CONCURRENT_TASKS=5                 # 最大并发任务数

# 会话配置
SESSION_TIMEOUT=3600                   # 会话超时 (秒)
```

### LLM参数调优

```bash
# 生成参数
MAX_TOKENS=4096                        # 最大生成token数
TEMPERATURE=0.7                        # 温度 (0-1, 越高越随机)
TOP_P=0.9                             # 核采样参数

# 调优建议:
# - 创意写作: TEMPERATURE=0.8-0.9
# - 代码生成: TEMPERATURE=0.2-0.4
# - 数据分析: TEMPERATURE=0.3-0.5
# - 学术写作: TEMPERATURE=0.6-0.7 (推荐)
```

---

## 🔧 特殊场景配置

### 1. 离线使用（无互联网）

```bash
# 使用本地Ollama
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# 禁用在线文献搜索
USE_PROXY=false

# 使用本地数据库
DATABASE_URL=sqlite:///./paperagent.db
```

### 2. 多用户生产环境

```bash
# 使用强密钥
SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# 配置数据库连接池
DATABASE_URL=postgresql://user:pass@db:5432/paperagent?pool_size=20

# 增加并发
MAX_CONCURRENT_TASKS=20

# 启用速率限制
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=100

# 生产模式
DEBUG=false
LOG_LEVEL=WARNING
```

### 3. 开发测试环境

```bash
# 使用本地数据库
DATABASE_URL=sqlite:///./test.db
REDIS_URL=redis://localhost:6379/0

# 调试模式
DEBUG=true
LOG_LEVEL=DEBUG

# 使用便宜的模型
DEFAULT_LLM_PROVIDER=ollama
# 或
# DEFAULT_LLM_PROVIDER=openai
# OPENAI_MODEL=gpt-3.5-turbo
```

### 4. 中国大陆用户配置

```bash
# 使用国内可访问的服务

# 选项1: 使用Ollama (本地)
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2                     # 通义千问

# 选项2: 使用API代理
OPENAI_API_KEY=your-key
# 配置代理服务器
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890

# 文献搜索使用代理
USE_PROXY=true
PROXY_URL=http://127.0.0.1:7890
```

---

## ✅ 配置验证

创建一个测试脚本验证配置：

```python
# test_config.py
import os
from dotenv import load_dotenv

load_dotenv()

print("🔍 验证配置...")

# 检查必需配置
required = ['DATABASE_URL', 'REDIS_URL', 'DEFAULT_LLM_PROVIDER']
for key in required:
    value = os.getenv(key)
    status = "✅" if value else "❌"
    print(f"{status} {key}: {value if value else 'NOT SET'}")

# 检查LLM配置
provider = os.getenv('DEFAULT_LLM_PROVIDER')
print(f"\n🤖 LLM提供商: {provider}")

if provider == 'openai':
    key = os.getenv('OPENAI_API_KEY')
    print(f"{'✅' if key and key != 'your_openai_api_key_here' else '❌'} OpenAI API Key")

elif provider == 'anthropic':
    key = os.getenv('ANTHROPIC_API_KEY')
    print(f"{'✅' if key and key != 'your_anthropic_api_key_here' else '❌'} Anthropic API Key")

elif provider == 'ollama':
    url = os.getenv('OLLAMA_BASE_URL')
    print(f"✅ Ollama URL: {url}")
    print("💡 请确保Ollama正在运行: ollama serve")

print("\n✅ 配置验证完成！")
```

运行验证：
```bash
python test_config.py
```

---

## 🚀 启动命令

### 使用Docker（推荐）

```bash
# 1. 复制配置
cp .env.example .env

# 2. 编辑配置
nano .env

# 3. 启动所有服务
docker-compose up -d

# 4. 查看日志
docker-compose logs -f

# 5. 访问应用
# Web界面: http://localhost:8501
# API文档: http://localhost:8000/docs
```

### 本地开发

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置.env
cp .env.example .env
nano .env

# 3. 启动数据库和Redis (如果使用Docker)
docker-compose up -d postgres redis

# 4. 运行数据库迁移
alembic upgrade head

# 5. 启动API服务
uvicorn paperagent.api.main:app --reload --host 0.0.0.0 --port 8000

# 6. 启动Web界面 (新终端)
streamlit run paperagent/web/app.py
```

---

## 💡 常见问题

### Q1: 忘记修改默认密钥？
```bash
# 生成新密钥
python -c "import secrets; print(secrets.token_urlsafe(32))"
# 复制到.env的SECRET_KEY
```

### Q2: Ollama连接失败？
```bash
# 检查Ollama是否运行
ollama list

# 启动Ollama
ollama serve

# 拉取模型
ollama pull llama3
```

### Q3: 数据库连接失败？
```bash
# 检查Docker容器
docker-compose ps

# 重启数据库
docker-compose restart postgres

# 查看数据库日志
docker-compose logs postgres
```

### Q4: API密钥无效？
- 检查密钥是否正确复制（无空格）
- 确认API账户有余额
- 检查密钥权限

---

## 📞 需要帮助？

如果配置遇到问题：

1. 检查日志: `docker-compose logs`
2. 运行配置验证: `python test_config.py`
3. 查看文档: `README.md`
4. 提交Issue: GitHub Issues

---

**配置完成后，即可开始使用PaperAgent！** 🎉📚✨
