# PaperAgent 项目结构说明

## 📁 项目文件结构

```
PaperAgent/
│
├── 📄 README.md                    # 完整的项目文档（英文）
├── 📄 LICENSE                      # MIT开源协议
├── 📄 .gitignore                   # Git忽略文件
├── 📄 .env.example                 # 环境变量模板
├── 📄 requirements.txt             # Python依赖
├── 📄 Dockerfile                   # Docker镜像配置
├── 📄 docker-compose.yml           # Docker编排配置
├── 📄 setup.py                     # 快速安装脚本
├── 📄 run.bat                      # Windows启动脚本
├── 📄 run.sh                       # Linux/Mac启动脚本
│
├── 📂 paperagent/                  # 核心代码目录
│   ├── __init__.py
│   │
│   ├── 📂 agents/                  # 智能体模块
│   │   ├── __init__.py
│   │   ├── base_agent.py          # 基础智能体类
│   │   ├── boss_agent.py          # 中枢调度智能体
│   │   ├── literature_agent.py    # 文献调研智能体
│   │   ├── experiment_agent.py    # 实验设计智能体
│   │   └── writing_agent.py       # 论文写作智能体
│   │
│   ├── 📂 core/                    # 核心功能模块
│   │   ├── __init__.py
│   │   ├── config.py              # 配置管理
│   │   ├── llm_manager.py         # LLM管理器
│   │   └── prompts.py             # 提示词模板
│   │
│   ├── 📂 database/                # 数据库模块
│   │   ├── __init__.py
│   │   ├── models.py              # 数据模型定义
│   │   └── database.py            # 数据库连接管理
│   │
│   ├── 📂 tools/                   # 工具集
│   │   ├── __init__.py
│   │   ├── literature_collector.py    # 文献采集
│   │   ├── latex_processor.py         # LaTeX处理
│   │   └── document_processor.py      # 文档处理
│   │
│   ├── 📂 api/                     # FastAPI后端
│   │   ├── __init__.py
│   │   ├── main.py                # API主入口
│   │   ├── schemas.py             # Pydantic模型
│   │   └── 📂 routers/            # API路由
│   │       ├── projects.py
│   │       ├── literature.py
│   │       ├── experiments.py
│   │       ├── papers.py
│   │       └── tasks.py
│   │
│   └── 📂 web/                     # Streamlit前端
│       ├── __init__.py
│       └── app.py                 # Web界面
│
├── 📂 data/                        # 数据存储目录
│   ├── papers/                    # 论文文件
│   ├── experiments/               # 实验数据
│   ├── literature/                # 文献PDF
│   └── outputs/                   # 输出文件
│
├── 📂 examples/                    # 示例代码
│   ├── __init__.py
│   └── usage_examples.py          # 使用示例
│
└── 📂 docs/                        # 文档目录
    └── (可添加更多文档)
```

## 🎯 核心模块说明

### 1. 智能体模块 (agents/)
包含所有AI智能体的实现：

- **BossAgent**: 中枢调度器，负责任务分解、进度监控、质量控制
- **LiteratureAgent**: 文献调研，包括搜索、分析、聚类、缺口识别
- **ExperimentAgent**: 实验设计与数据分析
- **WritingAgent**: 学术论文写作与润色

### 2. 核心模块 (core/)
基础功能组件：

- **config.py**: 全局配置管理，支持环境变量
- **llm_manager.py**: 统一的LLM接口，支持OpenAI、Anthropic、Ollama
- **prompts.py**: 基于KtR框架设计的提示词模板

### 3. 数据库模块 (database/)
持久化存储：

- **models.py**: SQLAlchemy数据模型（项目、任务、文献、论文等）
- **database.py**: 数据库连接和会话管理

### 4. 工具模块 (tools/)
专用工具：

- **literature_collector.py**: arXiv和Google Scholar文献采集
- **latex_processor.py**: LaTeX文档生成和处理
- **document_processor.py**: PDF/Word文档处理

### 5. API模块 (api/)
RESTful API服务：

- **main.py**: FastAPI应用主入口
- **schemas.py**: 请求/响应数据模型
- **routers/**: 按功能划分的API路由

### 6. Web模块 (web/)
用户界面：

- **app.py**: Streamlit Web应用

## 🚀 快速开始

### 方式1：使用Docker（推荐）
```bash
# 复制环境配置
cp .env.example .env

# 启动所有服务
docker-compose up -d

# 访问应用
# Web UI: http://localhost:8501
# API: http://localhost:8000/docs
```

### 方式2：本地安装

**Windows:**
```bash
run.bat
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

### 方式3：手动安装
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 初始化数据库
python setup.py

# 启动API服务
uvicorn paperagent.api.main:app --host 0.0.0.0 --port 8000

# 启动Web界面（另一个终端）
streamlit run paperagent/web/app.py
```

## 📝 配置说明

编辑 `.env` 文件配置以下选项：

```bash
# LLM提供商选择
DEFAULT_LLM_PROVIDER=ollama  # openai, anthropic, ollama

# 本地LLM (Ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# OpenAI (可选)
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o

# Anthropic Claude (可选)
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# 数据库
DATABASE_URL=postgresql://paperagent:password@localhost:5432/paperagent
# 或使用SQLite: sqlite:///./paperagent.db

# 文献采集
USE_PROXY=false
ARXIV_MAX_RESULTS=50
```

## 🔧 开发指南

### 添加新智能体
1. 在 `paperagent/agents/` 创建新文件
2. 继承 `BaseAgent` 类
3. 实现 `execute()` 方法
4. 在 `__init__.py` 导出

### 添加新API端点
1. 在 `paperagent/api/routers/` 创建路由文件
2. 定义API端点
3. 在 `main.py` 注册路由

### 添加新工具
1. 在 `paperagent/tools/` 创建工具文件
2. 实现工具类和方法
3. 在 `__init__.py` 导出

## 📚 使用示例

查看 `examples/usage_examples.py` 了解详细使用方法：

```python
from paperagent.agents import BossAgent

boss = BossAgent()

# 创建项目
project = boss.execute({
    'action': 'create_project',
    'name': 'My Research Project',
    'research_field': 'Computer Science',
    'keywords': ['AI', 'Machine Learning']
})

# 执行完整工作流
result = boss.execute({
    'action': 'execute_workflow',
    'project_id': project['project_id']
})
```

## 🐛 故障排除

### 数据库连接失败
- 确保PostgreSQL正在运行
- 或使用SQLite：`DATABASE_URL=sqlite:///./paperagent.db`

### LLM调用失败
- 检查API密钥配置
- 或使用本地Ollama：安装并运行 `ollama pull llama3`

### 端口占用
- 修改 `.env` 中的端口配置
- 或停止占用端口的其他服务

## 📖 更多资源

- 完整文档：查看 `README.md`
- API文档：http://localhost:8000/docs
- 问题反馈：https://github.com/yourusername/paperagent/issues

## 📄 许可证

MIT License - 详见 `LICENSE` 文件

---

**祝你科研顺利！🎓📚**
