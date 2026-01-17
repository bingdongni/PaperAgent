#!/bin/bash
# PaperAgent 快速启动脚本

set -e

echo "🚀 PaperAgent 快速启动脚本"
echo "================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker未安装${NC}"
        echo "请先安装Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose未安装${NC}"
        echo "请先安装Docker Compose"
        exit 1
    fi

    echo -e "${GREEN}✅ Docker已安装${NC}"
}

# 检查.env文件
check_env() {
    if [ ! -f .env ]; then
        echo -e "${YELLOW}⚠️  .env文件不存在，正在创建...${NC}"
        cp .env.example .env
        echo -e "${GREEN}✅ .env文件已创建${NC}"
        echo -e "${YELLOW}💡 请编辑.env文件配置LLM提供商${NC}"
    else
        echo -e "${GREEN}✅ .env文件已存在${NC}"
    fi
}

# 选择LLM提供商
select_llm() {
    echo ""
    echo "请选择LLM提供商:"
    echo "1) Ollama (本地免费)"
    echo "2) OpenAI GPT-4 (需要API密钥)"
    echo "3) Anthropic Claude (需要API密钥)"
    echo "4) 跳过（使用现有配置）"
    echo ""
    read -p "请输入选项 (1-4): " choice

    case $choice in
        1)
            echo "选择了Ollama"
            sed -i 's/DEFAULT_LLM_PROVIDER=.*/DEFAULT_LLM_PROVIDER=ollama/' .env
            echo -e "${GREEN}✅ 已配置为使用Ollama${NC}"
            echo -e "${YELLOW}💡 首次使用请运行: ollama pull llama3${NC}"
            ;;
        2)
            echo "选择了OpenAI"
            sed -i 's/DEFAULT_LLM_PROVIDER=.*/DEFAULT_LLM_PROVIDER=openai/' .env
            read -p "请输入OpenAI API密钥: " api_key
            sed -i "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$api_key/" .env
            echo -e "${GREEN}✅ 已配置OpenAI${NC}"
            ;;
        3)
            echo "选择了Anthropic Claude"
            sed -i 's/DEFAULT_LLM_PROVIDER=.*/DEFAULT_LLM_PROVIDER=anthropic/' .env
            read -p "请输入Anthropic API密钥: " api_key
            sed -i "s/ANTHROPIC_API_KEY=.*/ANTHROPIC_API_KEY=$api_key/" .env
            echo -e "${GREEN}✅ 已配置Anthropic Claude${NC}"
            ;;
        4)
            echo -e "${BLUE}使用现有配置${NC}"
            ;;
        *)
            echo -e "${RED}无效选项${NC}"
            exit 1
            ;;
    esac
}

# 启动服务
start_services() {
    echo ""
    echo -e "${BLUE}🚀 正在启动服务...${NC}"
    echo ""

    # 停止现有容器
    docker-compose down 2>/dev/null || true

    # 启动所有服务
    docker-compose up -d

    echo ""
    echo -e "${GREEN}✅ 服务启动成功！${NC}"
    echo ""
    echo "📊 服务状态:"
    docker-compose ps
}

# 等待服务就绪
wait_for_services() {
    echo ""
    echo -e "${BLUE}⏳ 等待服务就绪...${NC}"

    # 等待PostgreSQL
    echo -n "等待数据库启动"
    for i in {1..30}; do
        if docker-compose exec -T postgres pg_isready -U paperagent &>/dev/null; then
            echo -e " ${GREEN}✅${NC}"
            break
        fi
        echo -n "."
        sleep 1
    done

    # 等待API服务
    echo -n "等待API服务启动"
    for i in {1..30}; do
        if curl -s http://localhost:8000/health &>/dev/null; then
            echo -e " ${GREEN}✅${NC}"
            break
        fi
        echo -n "."
        sleep 1
    done

    # 等待Web服务
    echo -n "等待Web界面启动"
    for i in {1..30}; do
        if curl -s http://localhost:8501 &>/dev/null; then
            echo -e " ${GREEN}✅${NC}"
            break
        fi
        echo -n "."
        sleep 1
    done
}

# 显示访问信息
show_info() {
    echo ""
    echo "╔════════════════════════════════════════════════╗"
    echo "║                                                ║"
    echo "║        🎉 PaperAgent 已成功启动！              ║"
    echo "║                                                ║"
    echo "╚════════════════════════════════════════════════╝"
    echo ""
    echo "📍 访问地址:"
    echo ""
    echo -e "  ${GREEN}🌐 Web界面:${NC} http://localhost:8501"
    echo -e "  ${GREEN}📚 API文档:${NC} http://localhost:8000/docs"
    echo -e "  ${GREEN}🔧 API健康:${NC} http://localhost:8000/health"
    echo ""
    echo "🛠️  常用命令:"
    echo ""
    echo "  查看日志:     docker-compose logs -f"
    echo "  停止服务:     docker-compose stop"
    echo "  重启服务:     docker-compose restart"
    echo "  完全停止:     docker-compose down"
    echo ""
    echo "💡 提示:"
    echo "  - 首次使用建议先查看 README.md"
    echo "  - 配置文件位于 .env"
    echo "  - 数据保存在 data/ 目录"
    echo ""
}

# 主函数
main() {
    check_docker
    check_env
    select_llm
    start_services
    wait_for_services
    show_info

    # 询问是否打开浏览器
    echo ""
    read -p "是否现在打开Web界面? (y/n): " open_browser

    if [ "$open_browser" = "y" ] || [ "$open_browser" = "Y" ]; then
        if command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:8501
        elif command -v open &> /dev/null; then
            open http://localhost:8501
        else
            echo "请手动打开浏览器访问: http://localhost:8501"
        fi
    fi

    echo ""
    echo -e "${GREEN}🎓 开始您的学术研究之旅！${NC}"
    echo ""
}

# 运行主函数
main
