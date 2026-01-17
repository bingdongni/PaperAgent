@echo off
REM PaperAgent 快速启动脚本 (Windows版本)

setlocal enabledelayedexpansion

title PaperAgent 快速启动

echo.
echo ================================================
echo        PaperAgent 快速启动脚本
echo ================================================
echo.

REM 检查Docker是否安装
echo [1/5] 检查Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Docker未安装
    echo 请先安装Docker Desktop: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Docker Compose未安装
    pause
    exit /b 1
)

echo [成功] Docker已安装
echo.

REM 检查.env文件
echo [2/5] 检查配置文件...
if not exist .env (
    echo [警告] .env文件不存在，正在创建...
    copy .env.example .env >nul
    echo [成功] .env文件已创建
    echo [提示] 请编辑.env文件配置LLM提供商
) else (
    echo [成功] .env文件已存在
)
echo.

REM 选择LLM提供商
echo [3/5] 选择LLM提供商
echo.
echo 请选择LLM提供商:
echo   1) Ollama (本地免费)
echo   2) OpenAI GPT-4 (需要API密钥)
echo   3) Anthropic Claude (需要API密钥)
echo   4) 跳过 (使用现有配置)
echo.
set /p choice="请输入选项 (1-4): "

if "%choice%"=="1" (
    echo.
    echo [选择] Ollama
    powershell -Command "(Get-Content .env) -replace 'DEFAULT_LLM_PROVIDER=.*', 'DEFAULT_LLM_PROVIDER=ollama' | Set-Content .env"
    echo [成功] 已配置为使用Ollama
    echo [提示] 首次使用请运行: ollama pull llama3
) else if "%choice%"=="2" (
    echo.
    echo [选择] OpenAI
    powershell -Command "(Get-Content .env) -replace 'DEFAULT_LLM_PROVIDER=.*', 'DEFAULT_LLM_PROVIDER=openai' | Set-Content .env"
    set /p api_key="请输入OpenAI API密钥: "
    powershell -Command "(Get-Content .env) -replace 'OPENAI_API_KEY=.*', 'OPENAI_API_KEY=!api_key!' | Set-Content .env"
    echo [成功] 已配置OpenAI
) else if "%choice%"=="3" (
    echo.
    echo [选择] Anthropic Claude
    powershell -Command "(Get-Content .env) -replace 'DEFAULT_LLM_PROVIDER=.*', 'DEFAULT_LLM_PROVIDER=anthropic' | Set-Content .env"
    set /p api_key="请输入Anthropic API密钥: "
    powershell -Command "(Get-Content .env) -replace 'ANTHROPIC_API_KEY=.*', 'ANTHROPIC_API_KEY=!api_key!' | Set-Content .env"
    echo [成功] 已配置Anthropic Claude
) else if "%choice%"=="4" (
    echo.
    echo [信息] 使用现有配置
) else (
    echo.
    echo [错误] 无效选项
    pause
    exit /b 1
)
echo.

REM 停止现有服务
echo [4/5] 停止现有服务...
docker-compose down >nul 2>&1
echo [完成]
echo.

REM 启动服务
echo [5/5] 启动PaperAgent服务...
echo.
docker-compose up -d

if %errorlevel% neq 0 (
    echo.
    echo [错误] 服务启动失败
    pause
    exit /b 1
)

echo.
echo [成功] 服务启动成功！
echo.

REM 等待服务就绪
echo [等待] 服务正在启动，请稍候...
timeout /t 10 /nobreak >nul

echo.
echo ================================================
echo           服务已成功启动！
echo ================================================
echo.
echo 访问地址:
echo.
echo   Web界面: http://localhost:8501
echo   API文档: http://localhost:8000/docs
echo   健康检查: http://localhost:8000/health
echo.
echo 常用命令:
echo.
echo   查看日志:  docker-compose logs -f
echo   停止服务:  docker-compose stop
echo   重启服务:  docker-compose restart
echo   完全停止:  docker-compose down
echo.
echo 提示:
echo   - 首次使用建议先查看 README.md
echo   - 配置文件位于 .env
echo   - 数据保存在 data\ 目录
echo.
echo ================================================
echo.

REM 询问是否打开浏览器
set /p open_browser="是否现在打开Web界面? (y/n): "

if /i "%open_browser%"=="y" (
    start http://localhost:8501
)

echo.
echo 开始您的学术研究之旅！
echo.
pause
