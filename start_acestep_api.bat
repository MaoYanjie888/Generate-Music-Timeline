@echo off
chcp 65001 >nul
REM ============================================================
REM  ACE-Step 1.5 独立 REST API Server 启动脚本
REM  用途：为 WebUI 第 ⑦ 个 Tab「ACE-Step 音色跟随」提供后端服务
REM  环境隔离：使用 ACE-Step 自己的 .venv，不污染 venv_timeline
REM  数据只在本机回环 127.0.0.1:8001，不出本机
REM ============================================================

set ACE_ROOT=F:\ACE-Step-1.5
set PY=%ACE_ROOT%\.venv\Scripts\python.exe

if not exist "%PY%" (
    echo [ERROR] 找不到 ACE-Step 的 Python: %PY%
    pause
    exit /b 1
)

REM 本机走代理时放行回环地址，避免 Gradio/HTTP 自检被代理拦截
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost
REM cover 任务自动跳过 LM（文档说明），不初始化 LLM 以节省显存
set ACESTEP_INIT_LLM=false
set ACESTEP_NO_INIT=true
set ACESTEP_DOWNLOAD_SOURCE=modelscope

echo 正在启动 ACE-Step API Server ^(http://127.0.0.1:8001^) ...
cd /d "%ACE_ROOT%"
"%PY%" -m acestep.api_server --host 127.0.0.1 --port 8001

pause
