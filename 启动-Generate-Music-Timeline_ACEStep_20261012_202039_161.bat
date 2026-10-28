@echo off
chcp 65001 >nul
title Generate-Music-Timeline 一键启动 (WebUI + YuE2)
setlocal

set PROJECT=F:\Generate-Music-Timeline
set WEBUI_PY=%PROJECT%\venv_timeline\Scripts\python.exe

echo [1/2] 检查 YuE2 服务 (127.0.0.1:8189) ...
powershell -NoProfile -Command "try { $null = Invoke-RestMethod -Uri 'http://127.0.0.1:8189/' -TimeoutSec 5; exit 0 } catch { exit 1 }"
if %errorlevel% neq 0 (
    echo YuE2 服务未运行, 请先启动 YuE2 一键包服务(如 deps\YuE2-T8-Local-v1.6.0-Windows-NVIDIA-20260919\start_webui.bat)后再启动本脚本.
)

echo [2/2] 启动 Generate-Music-Timeline WebUI (端口 7860) ...
cd /d %PROJECT%
start "Generate-Music-Timeline-WebUI" "%WEBUI_PY%" "%PROJECT%\webui.py"

echo 启动完成: WebUI http://127.0.0.1:7860 ; YuE2 http://127.0.0.1:8189
endlocal
