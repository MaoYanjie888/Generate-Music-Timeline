@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo    Generate-Music-Timeline 整合启动器
echo    Tab1 唱歌创作 / Tab2 歌词时间戳
echo ============================================================
echo.

rem --- 自动定位 YuE2 一键包目录(目录名随版本可能变化, 不写死) ---
set "YUE2_KIT="
for /d %%D in ("%~dp0deps\YuE2-T8-Local-*") do set "YUE2_KIT=%%D"
if not defined YUE2_KIT (
    echo [!] 未找到 YuE2 一键包目录 deps\YuE2-T8-Local-*，请检查依赖完整性
    pause
    exit /b 1
)
echo [OK] 定位到 YuE2 一键包: %YUE2_KIT%
echo.

rem --- 检查 YuE2 服务(唱歌创作需要) ---
curl -s -m 3 http://127.0.0.1:8189/api/health >nul 2>&1
if "!errorlevel!"=="0" (
    echo [OK] YuE2 服务已运行 (127.0.0.1:8189)
    goto START_WEBUI
)

echo [!] YuE2 服务未运行，正在自动启动...
echo     首次加载模型约需 1~3 分钟，请耐心等待，就绪后自动打开界面。
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%YUE2_KIT%\启动本地整合包.ps1" -NoBrowser

curl -s -m 3 http://127.0.0.1:8189/api/health >nul 2>&1
if "!errorlevel!"=="0" (
    echo [OK] YuE2 服务已经就绪
) else (
    echo [!] 未能确认 YuE2 就绪，请查看 %YUE2_KIT%\logs\server.stderr.log；界面仍将启动。
)

:START_WEBUI
echo 正在启动整合界面(浏览器将自动打开)...
"%~dp0venv_timeline\Scripts\python.exe" webui.py
echo.
echo 界面已退出
pause
