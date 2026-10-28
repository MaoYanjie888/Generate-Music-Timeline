@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "ROOT=%~dp0"

echo ============================================================
echo   Generate-Music-Timeline 环境修复脚本
echo   作用: 将两个虚拟环境绑定到项目自带的 runtime_py310
echo   适用: 移动文件夹 / 换电脑后第一次运行
echo ============================================================
echo.

rem --- 检查 runtime_py310 ---
if not exist "%ROOT%runtime_py310\python.exe" (
    echo [!] 未找到 runtime_py310\python.exe，请确认项目文件完整。
    pause
    exit /b 1
)

rem --- 重写 venv_timeline\pyvenv.cfg ---
echo [1/2] 修复 venv_timeline ...
>  "%ROOT%venv_timeline\pyvenv.cfg" echo home = %ROOT%runtime_py310
>> "%ROOT%venv_timeline\pyvenv.cfg" echo include-system-site-packages = false
>> "%ROOT%venv_timeline\pyvenv.cfg" echo version = 3.10.16

rem --- 重写 SoulX venv\pyvenv.cfg ---
echo [2/2] 修复 SoulX venv ...
>  "%ROOT%deps\SoulX-Singer\venv\pyvenv.cfg" echo home = %ROOT%runtime_py310
>> "%ROOT%deps\SoulX-Singer\venv\pyvenv.cfg" echo include-system-site-packages = false
>> "%ROOT%deps\SoulX-Singer\venv\pyvenv.cfg" echo version = 3.10.16

echo.
echo [OK] 环境绑定完成。可直接双击「启动工具.bat」运行项目。
echo.
pause
