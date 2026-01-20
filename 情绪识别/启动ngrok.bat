@echo off
chcp 65001 >nul
title 情绪识别Web应用 - ngrok版
color 0A

echo ========================================
echo   情绪识别Web应用 - 一键启动（ngrok版）
echo ========================================
echo.
echo 此脚本将：
echo 1. 启动Flask服务器
echo 2. 启动ngrok隧道
echo.
echo 注意：需要两个命令行窗口
echo.
echo ========================================
echo.

REM 检查ngrok是否已配置
ngrok config check >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] 检测到ngrok未配置，正在自动配置...
    call 配置ngrok.bat
    echo.
)

echo [步骤1] 正在启动Flask服务器...
echo.
start "Flask服务器" cmd /k "python app.py"

echo [步骤2] 等待服务器启动（5秒）...
timeout /t 5 /nobreak >nul

echo [步骤3] 正在启动ngrok隧道...
echo.
echo ========================================
echo   重要提示
echo ========================================
echo ngrok启动后会显示一个HTTPS地址，例如：
echo   https://abc123.ngrok-free.app
echo.
echo 请在手机浏览器中访问该HTTPS地址！
echo ========================================
echo.
echo 按任意键启动ngrok...
pause >nul

start "ngrok隧道" cmd /k "ngrok http 8080"

echo.
echo ========================================
echo   启动完成！
echo ========================================
echo.
echo 已打开两个窗口：
echo 1. Flask服务器窗口 - 显示服务器运行状态
echo 2. ngrok窗口 - 显示HTTPS访问地址
echo.
echo 请在ngrok窗口中找到HTTPS地址，然后在手机浏览器访问
echo.
echo 按任意键退出此窗口（服务器和ngrok会继续运行）
pause >nul









