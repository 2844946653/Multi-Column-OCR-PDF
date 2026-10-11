@echo off
rem 战锤 PDF 提取工具 - 双击启动（无控制台窗口）
cd /d "%~dp0"
where pythonw >nul 2>nul
if %errorlevel%==0 (
  start "" pythonw "src\app.py"
) else (
  start "" python "src\app.py"
)
