@echo off
setlocal

REM Облегчённая сборка EXE без лишних collect-all.
REM Лучше запускать в отдельном чистом venv, где установлен PySide6-Essentials.

python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --windowed ^
  --onefile ^
  --name raschet-v17062026 ^
  --icon assets\app_icon.ico ^
  --version-file version_info.txt ^
  --hidden-import pyqtgraph.exporters ^
  --exclude-module matplotlib ^
  --exclude-module PyQt5 ^
  --exclude-module PyQt6 ^
  --exclude-module PySide2 ^
  --exclude-module tkinter ^
  --add-data "assets;assets" ^
  main.py

if errorlevel 1 (
  echo.
  echo Сборка завершилась с ошибкой.
  pause
  exit /b 1
)

echo.
echo Готово. Файл лежит в dist\raschet-v16062026.exe
pause
