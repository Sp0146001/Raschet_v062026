@echo off
setlocal

REM Сборка одного EXE с иконкой и версией.
REM Перед запуском: pip install -r requirements.txt

python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --windowed ^
  --onefile ^
  --name raschet-v16062026 ^
  --icon assets\app_icon.ico ^
  --version-file version_info.txt ^
  --hidden-import pyqtgraph.exporters ^
  --collect-all PySide6 ^
  --collect-all pyqtgraph ^
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
