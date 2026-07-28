# raschet-qt v2

Новая модульная версия приложения на **PySide6 + pyqtgraph**.

## Что изменено

### Архитектура
Проект разбит на модули:

```text
main.py
raschet_qt.py                # совместимый launcher
raschet_app/
  constants.py
  config.py
  models.py
  services/
    parser.py
    calculator.py
    exporters.py
  ui/
    main_window.py
    plot_widget.py
    channel_legend.py
    table_utils.py
assets/
  app_icon.png
  app_icon.ico
version_info.txt
build_exe.bat
requirements.txt
```

### Новый функционал

- переход с **matplotlib** на **pyqtgraph** для более быстрого интерактива;
- отдельная **легенда каналов** вне графика;
- **draggable bar-line** (`InfiniteLine`) с live-обновлением таблицы `Rᵢ`;
- **выбор интервала мышью** прямо на графике:
  - включите чекбокс `Режим выбора участка мышью`,
  - тяните ЛКМ по графику,
  - интервал сразу появится в полях и в зелёной области;
- **умный парсер TXT**:
  - ищет `Channels / Channals`,
  - понимает табы, пробелы, `;`, `,`,
  - умеет вытаскивать числовые строки через fallback-разбор,
  - подстраивает число столбцов по наиболее типичному формату;
- упаковка в `.exe` с **иконкой** и **информацией о версии**.

---

## Основные возможности

- загрузка TXT-файла;
- отображение всех видимых каналов;
- скрытие/показ каналов галочками;
- переименование каналов;
- логарифмическая/линейная шкала Y;
- автомасштаб графика;
- построение выделенного участка;
- сохранение участка в `.txt / .csv / .xlsx`;
- расчёт 4 режимов;
- поиск `t(0.9)` / `t(0.1)` по **первому пересечению** с **линейной интерполяцией**;
- усреднение `R0` и `R(g)` по `N` точкам;
- копирование таблиц в буфер для Excel;
- статус-бар с координатами курсора.

---

## Установка зависимостей

### CMD

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Запуск

```bat
python main.py
```

или

```bat
python raschet_qt.py
```

---

## Сборка в EXE

### Быстрый способ

```bat
build_exe.bat
```

### Вручную

```bat
python -m PyInstaller --noconfirm --clean --windowed --onefile --name raschet-qt --icon assets\app_icon.ico --version-file version_info.txt --hidden-import pyqtgraph.exporters --collect-all PySide6 --collect-all pyqtgraph --add-data "assets;assets" main.py
```

Результат:

```text
dist\raschet-qt.exe
```

---

## Как пользоваться выбором участка мышью

1. Включите чекбокс **Режим выбора участка мышью**.
2. На графике зажмите ЛКМ и протяните по оси X.
3. Интервал появится:
   - в полях `Начало` / `Конец`,
   - как зелёная область на графике,
   - в таблице предпросмотра участка.
4. Нажмите `Построить участок`, если хотите отрисовать только этот диапазон.
