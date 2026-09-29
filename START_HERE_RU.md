# С чего начать

Проект уже создан. Тебе не нужно вручную добавлять папки и пустые файлы.

## 1. Открой проект

В Visual Studio Code выбери **File → Open Folder** и открой папку
`eeg-emotion-classification`.

## 2. Создай Python-окружение

Открой **Terminal → New Terminal** и выполни команды.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 3. Выбери интерпретатор

Нажми `Ctrl/Cmd + Shift + P`, введи **Python: Select Interpreter** и выбери
интерпретатор из папки `.venv`.

## 4. Добавь DEAP

Скопируй файлы `s01.dat` — `s32.dat` в:

```text
data/raw/deap/
```

Эти тяжёлые файлы специально не попадут в GitHub.

## 5. Проверь датасет

```bash
python scripts/inspect_deap.py --data-dir data/raw/deap --subject 1
```

После этого открой `notebooks/01_deap_exploration.ipynb` и запускай ячейки
сверху вниз. На этом первом шаге мы только проверяем данные — пока не добавляем
новую фильтрацию и не обучаем нейросеть.

## 6. Проверь проект

```bash
pytest
```

Ожидаемый результат: все тесты пройдены. Они проверяют создание меток,
4-секундные окна, разделение по участникам и форму выхода CNN.

