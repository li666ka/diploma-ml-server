# Інструкція з переносу

## Чому це краще ніж великий notebook

Зараз у тебе:
- Один Jupyter notebook на ~3000+ рядків з 13 клітинками
- Якщо забути запустити Cell 5 (imports) — Cell 8 (routes) ламається
- Дебагити неможливо: нема breakpoint'ів, нема autocomplete, типи не перевіряються
- Кожний `git diff` — це нечитабельний JSON

Тепер буде:
- Окремий git репозиторій з нормальною структурою .py файлів
- VSCode/Cursor — autocomplete, типи, debugger, рефакторинг
- Кожен модуль 100-300 рядків з чіткою відповідальністю
- Colab — 5 коротких клітинок, які просто `git pull` і `start_server()`

## Покрокова інструкція

### Крок 1: Створи новий GitHub репозиторій

1. Зайди на github.com → New repository
2. Назви `diploma-ml-server` (або як хочеш)
3. Public або Private — не важливо для тебе
4. Не додавай README/gitignore — у нас вже є

### Крок 2: Розпакуй цей шаблон

```bash
# Завантаж diploma-ml-server.zip який я тобі дам
# Розпакуй у папку поряд з твоїм основним проектом

cd ~/Projects  # де у тебе живуть проекти
unzip diploma-ml-server.zip
cd diploma-ml-server

# Init git і push
git init
git add .
git commit -m "Initial: ML server skeleton"
git branch -M main
git remote add origin https://github.com/USERNAME/diploma-ml-server.git
git push -u origin main
```

### Крок 3: Перенеси повну логіку train_nb з v16 ноутбука

У цьому шаблоні `ml_server/nb_trainer.py` має **спрощену версію** train_nb 
без emotional/stylistic features. Ти можеш:

**Варіант A**: Залишити як є — ComplementNB + TF-IDF, цього достатньо для бейзлайну.

**Варіант B (рекомендую)**: Перенести повний код з v16 ноутбука Cell 7 
(функція `train_nb`) у `ml_server/nb_trainer.py`. Заодно перенеси Cell 4 
(NRC lexicon) у новий файл `ml_server/features.py`. Це тобі вже легко 
зробити локально у редакторі.

### Крок 4: Перевір локально

```bash
cd diploma-ml-server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Smoke test
python tests/test_smoke.py

# Якщо є локальний датасет (поклади у data/datasets/1/news.csv etc.)
python scripts/train_local.py --model nb --dataset-id 1 --epochs 1
```

### Крок 5: Запусти на Colab

1. Відкрий `colab_runner.ipynb` у Colab
2. У Cell 1 заміни `REPO_URL` на свій
3. Runtime → T4 GPU
4. Run all
5. Скопіюй ngrok URL з виводу Cell 5 у `.env` свого основного backend:
   ```
   COLAB_NGROK_URL=https://abc123.ngrok-free.app
   IS_COLAB=true
   ```

### Крок 6: Інтегрувати з основним застосунком

Твій застосунок (FastAPI бекенд) вже знає як ходити на Colab — 
endpoint той самий (`/run_training`, `/predict_distilbert`, etc.), 
структура payload'ів та сама. **Нічого міняти на бекенді не треба** — 
просто оновлюєш ngrok URL у `.env`.

## Workflow після setup

### Звичайний день

1. Відкриваєш `diploma-ml-server` у Cursor/VSCode
2. Міняєш код у `.py` файлі (наприклад `gnn_trainer.py`)
3. `git push`
4. На Colab: перезапускаєш Cell 1 (`!git pull`) + Cell 5 (`start_server()`)
5. Тестуєш через свій frontend

### Якщо щось зламалось

1. **Локально дебагиш:** `python -m ml_server.app` (без ngrok), 
   ставиш breakpoint у VSCode
2. Або `python scripts/train_local.py --model gnn --dataset-id 1` — 
   повний прогон без HTTP, видно весь stack trace

### Якщо Colab disconnected

1. Reconnect → ngrok URL зміниться
2. Cell 1 + Cell 5 → новий URL
3. Оновлюєш `.env` свого backend → готово

## Що ще можна додати

- `tests/test_data_loader.py` — перевірка що CSV правильно парсяться
- `Makefile` — short commands like `make test`, `make local-server`
- GitHub Actions — авто-тест на push
- Pre-commit hooks (`black`, `ruff`)

Але це все — якщо матимеш час. Базовий setup вище вже **на порядок** 
зручніший за теперішній великий notebook.

## Питання які можуть виникнути

**Q: А якщо я хочу швидко щось спробувати, навіщо коммітити?**

A: Можеш у Colab робити `%load_ext autoreload` і редагувати файли 
прямо там через Files panel. Але краще — миттєвий push з VSCode 
(`Ctrl+Shift+P` → `Git: Push`) занімає 2 секунди.

**Q: А якщо моделі великі, і я хочу зберігати їх у git?**

A: Не зберігай. У `.gitignore` вже є `*.pkl`, `*.pt`, `models/`. 
Моделі живуть на Drive, у git тільки код.

**Q: Як паралельно працювати — і зміни робити, і Colab крутить тренування?**

A: Тренування блокує тільки один сервер. Якщо змінив код — 
зроби `git push`. Коли тренування закінчиться, на Colab `git pull` 
підхопить твої зміни, не ламаючи поточну сесію.

**Q: Мій застосунок (api/colab_sync.py) синхронізує датасети по chunk'ах. 
Воно ще працюватиме?**

A: Так, я переніс upload_handlers зі старого ноутбука 1:1. Endpoints 
`/upload_chunk` і `/upload_finalize` є у новому сервері з тією самою 
логікою.

**Q: Що з NRC lexicon і emotional features?**

A: Я навмисно не переніс це у шаблон, бо там 400+ рядків коду. 
Перенеси `Cell 4` зі старого ноутбука у новий файл 
`ml_server/features.py`, потім імпортуй у `nb_trainer.py`.
