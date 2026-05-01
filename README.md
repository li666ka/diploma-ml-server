# diploma-ml-server

ML сервер для дипломної роботи з детекції фейкових новин.

Архітектура: код пишеться локально у VSCode/Cursor, push в git,
а на Colab/RunPod запускається тонкий notebook який робить `git clone`
+ `git pull` і стартує Flask сервер з ngrok тунелем.

## Структура

```
diploma-ml-server/
├── ml_server/
│   ├── __init__.py
│   ├── config.py              # paths, defaults
│   ├── data_loader.py         # resolve_dataset_path, build_article_level_data
│   ├── aggregated_loader.py   # старий aggregated pipeline (для NB)
│   ├── encoder.py             # MiniLM lazy-load + encode helpers
│   ├── graph_builder.py       # PyG graph construction
│   ├── nb_trainer.py          # train_nb (aggregated)
│   ├── distilbert_trainer.py  # train_distilbert_article_level
│   ├── gnn_models.py          # GraphSAGE + GIN
│   ├── gnn_trainer.py         # train_gnn end-to-end
│   ├── upload_handlers.py     # /upload_chunk + /upload_finalize
│   ├── routes.py              # всі Flask @app.route
│   ├── app.py                 # Flask app factory + start_server()
│   └── utils.py               # logging, preprocessing, metrics
├── scripts/
│   ├── train_local.py         # локальне тренування (без Flask)
│   └── test_inference.py      # тестовий запит до запущеного сервера
├── tests/
│   └── test_data_loader.py    # smoke tests
├── colab_runner.ipynb         # тонкий ноутбук для Colab (5 клітинок)
├── requirements.txt
├── .gitignore
└── README.md
```

## Як працювати

### Локально (для розробки і дебагу)

```bash
# 1. Клонуй репозиторій
git clone https://github.com/USERNAME/diploma-ml-server.git
cd diploma-ml-server

# 2. Створи venv і встанови залежності
python -m venv venv
source venv/bin/activate  # Linux/Mac
# або: venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 3. Запусти Flask локально (без GPU — повільно)
python -m ml_server.app

# Або тренуй модель напряму через скрипт:
python scripts/train_local.py --model gnn --dataset-path ./data/gossipcop
```

### На Colab (production-like з GPU)

1. Відкрий `colab_runner.ipynb` в Colab
2. Runtime → Change runtime type → T4 GPU
3. Запусти всі клітинки
4. Скопіюй ngrok URL у `.env` свого application backend

Після кожного `git push` локально — у Colab просто `!git pull` і
`%load_ext autoreload`. Жодних копі-паст, ніяких "забутих імпортів".

## Endpoints

- `GET /health` — ping
- `GET /list_datasets` — які датасети є на Drive
- `GET /dataset_status?dataset_id=X` — конкретний датасет
- `POST /upload_chunk` — chunked upload від backend
- `POST /upload_finalize` — розпакування ZIP в Drive
- `POST /run_training` — синхронне тренування (legacy)
- `POST /run_training_async` — асинхронне (рекомендується)
- `GET  /training_status/<job_id>` — статус async job
- `POST /run_evaluation` / `/run_evaluation_async`
- `POST /predict_nb` / `/predict_distilbert` / `/predict_gnn`
- `POST /analyze_dataset` — analytics (для UI Analytics modal)

## Розробка

### Як додати нову модель

1. Створи `ml_server/<model_name>_trainer.py` з функцією `train_X(...)`
2. Додай гілку у `ml_server/routes.py` → `run_training()`
3. (Опційно) додай predict endpoint
4. `git commit && git push`
5. У Colab `!git pull` і перезапусти Flask клітинку

### Як дебажити

Локально через VSCode debugger — встав breakpoint у `routes.py`,
запусти `python -m ml_server.app`, з іншого терміналу:
```bash
curl -X POST http://localhost:5050/health
```

Або через `scripts/test_inference.py` — швидкі smoke-тести.
