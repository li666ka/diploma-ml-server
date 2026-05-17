# diploma-ml-server

ML server для дипломної роботи з детекції фейкових новин. Працює на Colab
через Flask + ngrok, обслуговує FastAPI gateway (`diploma/api`).

## Запуск на Colab

1. Відкрити `colab_runner.ipynb` у Colab.
2. Cells по порядку:
   - Mount Google Drive
   - `!cd /content/diploma-ml-server && git pull`
   - **Після `git pull` або Runtime → Restart**: `!pip install -r /content/diploma-ml-server/requirements.txt`
     (саме після рестарту, інакше старі package versions залишаються
     підвантажені у пам'яті — captum/torch-geometric не побачать оновлень).
   - `from ml_server.app import start_server; start_server()` — стартує
     Flask + ngrok tunnel. URL друкується у логах.
3. Скопіювати `COLAB_NGROK_URL=…` у `.env` FastAPI.

## Endpoints

| Route | Метод | Призначення |
|---|---|---|
| `/health` | GET | Liveness + чи `distilbert_in_memory`, glob моделей |
| `/predict_nb` | POST | Inference NB bundle (.pkl) |
| `/predict_distilbert` | POST | Inference DistilBERT (lazy-load з Drive) |
| `/predict_gnn` | POST | Inference GIN/SAGE |
| `/explain_nb` | POST | Local explanation NB (log-odds per token + handcrafted features) |
| `/explain_distilbert` | POST | Local explanation DistilBERT через captum IG |
| `/explain_gnn` | POST | GNNExplainer для GIN/SAGE (cached 5 хв) |
| `/debug_bundle` | POST | Inspect .pkl bundle структуру (для діагностики) |
| `/run_training_async` | POST | Async тренування (NB/DistilBERT/GNN) |
| `/training_status/<job_id>` | GET | Polling прогресу |

## Troubleshooting

- **`captum_not_installed` від `/explain_distilbert`**: виконати у Colab cell
  `!pip install captum --quiet`, потім **Runtime → Restart current cell** і
  перезапустити Flask cell.
- **`model_dir_not_found` від `/predict_distilbert`**: модель була збережена
  у `/content/...` (local disk Colab) і втрачена при restart. Перетренувати
  з `MODELS_ROOT=/content/drive/...` (default — на Drive).
- **Ngrok 502 на стороні FastAPI**: `GET <ngrok_url>/health` напряму; якщо
  HTML — tunnel впав, перезапустити Flask cell.
