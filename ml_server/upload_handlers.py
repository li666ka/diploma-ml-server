"""Chunked upload handling: /upload_chunk + /upload_finalize."""
import os
import shutil
import zipfile
from pathlib import Path

from flask import jsonify, request

from ml_server.config import CHUNKS_ROOT, DATASETS_ROOT
from ml_server.utils import log


def _chunks_dir(dataset_id: str) -> Path:
    p = Path(CHUNKS_ROOT) / dataset_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def handle_upload_chunk():
    """POST /upload_chunk — receive single chunk."""
    dataset_id = request.form.get("dataset_id")
    chunk_idx = request.form.get("chunk_idx")
    total_chunks = request.form.get("total_chunks")
    chunk_file = request.files.get("chunk")

    if not all([dataset_id, chunk_idx, total_chunks, chunk_file]):
        return jsonify({"error": "Missing required fields"}), 400

    chunks_dir = _chunks_dir(dataset_id)
    chunk_path = chunks_dir / f"chunk_{chunk_idx}.bin"
    chunk_file.save(chunk_path)

    chunks_received = len(list(chunks_dir.glob("chunk_*.bin")))
    log.info(
        f"Chunk {chunk_idx}/{total_chunks} for dataset {dataset_id} received "
        f"({chunks_received} total)"
    )

    return jsonify({
        "ok": True,
        "chunks_received": chunks_received,
        "total_chunks": int(total_chunks),
    })


def finalize_upload(dataset_id: str) -> tuple[bool, str, dict]:
    """Об'єднати чанки → ZIP → розпакувати в DATASETS_ROOT."""
    chunks_dir = _chunks_dir(dataset_id)
    chunks = sorted(
        chunks_dir.glob("chunk_*.bin"),
        key=lambda p: int(p.stem.split("_")[1]),
    )

    if not chunks:
        return False, f"No chunks for dataset {dataset_id}", {}

    log.info(f"Finalizing dataset {dataset_id}: combining {len(chunks)} chunks")
    target_dir = Path(DATASETS_ROOT) / str(dataset_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Combine chunks → temp ZIP
    combined_path = chunks_dir / "combined.zip"
    with open(combined_path, "wb") as out:
        for chunk_path in chunks:
            with open(chunk_path, "rb") as cf:
                shutil.copyfileobj(cf, out)

    # Extract
    try:
        with zipfile.ZipFile(combined_path, "r") as zf:
            zf.extractall(target_dir)
        files = [f.name for f in target_dir.iterdir() if f.is_file()]
        log.info(f"Extracted {len(files)} files to {target_dir}")
    except zipfile.BadZipFile as e:
        return False, f"Invalid ZIP: {e}", {}

    # Cleanup chunks
    shutil.rmtree(chunks_dir, ignore_errors=True)

    return True, "Upload finalized", {
        "target_dir": str(target_dir),
        "files": files,
        "chunks_combined": len(chunks),
    }


def handle_upload_finalize():
    """POST /upload_finalize."""
    payload = request.json or {}
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        return jsonify({"error": "dataset_id required"}), 400

    success, message, stats = finalize_upload(str(dataset_id))
    if success:
        return jsonify({"ok": True, "message": message, **stats})
    return jsonify({"ok": False, "error": message, **stats}), 400


def list_datasets():
    """GET /list_datasets."""
    root = Path(DATASETS_ROOT)
    if not root.exists():
        return jsonify({"datasets": []})

    datasets = []
    for d in root.iterdir():
        if d.is_dir():
            datasets.append({
                "dataset_id": d.name,
                "files": [f.name for f in d.iterdir() if f.is_file()],
            })
    return jsonify({"datasets": datasets, "root": str(root)})


def dataset_status():
    """GET /dataset_status?dataset_id=X."""
    dataset_id = request.args.get("dataset_id")
    if not dataset_id:
        return jsonify({"error": "dataset_id required"}), 400

    target = Path(DATASETS_ROOT) / str(dataset_id)
    if not target.exists():
        return jsonify({"exists": False, "dataset_id": dataset_id})

    files = [f.name for f in target.iterdir() if f.is_file()]
    has_news = "news.csv" in files
    return jsonify({
        "exists": has_news,
        "dataset_id": dataset_id,
        "files": files,
        "path": str(target),
    })
