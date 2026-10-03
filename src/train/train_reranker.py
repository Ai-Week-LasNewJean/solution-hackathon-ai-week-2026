"""Etapa 3: ajuste fino de bge-reranker-v2-m3 con las preguntas sinteticas
(positivo = fragmento de origen, negativos = etapa 2), perdida binaria.

Checkpoints cada --save-steps en models/reranker-ft/checkpoint-*; al relanzar
se reanuda desde el ultimo (optimizador y scheduler incluidos) salvo --fresh.
El modelo final queda en models/reranker-ft/final.

    python -m src.train.train_reranker [--fresh] [--epochs 1] [--lr 1e-5]
"""
from __future__ import annotations

import argparse
import json
import shutil

from src import config
from src.train.common import EVAL_LOG_PATH, MODEL_DIR, PAIRS_PATH, chunks_by_key, read_jsonl

BASE_MODEL = "BAAI/bge-reranker-v2-m3"
TRAIN_MAX_LEN = 384
HOLDOUT_EVERY = 20   # 5% de las preguntas (por llave) para el evaluador interno


def build_datasets():
    from datasets import Dataset

    chunks = chunks_by_key()
    rows = [r for r in read_jsonl(PAIRS_PATH) if len(r["negs"]) >= 3 and r["key"] in chunks]
    train = {"query": [], "passage": [], "label": []}
    holdout = []
    for r in rows:
        pos = chunks[r["key"]]["texto"]
        negs = [chunks[k]["texto"] for k in r["negs"] if k in chunks]
        if int(r["key"], 16) % HOLDOUT_EVERY == 0:
            holdout.append({"query": r["query"], "positive": [pos], "documents": [pos] + negs})
            continue
        for texto, label in [(pos, 1.0)] + [(n, 0.0) for n in negs]:
            train["query"].append(r["query"])
            train["passage"].append(texto)
            train["label"].append(label)
    return Dataset.from_dict(train).shuffle(seed=13), holdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="ignora checkpoints previos")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--save-steps", type=int, default=200)
    args = ap.parse_args()

    import torch
    from sentence_transformers.cross_encoder import (CrossEncoder, CrossEncoderTrainer,
                                                     CrossEncoderTrainingArguments)
    from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from transformers.trainer_utils import get_last_checkpoint

    if args.fresh and MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
        # las rutas checkpoint-N se van a reutilizar: olvidar sus evaluaciones viejas
        keep = [r for r in read_jsonl(EVAL_LOG_PATH) if not r["model"].startswith(str(MODEL_DIR))]
        EVAL_LOG_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    last = get_last_checkpoint(str(MODEL_DIR))
    print(f"reanudando desde {last}" if last else "entrenamiento nuevo", flush=True)

    train_ds, holdout = build_datasets()
    n_pos = sum(train_ds["label"])
    print(f"{len(train_ds)} pares de entrenamiento ({int(n_pos)} positivos); {len(holdout)} preguntas "
          f"de validacion", flush=True)

    model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=TRAIN_MAX_LEN)
    pos_weight = torch.tensor((len(train_ds) - n_pos) / max(n_pos, 1))
    loss = BinaryCrossEntropyLoss(model, pos_weight=pos_weight)
    evaluator = CrossEncoderRerankingEvaluator(samples=holdout, at_k=6, name="synth_holdout",
                                               batch_size=32)
    targs = CrossEncoderTrainingArguments(
        output_dir=str(MODEL_DIR), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, warmup_ratio=0.1, bf16=True,
        eval_strategy="steps", eval_steps=args.save_steps, save_strategy="steps",
        save_steps=args.save_steps, save_total_limit=6, logging_steps=50, seed=13,
        report_to="none")
    trainer = CrossEncoderTrainer(model=model, args=targs, train_dataset=train_ds, loss=loss,
                                  evaluator=evaluator)
    trainer.train(resume_from_checkpoint=last)
    model.save_pretrained(str(MODEL_DIR / "final"))
    print(f"modelo final en {MODEL_DIR / 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
