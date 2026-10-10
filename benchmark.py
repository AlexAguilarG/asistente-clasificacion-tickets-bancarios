"""Benchmark reproducible de tres codificadores multilingües para BANKING77 traducido.

Desde la raíz del repositorio:
python src/benchmark.py --revision <SHA_COMPLETO> --train-per-class 80 --test-per-class 25 --output results/run_latest

Incluye métricas globales, latencia por lote, latencia individual p50/p95/p99,
reportes por clase, pares de confusión y análisis exploratorio de umbrales.
"""
from __future__ import annotations
import argparse, json, platform, random, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, torch, datasets, huggingface_hub
from datasets import load_dataset
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)

DATASET_ID = "DeepPavlov/banking77-translated"
DATASET_CONFIG = "es"
TEXT_FIELD = "text_gemma"
SEED = 42
MODEL_SPECS = [
    {"id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "name": "MiniLM-L12-v2", "dim": 384, "license": "Apache-2.0"},
    {"id": "sentence-transformers/distiluse-base-multilingual-cased-v2", "name": "DistilUSE multilingual v2", "dim": 512, "license": "Apache-2.0"},
    {"id": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "name": "MPNet multilingual v2", "dim": 768, "license": "Apache-2.0"},
]
# Intenciones del benchmark con correspondencia al orden canónico de PolyAI/BANKING77.
INTENTS = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support", "automatic_top_up",
    "balance_not_updated_after_bank_transfer", "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed", "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card", "contactless_not_working", "country_support",
    "declined_card_payment", "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app",
    "extra_charge_on_statement", "failed_transfer", "fiat_currency_support", "get_disposable_virtual_card",
    "get_physical_card", "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone",
    "order_physical_card", "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal", "pending_top_up",
    "pending_transfer", "pin_blocked", "receiving_money", "Refund_not_showing_up", "request_refund",
    "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account", "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
    "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged", "transfer_into_account",
    "transfer_not_received_by_recipient", "transfer_timing", "unable_to_verify_identity", "verify_my_identity",
    "verify_source_of_funds", "verify_top_up", "virtual_card_not_working", "visa_or_mastercard", "why_verify_identity",
    "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal"
]
SENSITIVE_INTENTS = {
    "card_payment_not_recognised", "cash_withdrawal_not_recognised", "direct_debit_payment_not_recognised",
    "compromised_card", "lost_or_stolen_card", "lost_or_stolen_phone", "transaction_charged_twice",
    "transfer_not_received_by_recipient"
}
THRESHOLDS = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

def sync(device: str) -> None:
    if device == "cuda": torch.cuda.synchronize()

def balanced_cap(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    groups = [g.sample(n=min(cap, len(g)), random_state=seed) for _, g in df.groupby("label", sort=True)]
    return pd.concat(groups, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

def load_frame(split, text_field: str) -> pd.DataFrame:
    cols = set(split.column_names)
    if text_field not in cols or "label" not in cols:
        raise ValueError(f"Se requieren {text_field!r} y 'label'; columnas disponibles: {split.column_names}")
    df = pd.DataFrame(split[:])[[text_field, "label"]].rename(columns={text_field: "text"})
    df["text"] = df["text"].astype("string").str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 0].copy()
    df["label"] = df["label"].astype(int)
    if not df["label"].between(0, 76).all():
        raise ValueError("Se detectaron etiquetas fuera del rango esperado 0–76.")
    df["intent_true"] = df["label"].map(lambda i: INTENTS[i])
    return df

def latency_single_ticket(encoder, clf, texts, device: str, sample_size: int, warmup: int) -> dict:
    if not texts:
        return {"latency_single_n": 0, "latency_single_mean_ms": np.nan,
                "latency_single_p50_ms": np.nan, "latency_single_p95_ms": np.nan, "latency_single_p99_ms": np.nan}
    rng = np.random.default_rng(SEED)
    n = min(sample_size, len(texts))
    ids = rng.choice(len(texts), size=n, replace=False)
    warm_texts = texts[:min(warmup, len(texts))]
    for text in warm_texts:
        _ = encoder.encode([text], batch_size=1, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        sync(device)
    timings = []
    for i in ids:
        text = texts[int(i)]
        sync(device)
        t0 = time.perf_counter()
        emb = encoder.encode([text], batch_size=1, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        probs = clf.predict_proba(emb)
        _ = int(np.argmax(probs[0]))
        sync(device)
        timings.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(timings, dtype=float)
    return {"latency_single_n": int(len(arr)), "latency_single_mean_ms": float(np.mean(arr)),
            "latency_single_p50_ms": float(np.percentile(arr, 50)),
            "latency_single_p95_ms": float(np.percentile(arr, 95)),
            "latency_single_p99_ms": float(np.percentile(arr, 99))}

def evaluate_model(spec, train_df, test_df, args, device, output_dir):
    print(f"\n=== {spec['name']} ===")
    if device == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter(); encoder = SentenceTransformer(spec["id"], device=device); sync(device)
    load_seconds = time.perf_counter() - t0
    parameters = int(sum(p.numel() for p in encoder.parameters()))
    t0 = time.perf_counter()
    X_train = encoder.encode(train_df.text.tolist(), batch_size=args.batch_size, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=True)
    sync(device); train_embedding_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    X_test = encoder.encode(test_df.text.tolist(), batch_size=args.batch_size, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=True)
    sync(device); test_embedding_seconds = time.perf_counter() - t0
    clf = LogisticRegression(max_iter=300, class_weight="balanced", random_state=SEED)
    t0 = time.perf_counter(); clf.fit(X_train, train_df.label.to_numpy()); classifier_fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = clf.predict(X_test); sync(device); classifier_predict_seconds = time.perf_counter() - t0
    probabilities = clf.predict_proba(X_test)
    confidence = probabilities.max(axis=1)
    y_true = test_df.label.to_numpy()
    y_pred = pred.astype(int)
    names_true = [INTENTS[int(x)] for x in y_true]
    names_pred = [INTENTS[int(x)] for x in y_pred]
    slug = spec["name"].lower().replace(" ", "_").replace("-", "_")
    prediction_df = pd.DataFrame({"text": test_df.text.to_list(), "label_true": y_true,
                                  "intent_true": names_true, "label_pred": y_pred,
                                  "intent_pred": names_pred, "confidence": confidence,
                                  "correct": y_true == y_pred})
    prediction_df.to_csv(output_dir / f"predictions_{slug}.csv", index=False, encoding="utf-8-sig")
    report = classification_report(y_true, y_pred, labels=list(range(77)), target_names=INTENTS,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(output_dir / f"classification_report_{slug}.csv", encoding="utf-8-sig")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(77)))
    pd.DataFrame(cm, index=INTENTS, columns=INTENTS).to_csv(output_dir / f"confusion_matrix_{slug}.csv", encoding="utf-8-sig")
    pair_counts = pd.DataFrame({"intent_true": names_true, "intent_pred": names_pred})
    pair_counts = (pair_counts[pair_counts["intent_true"] != pair_counts["intent_pred"]]
                   .groupby(["intent_true", "intent_pred"], as_index=False).size()
                   .rename(columns={"size": "count"})
                   .sort_values("count", ascending=False))
    pair_counts.head(50).to_csv(output_dir / f"top_confusion_pairs_{slug}.csv", index=False, encoding="utf-8-sig")
    singles = latency_single_ticket(encoder, clf, test_df.text.tolist(), device, args.latency_sample_size, args.warmup_samples)
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else np.nan
    row = {
        "model": spec["name"], "model_id": spec["id"], "embedding_dim": spec["dim"], "license": spec["license"],
        "parameters": parameters, "parameters_millions": parameters / 1e6, "train_samples": len(train_df), "test_samples": len(test_df),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "model_load_seconds": load_seconds, "train_embedding_seconds": train_embedding_seconds,
        "classifier_fit_seconds": classifier_fit_seconds, "test_embedding_seconds": test_embedding_seconds,
        "classifier_predict_seconds": classifier_predict_seconds,
        "batch_amortized_ms_per_ticket": 1000 * (test_embedding_seconds + classifier_predict_seconds) / len(test_df),
        **singles, "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "gpu_peak_allocated_mb": peak_memory,
    }
    del encoder, X_train, X_test, clf
    return row, prediction_df

def summarize_confidence(prediction_frames: dict, output_dir: Path) -> None:
    rows = []
    for model, df in prediction_frames.items():
        y_true = df["label_true"].to_numpy()
        y_pred = df["label_pred"].to_numpy()
        conf = df["confidence"].to_numpy(float)
        pred_intents = df["intent_pred"].to_numpy()
        sensitive = np.isin(pred_intents, list(SENSITIVE_INTENTS))
        correct = y_true == y_pred
        for threshold in THRESHOLDS:
            for policy in ("confidence_only", "sensitive_intents_to_human"):
                accepted = conf >= threshold
                if policy == "sensitive_intents_to_human": accepted &= ~sensitive
                n = int(accepted.sum())
                precision = float(correct[accepted].mean()) if n else np.nan
                rows.append({"model": model, "threshold": threshold, "policy": policy,
                             "n_test": len(df), "accepted_n": n, "coverage": n / len(df),
                             "human_review_rate": 1 - n / len(df), "accepted_precision": precision,
                             "accepted_error_rate": 1 - precision if n else np.nan,
                             "incorrect_suggestions_as_pct_all_tickets": float((accepted & ~correct).sum() / len(df)),
                             "sensitive_predicted_n": int(sensitive.sum())})
    pd.DataFrame(rows).to_csv(output_dir / "confidence_threshold_analysis.csv", index=False, encoding="utf-8-sig")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="Rama, etiqueta o SHA; registra el SHA resuelto en el manifiesto.")
    parser.add_argument("--text-field", default=TEXT_FIELD, help="Campo de texto de entrada; por defecto text_gemma.")
    parser.add_argument("--train-per-class", type=int, default=80)
    parser.add_argument("--test-per-class", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latency-sample-size", type=int, default=300)
    parser.add_argument("--warmup-samples", type=int, default=10)
    parser.add_argument("--output", default="results/run_latest")
    args = parser.parse_args()
    if min(args.train_per_class, args.test_per_class, args.batch_size, args.latency_sample_size) < 1:
        parser.error("train/test per class, batch size y latency sample size deben ser positivos.")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_sha = HfApi().dataset_info(repo_id=DATASET_ID, revision=args.revision).sha
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, revision=resolved_sha)
    train_all, test_all = load_frame(dataset["train"], args.text_field), load_frame(dataset["test"], args.text_field)
    train = balanced_cap(train_all, args.train_per_class, SEED)
    test = balanced_cap(test_all, args.test_per_class, SEED)
    if set(train.label) != set(test.label):
        raise ValueError("Train/test deben contener el mismo conjunto de etiquetas; aumenta test-per-class.")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    results, predictions = [], {}
    for spec in MODEL_SPECS:
        row, pred_df = evaluate_model(spec, train, test, args, device, out)
        results.append(row); predictions[spec["name"]] = pred_df
        pd.DataFrame({"id": range(77), "intent": INTENTS}).to_csv(out / "label_mapping.csv", index=False, encoding="utf-8-sig")
    result_df = pd.DataFrame(results).sort_values(["f1_macro", "latency_single_p95_ms"], ascending=[False, True])
    result_df.to_csv(out / "model_comparison.csv", index=False, encoding="utf-8-sig")
    summarize_confidence(predictions, out)
    manifest = {
        "dataset_id": DATASET_ID, "config": DATASET_CONFIG, "text_field": args.text_field,
        "requested_revision": args.revision, "resolved_revision_sha": resolved_sha,
        "seed": SEED, "train_per_class": args.train_per_class, "test_per_class": args.test_per_class,
        "train_samples_observed": len(train), "test_samples_observed": len(test),
        "batch_size_for_embeddings": args.batch_size, "latency_sample_size": args.latency_sample_size,
        "warmup_samples": args.warmup_samples, "model_ids": [x["id"] for x in MODEL_SPECS],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "torch": torch.__version__, "datasets": datasets.__version__, "huggingface_hub": huggingface_hub.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "latency_note": "La latencia individual incluye encoder+clasificador, se mide secuencialmente con batch_size=1 tras warm-up. No incluye carga del modelo, red, colas ni servicio web."
    }
    (out / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nRESULTADOS GLOBALES")
    cols = ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "batch_amortized_ms_per_ticket", "latency_single_p50_ms", "latency_single_p95_ms", "latency_single_p99_ms"]
    print(result_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nArtefactos guardados en: {out.resolve()}")

if __name__ == "__main__":
    main()
