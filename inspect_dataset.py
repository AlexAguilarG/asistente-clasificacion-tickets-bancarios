"""Inspecciona y documenta el snapshot de Banking77 traducido al español.

Ejemplo:
    python src/inspect_dataset.py --revision main --output results/dataset_profile

La salida registra el SHA completo del snapshot de Hugging Face. Para repetir el
experimento con exactamente la misma versión, usa ese SHA en --revision tanto
para este script como para prepare_dataset.py y benchmark.py.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import datasets
import huggingface_hub
from datasets import load_dataset
from huggingface_hub import HfApi

DATASET_ID = "DeepPavlov/banking77-translated"
CONFIG = "es"
TEXT_FIELDS = ["text", "text_gemma", "text_qwen"]
LABEL_FIELD = "label"

# Nombres de intención oficiales ordenados por el ID 0..76 del BANKING77 original.
INTENTS = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed",
    "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin",
    "compromised_card", "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge", "exchange_rate",
    "exchange_via_app", "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card", "getting_virtual_card",
    "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten",
    "pending_card_payment", "pending_cash_withdrawal", "pending_top_up", "pending_transfer",
    "pin_blocked", "receiving_money", "Refund_not_showing_up", "request_refund",
    "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "top_up_by_cash_or_cheque",
    "top_up_failed", "top_up_limits", "top_up_reverted", "topping_up_by_card",
    "transaction_charged_twice", "transfer_fee_charged", "transfer_into_account",
    "transfer_not_received_by_recipient", "transfer_timing", "unable_to_verify_identity",
    "verify_my_identity", "verify_source_of_funds", "verify_top_up", "virtual_card_not_working",
    "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]

DESCRIPTIONS = {
    "text": "Consulta original en inglés del corpus BANKING77.",
    "label": "Identificador entero de la intención; IDs 0–76, asociados con las 77 clases del BANKING77 original.",
    "text_gemma": "Variante en español que la ficha del dataset identifica con el sufijo gemma; usada por defecto en este experimento. La ficha consultada no describe completamente el proceso de traducción.",
    "text_qwen": "Variante alternativa en español que la ficha identifica con el sufijo qwen; no se utiliza en el experimento base.",
}


def resolve_revision(revision: str) -> str:
    """Resuelve `main` u otra referencia a un SHA de commit inmutable."""
    return HfApi().dataset_info(repo_id=DATASET_ID, revision=revision).sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="Rama, tag o SHA del dataset (default: main).")
    parser.add_argument("--output", default="results/dataset_profile", help="Directorio de salida.")
    args = parser.parse_args()

    resolved_sha = resolve_revision(args.revision)
    print(f"Dataset: {DATASET_ID} / config={CONFIG} / revision={resolved_sha}")
    ds = load_dataset(DATASET_ID, CONFIG, revision=resolved_sha)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    split_summary = {}
    schema_rows = []
    for field in ds["train"].column_names:
        feature = ds["train"].features[field]
        schema_rows.append({
            "field": field,
            "type": str(feature),
            "description": DESCRIPTIONS.get(field, "Campo disponible en la versión consultada; revisar la ficha del dataset."),
            "used_in_baseline": field in {"text_gemma", "label"},
        })
    pd.DataFrame(schema_rows).to_csv(out / "dataset_schema.csv", index=False, encoding="utf-8-sig")

    for split_name in ds.keys():
        frame = ds[split_name].to_pandas()
        summary = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "unique_labels": int(frame[LABEL_FIELD].nunique()) if LABEL_FIELD in frame else None,
            "min_label": int(frame[LABEL_FIELD].min()) if LABEL_FIELD in frame and len(frame) else None,
            "max_label": int(frame[LABEL_FIELD].max()) if LABEL_FIELD in frame and len(frame) else None,
        }
        for text_field in TEXT_FIELDS:
            if text_field in frame:
                values = frame[text_field].astype("string")
                summary[f"{text_field}_missing"] = int(values.isna().sum())
                summary[f"{text_field}_empty_after_strip"] = int(values.fillna("").str.strip().eq("").sum())
                summary[f"{text_field}_duplicate_nonempty"] = int(values.dropna().duplicated().sum())
        split_summary[split_name] = summary

        if LABEL_FIELD in frame:
            counts = frame[LABEL_FIELD].value_counts(dropna=False).sort_index()
            distribution = pd.DataFrame({"label": counts.index.astype(int), "count": counts.values})
            distribution["intent"] = distribution["label"].map(lambda x: INTENTS[x] if 0 <= x < len(INTENTS) else f"unknown_{x}")
            distribution.to_csv(out / f"label_distribution_{split_name}.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "dataset_id": DATASET_ID,
        "config": CONFIG,
        "requested_revision": args.revision,
        "resolved_revision_sha": resolved_sha,
        "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
        "library_versions": {
            "python": platform.python_version(),
            "datasets": datasets.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "pandas": pd.__version__,
        },
        "splits": split_summary,
        "notes": [
            "Se descarga el snapshot público fijado por resolved_revision_sha.",
            "El experimento base utiliza text_gemma como entrada y label como objetivo.",
            "El dataset traducido no equivale a tickets reales de una entidad bancaria mexicana.",
            "La ficha pública consultada no especifica suficientemente el proceso de traducción ni una licencia propia inequívoca para esta versión traducida; verificar antes de redistribuir o usar comercialmente.",
        ],
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Resumen:")
    for split_name, summary in split_summary.items():
        print(f"  {split_name}: {summary['rows']:,} filas; {summary['unique_labels']} clases; label {summary['min_label']}–{summary['max_label']}")
    print("Campos:", ds["train"].column_names)
    print("SHA del dataset:", resolved_sha)
    print("Manifest y perfiles guardados en:", out.resolve())
    print("Reutiliza el SHA impreso para fijar el snapshot en los siguientes scripts.")


if __name__ == "__main__":
    main()
