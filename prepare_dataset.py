"""Descarga Banking77 traducido al español y exporta una muestra balanceada reproducible."""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import datasets
import huggingface_hub
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

DATASET_ID = "DeepPavlov/banking77-translated"
CONFIG = "es"
TEXT_FIELD = "text_gemma"
SEED = 42

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


def resolve_revision(revision: str) -> str:
    return HfApi().dataset_info(repo_id=DATASET_ID, revision=revision).sha


def normalize(split, text_field: str = TEXT_FIELD) -> pd.DataFrame:
    required = {text_field, "label"}
    if not required.issubset(set(split.column_names)):
        raise ValueError(f"Faltan campos {sorted(required - set(split.column_names))}; disponibles: {split.column_names}")
    df = pd.DataFrame(split[:])[[text_field, "label"]].rename(columns={text_field: "text"})
    df["text"] = df["text"].astype("string").str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 0].copy()
    df["label"] = df["label"].astype(int)
    if not df["label"].between(0, 76).all():
        raise ValueError("Se encontraron etiquetas fuera del rango 0–76.")
    df["intent"] = df["label"].map(lambda x: INTENTS[x])
    return df


def cap_per_class(df: pd.DataFrame, cap: int, seed: int = SEED) -> pd.DataFrame:
    if cap < 1:
        raise ValueError("El límite de ejemplos por clase debe ser mayor que cero.")
    chunks = [g.sample(n=min(cap, len(g)), random_state=seed) for _, g in df.groupby("label", sort=True)]
    return pd.concat(chunks, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-per-class", type=int, default=80)
    ap.add_argument("--test-per-class", type=int, default=25)
    ap.add_argument("--revision", default="main", help="Rama, tag o SHA del dataset; preferir el SHA de inspect_dataset.py.")
    ap.add_argument("--output", default="data/processed")
    args = ap.parse_args()

    resolved_sha = resolve_revision(args.revision)
    ds = load_dataset(DATASET_ID, CONFIG, revision=resolved_sha)
    train_raw = normalize(ds["train"])
    test_raw = normalize(ds["test"])
    train = cap_per_class(train_raw, args.train_per_class)
    test = cap_per_class(test_raw, args.test_per_class)
    if set(train.label) != set(test.label):
        raise ValueError("Train y test no contienen las mismas clases. Aumenta test-per-class o revisa el dataset.")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train_es.csv", index=False, encoding="utf-8-sig")
    test.to_csv(out / "test_es.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "dataset_id": DATASET_ID,
        "config": CONFIG,
        "requested_revision": args.revision,
        "resolved_revision_sha": resolved_sha,
        "text_field": TEXT_FIELD,
        "label_field": "label",
        "seed": SEED,
        "max_train_per_class": args.train_per_class,
        "max_test_per_class": args.test_per_class,
        "n_train_source_after_cleaning": int(len(train_raw)),
        "n_test_source_after_cleaning": int(len(test_raw)),
        "n_train_exported": int(len(train)),
        "n_test_exported": int(len(test)),
        "n_classes_train": int(train.label.nunique()),
        "n_classes_test": int(test.label.nunique()),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": platform.python_version(),
            "datasets": datasets.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "pandas": pd.__version__,
        },
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exportados train={len(train):,}, test={len(test):,}, clases={train.label.nunique()} en {out.resolve()}")
    print("Snapshot inmutable:", resolved_sha)
    print("Semilla fija:", SEED)
    print("Manifest:", (out / "dataset_manifest.json").resolve())


if __name__ == "__main__":
    main()
