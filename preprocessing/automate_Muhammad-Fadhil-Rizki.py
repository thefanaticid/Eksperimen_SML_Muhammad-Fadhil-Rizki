"""Automasi preprocessing California Housing dataset.

Mereplikasi seluruh tahap preprocessing dari `Eksperimen_Muhammad-Fadhil-Rizki.ipynb`
dalam bentuk fungsi modular yang dapat dipanggil dari pipeline (mis. GitHub Actions).

Penulis  : Muhammad Fadhil Rizki
Username : fadhilspooky
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger("automate_preprocessing")
RANDOM_STATE = 42
SKEWED_FEATURES = ["AveRooms", "AveBedrms", "AveOccup", "Population", "TotalBedrooms_proxy"]
TARGET = "MedHouseValue"


def load_raw(raw_path: Path | None) -> pd.DataFrame:
    """Muat dataset mentah. Jika file tidak ada, fetch dari sklearn dan simpan."""
    if raw_path is not None and raw_path.exists():
        LOGGER.info("Memuat raw dataset dari %s", raw_path)
        return pd.read_csv(raw_path)

    LOGGER.info("Raw file tidak ditemukan, fetch dari sklearn.fetch_california_housing")
    bunch = fetch_california_housing(as_frame=True)
    df = bunch.frame.rename(columns={"MedHouseVal": TARGET}).copy()
    df["TotalBedrooms_proxy"] = df["AveBedrms"] * df["Population"] / df["AveOccup"]
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
        LOGGER.info("Raw dataset disimpan ke %s", raw_path)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    LOGGER.info("Hapus %d duplikat (sisa %d baris)", before - len(out), len(out))
    return out


def impute_missing(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != target]
    imputer = SimpleImputer(strategy="median")
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    LOGGER.info("Imputasi median selesai. Sisa NaN: %d", int(df.isnull().sum().sum()))
    return df


def cap_outliers(df: pd.DataFrame, columns: list[str], k: float = 1.5) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        df[col] = df[col].clip(lo, hi)
    LOGGER.info("Outlier capping (IQR k=%.1f) pada %d kolom", k, len(columns))
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df["RoomsPerHousehold"] = df["AveRooms"]
    df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"]
    df["PopulationPerHousehold"] = df["AveOccup"]
    df["IncomePerRoom"] = df["MedInc"] / (df["AveRooms"] + 1e-6)
    LOGGER.info("Feature engineering selesai (4 fitur baru)")
    return df


def add_income_category(df: pd.DataFrame) -> pd.DataFrame:
    df["IncomeCategory"] = pd.cut(
        df["MedInc"], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5]
    ).astype(int)
    return df


def split_and_scale(df: pd.DataFrame, target: str = TARGET, test_size: float = 0.2):
    df = add_income_category(df)
    y = df[target]
    X = df.drop(columns=[target, "IncomeCategory"])
    strat = df["IncomeCategory"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=strat
    )
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    LOGGER.info("Split selesai. Train=%s | Test=%s", X_train_s.shape, X_test_s.shape)
    return X_train_s, X_test_s, y_train, y_test, scaler


def save_outputs(out_dir: Path, X_train, X_test, y_train, y_test) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    train_full = X_train.copy()
    train_full[TARGET] = y_train.values
    test_full = X_test.copy()
    test_full[TARGET] = y_test.values
    train_full.to_csv(out_dir / "housing_train.csv", index=False)
    test_full.to_csv(out_dir / "housing_test.csv", index=False)
    LOGGER.info("Output preprocessing tersimpan di %s", out_dir.resolve())


def preprocess_pipeline(raw_path: Path | None, out_dir: Path) -> dict:
    """End-to-end pipeline: load → clean → engineer → split → scale → save.

    Returns ringkasan metadata untuk logging/audit.
    """
    df = load_raw(raw_path)
    df = remove_duplicates(df)
    df = impute_missing(df)
    df = cap_outliers(df, SKEWED_FEATURES)
    df = add_engineered_features(df)
    X_train, X_test, y_train, y_test, _ = split_and_scale(df)
    save_outputs(out_dir, X_train, X_test, y_train, y_test)
    return {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "feature_names": list(X_train.columns),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Automate California Housing preprocessing")
    p.add_argument("--raw", type=Path, default=Path("housing_raw/housing.csv"),
                   help="Path ke file CSV mentah. Akan dibuat dari sklearn jika tidak ada.")
    p.add_argument("--out", type=Path, default=Path("housing_preprocessing"),
                   help="Direktori output dataset siap latih.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    summary = preprocess_pipeline(args.raw, args.out)
    LOGGER.info("Pipeline selesai: %s", summary)


if __name__ == "__main__":
    main()
