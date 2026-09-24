from __future__ import annotations

from pathlib import Path
import hashlib
import os
import pandas as pd


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_table(path: str | Path, sheet: str | None = None) -> pd.DataFrame:
    """Charge un CSV ou un classeur Excel sans modifier les donnees sources."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        book = pd.ExcelFile(path)
        chosen = sheet or ("Données" if "Données" in book.sheet_names else book.sheet_names[-1])
        df = pd.read_excel(path, sheet_name=chosen)
    elif suffix in {".csv", ".txt"}:
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                df = pd.read_csv(path, sep=None, engine="python", encoding=encoding)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
        else:
            raise last_error or ValueError("Impossible de lire le fichier CSV")
    else:
        raise ValueError("Format non pris en charge. Utiliser CSV, XLSX ou XLS.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def audit_dataframe(df: pd.DataFrame, id_col: str | None) -> tuple[pd.DataFrame, dict]:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "variable": col,
            "type_python": str(s.dtype),
            "n_manquants": int(s.isna().sum()),
            "taux_manquants": float(s.isna().mean()),
            "n_modalites": int(s.nunique(dropna=True)),
            "constante": bool(s.nunique(dropna=True) <= 1),
        })
    audit = pd.DataFrame(rows).sort_values(["taux_manquants", "variable"], ascending=[False, True])
    duplicate_ids = int(df[id_col].duplicated().sum()) if id_col and id_col in df else None
    summary = {
        "n_lignes": int(len(df)),
        "n_colonnes": int(df.shape[1]),
        "id": id_col,
        "doublons_identifiant": duplicate_ids,
        "doublons_lignes_completes": int(df.duplicated().sum()),
        "variables_constantes": audit.loc[audit["constante"], "variable"].tolist(),
        "variables_plus_20pct_manquants": audit.loc[audit["taux_manquants"] > .20, "variable"].tolist(),
    }
    return audit, summary


def first_available(columns, candidates):
    return next((c for c in candidates if c in columns), None)


def pseudonymize(values: pd.Series) -> pd.Series:
    salt = os.environ.get("DIEM_HASH_SALT", "memoire-dea-local-changez-ce-sel")
    def one(value):
        digest = hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()[:12].upper()
        return f"HH-{digest}"
    return values.astype(str).map(one)
