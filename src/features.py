from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_label(value) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _numeric(series: pd.Series, log=False) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if log:
        out = np.log1p(out.clip(lower=0))
    return out.astype(float)


def _binary_indicator(series: pd.Series) -> pd.Series:
    """Convertit sans ambiguite les indicateurs binaires numeriques ou Yes/No."""
    result = pd.Series(np.nan, index=series.index, dtype=float)
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna()
    result.loc[numeric_mask] = numeric.loc[numeric_mask].gt(0).astype(float)

    normalized = series.map(normalize_label)
    yes_values = {"yes", "oui", "true", "vrai", "y"}
    no_values = {"no", "non", "false", "faux", "n"}
    result.loc[normalized.isin(yes_values)] = 1.0
    result.loc[normalized.isin(no_values)] = 0.0
    return result


def _binary_any(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    present = [c for c in columns if c in df]
    if not present:
        return pd.Series(np.nan, index=df.index, dtype=float)
    block = pd.DataFrame({column: _binary_indicator(df[column]) for column in present})
    observed = block.notna().any(axis=1)
    result = block.fillna(0).gt(0).any(axis=1).astype(float)
    return result.where(observed)


def _apply_expected_range(series: pd.Series, spec: dict, warnings: list[str]) -> pd.Series:
    """Met hors analyse les valeurs impossibles sans modifier le fichier source."""
    lower = spec.get("expected_min")
    upper = spec.get("expected_max")
    invalid = pd.Series(False, index=series.index)
    if lower is not None:
        invalid |= series.lt(lower)
    if upper is not None:
        invalid |= series.gt(upper)
    count = int(invalid.fillna(False).sum())
    if count:
        bounds = f"[{lower if lower is not None else '-inf'}, {upper if upper is not None else '+inf'}]"
        warnings.append(
            f"{spec['name']} : {count} valeur(s) hors de la plage documentee {bounds}; "
            "elles sont traitees comme manquantes pour l'analyse."
        )
        series = series.mask(invalid)
    return series


def build_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Construit les variables actives et documente chaque transformation."""
    out = pd.DataFrame(index=df.index)
    catalogue = []
    warnings = []

    for spec in config["direct_features"]:
        source = next((c for c in spec["sources"] if c in df), None)
        if source is None:
            warnings.append(f"Variable absente : {spec['name']} ({', '.join(spec['sources'])})")
            continue
        raw = _numeric(df[source], log=False)
        raw = _apply_expected_range(raw, spec, warnings)
        out[spec["name"]] = np.log1p(raw.clip(lower=0)) if spec["kind"] == "numeric_log" else raw
        catalogue.append({**spec, "source_retenue": source, "statut": "active"})

    for spec in config["ordinal_features"]:
        source = spec["source"]
        if source not in df:
            warnings.append(f"Variable ordinale absente : {source}")
            catalogue.append({
                **spec,
                "source_retenue": "",
                "kind": "ordinal",
                "statut": "exclue_absente",
                "couverture_recodage": 0.0,
            })
            continue
        normalized = df[source].map(normalize_label)
        mapping = {normalize_label(k): v for k, v in spec["mapping"].items()}
        mapped = normalized.map(mapping).astype(float)
        nonmissing = int(df[source].notna().sum())
        coverage = float(mapped.notna().sum() / nonmissing) if nonmissing else 0.0
        if coverage < config.get("minimum_mapping_coverage", .8):
            warnings.append(
                f"{source} exclue des variables actives : couverture du recodage {coverage:.1%}. "
                "Completer le dictionnaire de modalites avant l'analyse finale."
            )
            catalogue.append({
                **spec,
                "source_retenue": source,
                "kind": "ordinal",
                "statut": "exclue_recodage",
                "couverture_recodage": coverage,
            })
            continue
        out[spec["name"]] = mapped
        catalogue.append({**spec, "source_retenue": source, "kind": "ordinal", "statut": "active", "couverture_recodage": coverage})

    for name, columns in config["shock_groups"].items():
        series = _binary_any(df, columns)
        if series.notna().sum() == 0:
            warnings.append(f"Aucune variable disponible pour {name}")
            continue
        used = [c for c in columns if c in df]
        out[name] = series
        catalogue.append({
            "name": name,
            "sources": columns,
            "source_retenue": ", ".join(used),
            "dimension": "exposition_chocs",
            "kind": "asymmetric_binary",
            "direction": "higher_worse",
            "label": f"Presence d'au moins un choc dans le groupe {name}",
            "statut": "active",
        })

    cat = pd.DataFrame(catalogue)
    max_missing = config.get("maximum_active_missing_rate", .40)
    for col in list(out.columns):
        rate = float(out[col].isna().mean())
        if rate > max_missing:
            warnings.append(f"{col} exclue : taux de valeurs manquantes {rate:.1%} superieur a {max_missing:.0%}.")
            out = out.drop(columns=col)
            cat.loc[cat["name"].eq(col), "statut"] = "exclue_manquants"
            cat.loc[cat["name"].eq(col), "taux_manquants"] = rate
    for col in list(out.columns):
        if out[col].nunique(dropna=True) <= 1:
            warnings.append(f"{col} exclue : variable constante dans l'echantillon analyse.")
            out = out.drop(columns=col)
            cat.loc[cat["name"].eq(col), "statut"] = "exclue_constante"
    active_cat = cat[cat["statut"].eq("active")].copy()
    dimensions = set(active_cat["dimension"]) if not active_cat.empty else set()
    required_dimensions = set(config.get("required_dimensions", []))
    missing_dimensions = sorted(required_dimensions - dimensions)
    if missing_dimensions:
        raise ValueError(
            "Dimensions obligatoires absentes apres preparation : "
            + ", ".join(missing_dimensions)
            + ". Corriger les variables ou les recodages avant l'analyse finale."
        )
    if len(dimensions) < config.get("minimum_dimension_count", 3):
        raise ValueError(f"Seulement {len(dimensions)} dimensions exploitables. Verifier la base et les recodages.")
    return out, cat, warnings


def robust_scale(series: pd.Series) -> pd.Series:
    """Normalisation 0-1 avec bornes aux percentiles 1 et 99."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return s
    lo, hi = s.quantile([.01, .99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = s.min(), s.max()
    if hi <= lo:
        return pd.Series(0.0, index=s.index)
    return ((s.clip(lo, hi) - lo) / (hi - lo)).astype(float)


def oriented_matrix(features: pd.DataFrame, catalogue: pd.DataFrame, minimum_fraction: float = .50) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Oriente toutes les variables vers 1 = vulnerabilite plus forte."""
    oriented = pd.DataFrame(index=features.index)
    feature_weights = {}
    active_catalogue = catalogue[catalogue["statut"].eq("active") & catalogue["name"].isin(features.columns)].copy()
    dimension_counts = active_catalogue.groupby("dimension")["name"].count().to_dict()
    for row in active_catalogue.itertuples(index=False):
        scaled = robust_scale(features[row.name])
        if row.direction == "lower_worse":
            scaled = 1.0 - scaled
        oriented[row.name] = scaled
        feature_weights[row.name] = 1.0 / dimension_counts[row.dimension]
    dim_scores = pd.DataFrame(index=features.index)
    for dim, block in active_catalogue.groupby("dimension"):
        cols = block["name"].tolist()
        observed = oriented[cols].notna().sum(axis=1)
        required = max(1, int(np.ceil(len(cols) * minimum_fraction)))
        dim_scores[dim] = oriented[cols].mean(axis=1, skipna=True).where(observed >= required)
    return oriented, feature_weights, dim_scores
