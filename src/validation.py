from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.stats import spearmanr

from .clustering import kmedoids_alternate, hierarchical_average
from .gower_distance import weighted_gower_numeric


def _add_selection_ranks(table: pd.DataFrame, stability_column: str) -> pd.DataFrame:
    """Ajoute la règle de sélection 40 % / 20 % / 40 % aux candidats admissibles."""
    result = table.copy()
    eligible = result[result["part_plus_petit_groupe"] >= .05].copy()
    if eligible.empty:
        eligible = result.copy()
    for metric in ["silhouette", "dunn", stability_column]:
        eligible[f"rang_{metric}"] = eligible[metric].rank(ascending=False, method="min")
    eligible["score_selection"] = (
        .40 * eligible["rang_silhouette"]
        + .20 * eligible["rang_dunn"]
        + .40 * eligible[f"rang_{stability_column}"]
    )
    keys = [c for c in ["algorithme", "k"] if c in result.columns]
    rank_columns = [
        "rang_silhouette", "rang_dunn", f"rang_{stability_column}", "score_selection",
    ]
    return result.merge(eligible[keys + rank_columns], on=keys, how="left")


def dunn_index(distance: np.ndarray, labels: np.ndarray) -> float:
    groups = np.unique(labels)
    max_intra = 0.0
    min_inter = np.inf
    for g in groups:
        idx = np.flatnonzero(labels == g)
        if len(idx) > 1:
            max_intra = max(max_intra, float(distance[np.ix_(idx, idx)].max()))
    for i, g1 in enumerate(groups):
        idx1 = np.flatnonzero(labels == g1)
        for g2 in groups[i + 1:]:
            idx2 = np.flatnonzero(labels == g2)
            min_inter = min(min_inter, float(distance[np.ix_(idx1, idx2)].min()))
    return float(min_inter / max_intra) if max_intra > 0 and np.isfinite(min_inter) else np.nan


def stability_subsamples(distance, full_labels, method, k, seed=42, repeats=15, fraction=.80):
    rng = np.random.default_rng(seed)
    n = len(full_labels)
    values = []
    for b in range(repeats):
        sample_size = min(n, max(k * 5, int(n * fraction)))
        idx = np.sort(rng.choice(n, size=sample_size, replace=False))
        sub = distance[np.ix_(idx, idx)]
        if method == "kmedoids":
            labels = kmedoids_alternate(sub, k, seed=seed + b, n_init=3)["labels"]
        else:
            labels = hierarchical_average(sub, k)["labels"]
        values.append(adjusted_rand_score(full_labels[idx], labels))
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def evaluate_k_range(distance, k_values, seed=42, repeats=15):
    records = []
    fitted = {}
    for k in k_values:
        result = kmedoids_alternate(distance, k, seed=seed, n_init=8)
        labels = result["labels"]
        stability_mean, stability_sd = stability_subsamples(
            distance, labels, "kmedoids", k, seed=seed, repeats=repeats
        )
        counts = np.bincount(labels, minlength=k)
        records.append({
            "k": k,
            "silhouette": float(silhouette_score(distance, labels, metric="precomputed")),
            "dunn": dunn_index(distance, labels),
            "stabilite_ari_moyenne": stability_mean,
            "stabilite_ari_ecart_type": stability_sd,
            "part_plus_petit_groupe": float(counts.min() / counts.sum()),
            "cout_kmedoids": result["cost"],
        })
        fitted[k] = result
    table = pd.DataFrame(records)
    eligible = table[table["part_plus_petit_groupe"] >= .05].copy()
    if eligible.empty:
        eligible = table.copy()
    for metric in ["silhouette", "dunn", "stabilite_ari_moyenne"]:
        eligible[f"rang_{metric}"] = eligible[metric].rank(ascending=False, method="min")
    eligible["score_selection"] = (
        .40 * eligible["rang_silhouette"]
        + .20 * eligible["rang_dunn"]
        + .40 * eligible["rang_stabilite_ari_moyenne"]
    )
    chosen_k = int(eligible.sort_values(["score_selection", "k"]).iloc[0]["k"])
    table = table.merge(
        eligible[[
            "k", "rang_silhouette", "rang_dunn", "rang_stabilite_ari_moyenne",
            "score_selection",
        ]],
        on="k", how="left",
    )
    table["k_retenu"] = table["k"].eq(chosen_k)
    return table, chosen_k, fitted[chosen_k]


def evaluate_candidate_grid(distance, k_values, seed=42, repeats=15):
    """Évalue PAM et la hiérarchie moyenne pour chaque valeur de k.

    Le nombre de groupes et l'algorithme sont ainsi choisis conjointement, sans
    favoriser une méthode lors de la détermination préalable de k.
    """
    records = []
    fitted = {}
    for k in k_values:
        km = kmedoids_alternate(distance, k, seed=seed, n_init=8)
        candidates = [
            ("k-medoids", km["labels"], km["cost"]),
            ("hierarchique_moyenne", hierarchical_average(distance, k)["labels"], np.nan),
        ]
        for method, labels, cost in candidates:
            stability_mean, stability_sd = stability_subsamples(
                distance,
                labels,
                "kmedoids" if method == "k-medoids" else "hierarchical",
                k,
                seed=seed,
                repeats=repeats,
            )
            counts = np.bincount(labels, minlength=k)
            records.append({
                "algorithme": method,
                "k": k,
                "silhouette": float(silhouette_score(distance, labels, metric="precomputed")),
                "dunn": dunn_index(distance, labels),
                "stabilite_ari_moyenne": stability_mean,
                "stabilite_ari_ecart_type": stability_sd,
                "part_plus_petit_groupe": float(counts.min() / counts.sum()),
                "cout_kmedoids": cost,
            })
            fitted[(method, k)] = labels

    table = pd.DataFrame(records)
    table["admissible"] = table["part_plus_petit_groupe"].ge(.05)
    table = _add_selection_ranks(table, "stabilite_ari_moyenne")
    eligible = table[table["admissible"] & table["score_selection"].notna()].copy()
    if eligible.empty:
        eligible = table[table["score_selection"].notna()].copy()
    winner = eligible.sort_values(
        ["score_selection", "k", "silhouette", "stabilite_ari_moyenne", "algorithme"],
        ascending=[True, True, False, False, True],
    ).iloc[0]
    chosen_method = str(winner["algorithme"])
    chosen_k = int(winner["k"])
    table["retenu"] = table["algorithme"].eq(chosen_method) & table["k"].eq(chosen_k)
    return table, chosen_k, chosen_method, fitted[(chosen_method, chosen_k)]


def compare_algorithms(distance, k, kmedoids_result, seed=42, repeats=15):
    km_labels = kmedoids_result["labels"]
    h = hierarchical_average(distance, k)
    h_labels = h["labels"]
    km_stab, km_sd = stability_subsamples(distance, km_labels, "kmedoids", k, seed, repeats)
    h_stab, h_sd = stability_subsamples(distance, h_labels, "hierarchical", k, seed, repeats)
    rows = []
    for name, labels, stability, stability_sd in [
        ("k-medoids", km_labels, km_stab, km_sd),
        ("hierarchique_moyenne", h_labels, h_stab, h_sd),
    ]:
        counts = np.bincount(labels, minlength=k)
        rows.append({
            "algorithme": name,
            "silhouette": silhouette_score(distance, labels, metric="precomputed"),
            "dunn": dunn_index(distance, labels),
            "stabilite_ari": stability,
            "stabilite_ecart_type": stability_sd,
            "part_plus_petit_groupe": float(counts.min() / counts.sum()),
        })
    table = pd.DataFrame(rows)
    table["admissible"] = table["part_plus_petit_groupe"].ge(.05)
    eligible = table[table["admissible"]].copy()
    if eligible.empty:
        eligible = table.copy()
    for metric in ["silhouette", "dunn", "stabilite_ari"]:
        eligible[f"rang_{metric}"] = eligible[metric].rank(ascending=False, method="min")
    eligible["score_selection"] = (
        .40 * eligible["rang_silhouette"]
        + .20 * eligible["rang_dunn"]
        + .40 * eligible["rang_stabilite_ari"]
    )
    winner = eligible.sort_values(["score_selection", "algorithme"]).iloc[0]["algorithme"]
    table = table.merge(
        eligible[[
            "algorithme", "rang_silhouette", "rang_dunn", "rang_stabilite_ari",
            "score_selection",
        ]],
        on="algorithme", how="left"
    )
    table["retenu"] = table["algorithme"].eq(winner)
    labels = km_labels if winner == "k-medoids" else h_labels
    return table, labels, h


def rank_sensitivity(dim_scores: pd.DataFrame, main_score: pd.Series) -> pd.DataFrame:
    def weighted_score(weights):
        columns = [c for c in weights if c in dim_scores]
        if not columns:
            return pd.Series(np.nan, index=dim_scores.index, dtype=float)
        w = pd.Series({c: float(weights[c]) for c in columns})
        observed_weight = dim_scores[columns].notna().mul(w, axis=1).sum(axis=1)
        numerator = dim_scores[columns].mul(w, axis=1).sum(axis=1, skipna=True)
        return numerator.div(observed_weight.replace(0, np.nan))

    columns = list(dim_scores.columns)
    scenarios = {
        "dimensions_poids_egaux": weighted_score({c: 1.0 for c in columns}),
        "acces_consommation_renforce": weighted_score({c: (3.0 if c == "acces_consommation" else 1.0) for c in columns}),
        "strategies_adaptation_renforcees": weighted_score({c: (3.0 if c == "strategies_adaptation" else 1.0) for c in columns}),
        "sans_capacites_structurelles": weighted_score({c: 1.0 for c in columns if c != "capacites_structurelles"}),
        "acces_et_adaptation": weighted_score({c: 1.0 for c in columns if c in {"acces_consommation", "strategies_adaptation"}}),
    }

    def top_set(score, fraction):
        n_top = max(1, int(np.ceil(score.notna().sum() * fraction)))
        return set(score.dropna().sort_values(ascending=False, kind="mergesort").head(n_top).index)

    rows = []
    for name, score in scenarios.items():
        valid = main_score.notna() & score.notna()
        rho, p = spearmanr(main_score[valid], score[valid])
        row = {"scenario": name, "rho_spearman": float(rho), "p_value": float(p), "n": int(valid.sum())}
        for pct in (10, 20, 30):
            main_top = top_set(main_score[valid], pct / 100)
            scenario_top = top_set(score[valid], pct / 100)
            row[f"recouvrement_top_{pct}pct"] = (
                len(main_top & scenario_top) / len(main_top) if main_top else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def partition_sensitivity(
    main_labels: np.ndarray,
    dim_scores: pd.DataFrame,
    oriented_variables: pd.DataFrame,
    variable_weights: dict[str, float],
    asymmetric_binary: set[str],
    k: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Mesure l'effet de deux représentations alternatives par l'ARI."""
    rows = []
    variable_distance = weighted_gower_numeric(
        oriented_variables,
        variable_weights,
        asymmetric_binary,
    )
    variable_labels = kmedoids_alternate(
        variable_distance, k, seed=seed, n_init=8
    )["labels"]
    rows.append({
        "scenario": "gower_variables_actives",
        "comparaison": "PAM sur variables actives versus PAM sur quatre dimensions",
        "ari_avec_partition_principale": float(adjusted_rand_score(main_labels, variable_labels)),
    })

    complete = dim_scores.copy()
    for col in complete:
        complete[col] = complete[col].fillna(complete[col].median())
    km_labels = KMeans(n_clusters=k, random_state=seed, n_init=50).fit_predict(complete)
    rows.append({
        "scenario": "kmeans_dimensions",
        "comparaison": "k-means euclidien versus méthode principale sur quatre dimensions",
        "ari_avec_partition_principale": float(adjusted_rand_score(main_labels, km_labels)),
    })
    return pd.DataFrame(rows)
