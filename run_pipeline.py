from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import platform
import sys
import os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PIPELINE_VERSION = "2.0.0-correctif-recodages-r5"

from src.data_io import load_table, audit_dataframe, first_available, pseudonymize, sha256_file
from src.features import load_config, build_features, oriented_matrix
from src.gower_distance import weighted_gower_numeric
from src.validation import (
    evaluate_candidate_grid,
    rank_sensitivity,
    partition_sensitivity,
)
from src.reporting import cluster_profiles, save_figures, dump_json


def main():
    parser = argparse.ArgumentParser(description="Classification et priorisation exploratoires des menages DIEM RDC R5")
    parser.add_argument("--input", required=True, help="Chemin vers le CSV officiel FAO ou le fichier Excel de travail")
    parser.add_argument("--sheet", default=None, help="Feuille Excel, si necessaire")
    parser.add_argument("--output", default=str(ROOT / "outputs"), help="Dossier de resultats")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument(
        "--repetitions", "--bootstrap", dest="repetitions", type=int, default=15,
        help="Nombre de sous-echantillonnages sans remise pour la stabilite (30 recommande pour l'analyse finale)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity", type=int, default=0, help="Nombre de menages a preslectionner pour verification; 0 = aucun seuil")
    parser.add_argument("--expected-rows", type=int, default=0, help="Effectif attendu; 0 desactive ce controle")
    parser.add_argument("--expected-columns", type=int, default=0, help="Nombre de colonnes attendu; 0 desactive ce controle")
    parser.add_argument("--quick", action="store_true", help="Test rapide : k=2..4 et 3 sous-echantillons")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "config" / "variables.json")
    df = load_table(args.input, args.sheet)
    if args.expected_rows and len(df) != args.expected_rows:
        raise ValueError(
            f"Le fichier contient {len(df)} lignes, alors que {args.expected_rows} etaient attendues. "
            "Ne poursuivez pas avant d'avoir identifie la version du fichier."
        )
    if args.expected_columns and df.shape[1] != args.expected_columns:
        raise ValueError(
            f"Le fichier contient {df.shape[1]} colonnes, alors que {args.expected_columns} etaient attendues. "
            "Ne remplacez pas ce controle sans documenter l'ecart dans le memoire."
        )
    id_col = first_available(df.columns, config["id_candidates"])
    weight_col = first_available(df.columns, config["weight_candidates"])

    audit, audit_summary = audit_dataframe(df, id_col)
    audit.to_csv(out / "audit_variables.csv", index=False, encoding="utf-8-sig")
    dump_json(audit_summary, out / "audit_resume.json")

    features, catalogue, warnings = build_features(df, config)
    if not os.environ.get("DIEM_HASH_SALT"):
        warnings.append(
            "DIEM_HASH_SALT n'est pas defini : le sel local par defaut a ete utilise. "
            "Definir un secret local avant l'execution finale et ne pas le diffuser."
        )
    oriented, feature_weights, dim_scores = oriented_matrix(
        features,
        catalogue,
        minimum_fraction=config.get("minimum_dimension_feature_fraction", .50),
    )
    catalogue.to_csv(out / "variables_actives_et_transformations.csv", index=False, encoding="utf-8-sig")
    dim_scores.to_csv(out / "scores_dimensions_intermediaires.csv", index=False, encoding="utf-8-sig")

    minimum_dimensions = config.get("minimum_dimensions_for_score", 3)
    observed_dimensions = dim_scores.notna().sum(axis=1)
    analyzable = observed_dimensions.ge(minimum_dimensions)
    n_excluded = int((~analyzable).sum())
    if n_excluded:
        warnings.append(
            f"{n_excluded} menage(s) exclus de la classification : moins de {minimum_dimensions} "
            "dimensions suffisamment observees."
        )
    if int(analyzable.sum()) < max(args.k_max + 1, 20):
        raise ValueError("Effectif analysable insuffisant pour comparer les partitions demandees.")

    oriented_analysis = oriented.loc[analyzable].copy()
    dim_scores_analysis = dim_scores.loc[analyzable].copy()
    dimension_weights = {column: 1.0 for column in dim_scores_analysis.columns}
    distance = weighted_gower_numeric(dim_scores_analysis, dimension_weights)
    if np.any(~np.isfinite(distance)) or np.any(distance < 0) or np.any(distance > 1):
        raise ValueError("La matrice de Gower contient des valeurs invalides.")
    np.save(out / "distance_gower.npy", distance)

    k_values = range(2, 5) if args.quick else range(args.k_min, args.k_max + 1)
    repeats = 3 if args.quick else args.repetitions
    k_table, chosen_k, chosen_method, labels = evaluate_candidate_grid(
        distance, k_values, args.seed, repeats
    )
    k_table.to_csv(out / "validation_nombre_profils.csv", index=False, encoding="utf-8-sig")
    algo_table = k_table.loc[k_table["k"].eq(chosen_k)].copy()
    algo_table.to_csv(out / "comparaison_algorithmes.csv", index=False, encoding="utf-8-sig")

    main_score_analysis = dim_scores_analysis.mean(axis=1, skipna=True)
    label_order = sorted(np.unique(labels), key=lambda value: int(np.flatnonzero(labels == value).min()))
    neutral_codes = {raw: rank + 1 for rank, raw in enumerate(label_order)}
    profile_analysis = pd.Series(labels, index=oriented_analysis.index).map(neutral_codes).astype(int)

    main_score = pd.Series(np.nan, index=df.index, dtype=float)
    main_score.loc[analyzable] = main_score_analysis
    profile = pd.Series(pd.NA, index=df.index, dtype="Int64")
    profile.loc[analyzable] = profile_analysis.astype("Int64")

    if id_col:
        pseudo = pseudonymize(df[id_col])
    else:
        pseudo = pseudonymize(pd.Series(df.index, index=df.index))
        warnings.append("Aucun identifiant officiel detecte; pseudonymes fondes sur l'index des lignes.")

    rank = pd.Series(pd.NA, index=df.index, dtype="Int64")
    ranking_frame = pd.DataFrame({
        "score": main_score_analysis,
        "pseudonyme": pseudo.loc[analyzable],
    }).sort_values(["score", "pseudonyme"], ascending=[False, True], kind="mergesort")
    rank.loc[ranking_frame.index] = pd.Series(
        np.arange(1, len(ranking_frame) + 1), index=ranking_frame.index, dtype="Int64"
    )
    selected = rank.le(args.capacity).fillna(False) if args.capacity > 0 else pd.Series(False, index=df.index)
    priority_status = pd.Series("Non prioritaire dans ce scénario", index=df.index, dtype="object")
    if args.capacity <= 0:
        priority_status[:] = "Non défini - capacité non fixée"
    else:
        priority_status.loc[selected.fillna(False)] = "Prioritaire pour vérification"
    priority_status.loc[~analyzable] = "Non classé - données insuffisantes"
    assignments = pd.DataFrame({
        "menage_pseudonyme": pseudo,
        "profil": profile,
        "score_priorisation": main_score,
        "rang_priorisation": rank,
        "prioritaire_verification": selected.fillna(False),
        "statut_priorite": priority_status,
    })
    for column in dim_scores.columns:
        assignments[f"score_dimension__{column}"] = dim_scores[column]
    for col in ["adm1_name"]:
        if col in df:
            assignments[col] = df[col].values
    assignments.to_csv(out / "classement_pseudonymise_usage_local.csv", index=False, encoding="utf-8-sig")

    weights = df.loc[analyzable, weight_col] if weight_col else None
    profiles = cluster_profiles(
        oriented_analysis, dim_scores_analysis, profile_analysis, weights
    )
    profiles.to_csv(out / "profils_agreges.csv", index=False, encoding="utf-8-sig")
    sensitivity = rank_sensitivity(dim_scores_analysis, main_score_analysis)
    sensitivity.to_csv(out / "sensibilite_priorisation.csv", index=False, encoding="utf-8-sig")
    asym = set(catalogue.loc[catalogue["kind"].eq("asymmetric_binary"), "name"])
    partition_sensitivity_table = partition_sensitivity(
        labels,
        dim_scores_analysis,
        oriented_analysis,
        feature_weights,
        asym,
        chosen_k,
        args.seed,
    )
    partition_sensitivity_table.to_csv(
        out / "sensibilite_partitions.csv", index=False, encoding="utf-8-sig"
    )
    save_figures(k_table, profiles, out)

    summary = {
        **audit_summary,
        "version_pipeline": PIPELINE_VERSION,
        "fichier_source": str(Path(args.input).name),
        "colonne_pondération": weight_col,
        "variables_actives": catalogue.loc[catalogue["statut"].eq("active"), "name"].tolist(),
        "dimensions": sorted(catalogue.loc[catalogue["statut"].eq("active"), "dimension"].unique().tolist()),
        "representation_principale": (
            f"distance de Gower sur {len(dim_scores.columns)} scores dimensionnels à poids égaux"
        ),
        "k_retenu": chosen_k,
        "algorithme_retenu": chosen_method,
        "capacity_scenario": args.capacity,
        "n_menages_analysables": int(analyzable.sum()),
        "n_menages_exclus_donnees_insuffisantes": n_excluded,
        "repetitions_sous_echantillonnage": repeats,
        "avertissements": warnings,
        "regle_interpretation": "Les profils et le rang sont analytiques. Ils ne constituent pas une attribution d'assistance.",
    }
    dump_json(summary, out / "resume_resultats.json")
    packages = {}
    for package in ["pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "seaborn"]:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "non détecté"
    journal = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "version_pipeline": PIPELINE_VERSION,
        "fichier_source": Path(args.input).name,
        "empreinte_sha256_source": sha256_file(args.input),
        "empreinte_sha256_configuration": sha256_file(ROOT / "config" / "variables.json"),
        "python": platform.python_version(),
        "plateforme": platform.platform(),
        "bibliotheques": packages,
        "parametres": {
            "k_min": 2 if args.quick else args.k_min,
            "k_max": 4 if args.quick else args.k_max,
            "repetitions_sous_echantillonnage": repeats,
            "graine": args.seed,
            "capacite_verification": args.capacity,
            "effectif_attendu": args.expected_rows,
            "colonnes_attendues": args.expected_columns,
            "mode_rapide": bool(args.quick),
            "dimensions_effectives": dim_scores.columns.tolist(),
        },
    }
    dump_json(journal, out / "journal_execution.json")
    (out / "AVERTISSEMENTS.txt").write_text("\n".join(warnings) if warnings else "Aucun avertissement technique.", encoding="utf-8")
    print(f"Analyse terminee. Resultats : {out}")
    print(f"Version du pipeline : {PIPELINE_VERSION}")
    print(f"Dimensions effectives : {', '.join(dim_scores.columns)}")
    print(f"k retenu : {chosen_k}")
    print(f"algorithme retenu : {summary['algorithme_retenu']}")
    if not weight_col:
        print("ATTENTION : weight_final absent. Les profils agreges sont non ponderes.")


if __name__ == "__main__":
    main()
