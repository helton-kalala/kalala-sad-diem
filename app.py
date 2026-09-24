from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import html
import json
import math
import os
import subprocess
import sys

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parent
OFFICIAL_OUTPUT = ROOT / "outputs"
RUNS_ROOT = ROOT / "runs"
IMPORTS_ROOT = ROOT / "data" / "imports"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
IMPORTS_ROOT.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="SAD - Classification et priorisation", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background-color:#eeeeee; color:#202020;}
    div[data-testid="stMetric"] {background:#f7f7f7; border:1px solid #b8b8b8; padding:10px;}
    div[data-testid="stDataFrame"] {border:1px solid #a8a8a8;}
    .stButton > button, .stDownloadButton > button {
        background:#dedede; color:#111111; border:1px solid #8d8d8d; border-radius:2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Classification et priorisation des ménages")
st.caption("Interface V2.7 — restitution, analyses versionnées et guide du système")
st.caption(
    "Consultation locale des profils et des rangs. Le statut de priorité concerne "
    "uniquement un scénario de vérification et ne vaut pas décision d’assistance."
)


def render_system_guide() -> None:
    st.title("Guide du système")
    st.caption(
        "Vue synthétique du fonctionnement du SAD DIEM. "
        "Cliquez sur une étape pour afficher son rôle, ses entrées, ses traitements, "
        "ses sorties et le fichier Python associé."
    )

    steps = {
        "source": {
            "label": "1. Données DIEM",
            "role": "Point d’entrée des données utilisées par l’analyse.",
            "input": "CSV DIEM ou fichier compatible",
            "treatment": "Lecture contrôlée et vérification de la structure du fichier.",
            "output": "Données contrôlées + informations d’audit",
            "subs": [
                "Lecture sans modification du fichier source",
                "Contrôle des lignes, colonnes et variables requises",
                "Repérage des identifiants et anomalies structurelles",
            ],
            "files": [ROOT / "src" / "data_io.py"],
        },
        "features": {
            "label": "2. Préparation",
            "role": "Transformation des données en représentation analytique comparable.",
            "input": "Données contrôlées",
            "treatment": "Recodage, orientation, normalisation robuste et agrégation.",
            "output": "14 variables actives + 4 scores dimensionnels",
            "subs": [
                "Recodage des variables retenues",
                "Orientation : valeur élevée = vulnérabilité plus forte",
                "Normalisation robuste",
                "Agrégation en quatre dimensions",
            ],
            "files": [ROOT / "src" / "features.py"],
        },
        "gower": {
            "label": "3. Distance de Gower",
            "role": "Mesure de la dissimilarité entre les ménages.",
            "input": "4 scores dimensionnels",
            "treatment": "Comparaison paire à paire et combinaison des différences.",
            "output": "Matrice de dissimilarité",
            "subs": [
                "Comparaison dimension par dimension",
                "Pondération égale des dimensions",
                "Calcul de la distance finale pour chaque paire",
            ],
            "files": [ROOT / "src" / "gower_distance.py"],
        },
        "cluster": {
            "label": "4. Classification",
            "role": "Construction de partitions candidates sans variable cible.",
            "input": "Matrice de Gower",
            "treatment": "PAM/k-medoids et classification hiérarchique pour k = 2 à 6.",
            "output": "Partitions candidates",
            "subs": [
                "Calcul des partitions PAM/k-medoids",
                "Calcul des partitions hiérarchiques",
                "Répétition pour plusieurs valeurs de k",
            ],
            "files": [ROOT / "src" / "clustering.py"],
        },
        "validation": {
            "label": "5. Validation",
            "role": "Comparaison des partitions avant sélection finale.",
            "input": "Partitions candidates",
            "treatment": "Silhouette, Dunn, stabilité ARI et contrôle de la taille des groupes.",
            "output": "Partition analytique retenue",
            "subs": [
                "Mesure de la cohésion et de la séparation",
                "Mesure de la stabilité par rééchantillonnage",
                "Rejet des partitions avec groupe trop petit",
                "Sélection comparative de la solution finale",
            ],
            "files": [ROOT / "src" / "validation.py"],
        },
        "profiles": {
            "label": "6. Profils",
            "role": "Interprétation des groupes issus de la partition retenue.",
            "input": "Partition finale + scores dimensionnels",
            "treatment": "Résumé des caractéristiques et comparaison des dimensions.",
            "output": "Profils agrégés interprétables",
            "subs": [
                "Calcul des résumés par profil",
                "Comparaison des quatre dimensions",
                "Préparation des tableaux et graphiques",
            ],
            "files": [ROOT / "src" / "reporting.py"],
        },
        "ranking": {
            "label": "7. Score et rang",
            "role": "Priorisation continue, distincte de la classification.",
            "input": "4 scores dimensionnels orientés",
            "treatment": "Moyenne à poids égaux puis classement décroissant.",
            "output": "Score de priorisation + rang + statut du scénario",
            "subs": [
                "Calcul du score continu",
                "Classement de tous les ménages",
                "Application éventuelle d’un quota",
                "Présélection pour vérification humaine",
            ],
            "files": [ROOT / "run_pipeline.py"],
        },
        "store": {
            "label": "8. Historisation",
            "role": "Conservation séparée des résultats officiels et des nouvelles analyses.",
            "input": "Sorties du pipeline",
            "treatment": "Écriture dans outputs/ ou dans un dossier versionné de runs/.",
            "output": "Historique analytique consultable",
            "subs": [
                "Conservation de l’analyse officielle",
                "Création d’un dossier par nouvelle exécution",
                "Conservation des métadonnées de l’exécution",
            ],
            "files": [ROOT / "run_pipeline.py", ROOT / "app.py"],
        },
        "interface": {
            "label": "9. Interface",
            "role": "Consultation et exploitation locale des résultats.",
            "input": "Exécution analytique sélectionnée",
            "treatment": "Lecture, filtrage, visualisation, suivi et export.",
            "output": "Tableau de bord décisionnel",
            "subs": [
                "Liste classée",
                "Profils et visualisations",
                "Synthèse provinciale",
                "Fiche ménage et suivi",
                "Traçabilité et nouvelle analyse",
            ],
            "files": [ROOT / "app.py"],
        },
    }

    if "guide_step" not in st.session_state:
        st.session_state.guide_step = "source"

    def step_button(column, key):
        with column:
            if st.button(steps[key]["label"], key=f"guide_{key}", use_container_width=True):
                st.session_state.guide_step = key

    # Chaîne visuelle en trois lignes avec flèches.
    c1, a1, c2, a2, c3 = st.columns([1, 0.12, 1, 0.12, 1])
    step_button(c1, "source")
    with a1:
        st.markdown("### →")
    step_button(c2, "features")
    with a2:
        st.markdown("### →")
    step_button(c3, "gower")

    st.markdown("<div style='text-align:right;padding-right:16%;font-size:28px'>↓</div>", unsafe_allow_html=True)

    c6, a5, c5, a4, c4 = st.columns([1, 0.12, 1, 0.12, 1])
    step_button(c6, "profiles")
    with a5:
        st.markdown("### ←")
    step_button(c5, "validation")
    with a4:
        st.markdown("### ←")
    step_button(c4, "cluster")

    st.markdown("<div style='text-align:left;padding-left:16%;font-size:28px'>↓</div>", unsafe_allow_html=True)

    c7, a7, c8, a8, c9 = st.columns([1, 0.12, 1, 0.12, 1])
    step_button(c7, "ranking")
    with a7:
        st.markdown("### →")
    step_button(c8, "store")
    with a8:
        st.markdown("### →")
    step_button(c9, "interface")

    current = steps[st.session_state.guide_step]

    st.divider()
    st.subheader(current["label"])
    st.write(current["role"])

    e1, e2, e3 = st.columns(3)
    e1.markdown(f"**Entrée**  \n{current['input']}")
    e2.markdown(f"**Traitement**  \n{current['treatment']}")
    e3.markdown(f"**Sortie**  \n{current['output']}")

    st.markdown("**Sous-activités**")
    for item in current["subs"]:
        st.markdown(f"- {item}")

    existing_sources = [path for path in current["files"] if path.exists()]
    if existing_sources:
        with st.expander("Voir le code associé"):
            for source_path in existing_sources:
                st.markdown(f"**{source_path.relative_to(ROOT)}**")
                try:
                    source_text = source_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    source_text = source_path.read_text(encoding="latin-1")
                st.code(source_text, language="python", line_numbers=True)

    st.divider()
    st.subheader("Comprendre les algorithmes")
    st.caption(
        "Présentation courte des méthodes effectivement utilisées dans le mémoire. "
        "Les panneaux ci-dessous s’ouvrent sans relancer l’analyse."
    )

    with st.expander("Distance de Gower"):
        st.markdown(
            "**Rôle :** mesurer la différence entre deux ménages à partir des dimensions de vulnérabilité."
        )
        st.markdown(
            "**Principe :** chaque dimension produit une différence partielle ; "
            "ces différences sont combinées pour obtenir une distance globale comprise entre 0 et 1."
        )
        st.markdown(
            "**Chaîne :** ménage A + ménage B → différences par dimension → pondération → distance finale."
        )

    with st.expander("PAM / k-medoids"):
        st.markdown(
            "**Rôle :** former k groupes autour de ménages réels représentatifs appelés médoïdes."
        )
        st.markdown(
            "**Chaîne :** choisir k → initialiser les médoïdes → affecter les ménages → "
            "tester des échanges → conserver la partition qui réduit la dissimilarité interne."
        )
        st.markdown("Dans le protocole, **k est exploré de 2 à 6**.")

    with st.expander("Classification hiérarchique"):
        st.markdown(
            "**Rôle :** fournir une méthode comparative de regroupement."
        )
        st.markdown(
            "**Chaîne :** ménages séparés → calcul des proximités → fusions successives → "
            "dendrogramme logique → coupe selon k."
        )
        st.markdown("La liaison moyenne est utilisée avec la même matrice de Gower.")

    with st.expander("Validation et choix final"):
        st.markdown(
            "**Critères :** silhouette, indice de Dunn, stabilité ARI et taille minimale des groupes."
        )
        st.markdown(
            "**Chaîne :** partitions candidates → contrôle des groupes trop petits → "
            "mesure de qualité → mesure de stabilité → comparaison → solution finale."
        )
        st.markdown(
            "Le score synthétique du protocole attribue **40 % à la silhouette, 20 % à Dunn "
            "et 40 % à la stabilité**."
        )

    st.info(
        "Cette page est un guide de fonctionnement. Les diagrammes UML formels restent dans le mémoire "
        "et dans la documentation technique, pas dans l’interface utilisateur."
    )


def discover_executions() -> dict[str, Path]:
    """Return available analytical result folders without mixing runs."""
    executions: dict[str, Path] = {}
    if (OFFICIAL_OUTPUT / "classement_pseudonymise_usage_local.csv").exists():
        executions["Analyse officielle - DIEM RDC R5 2023"] = OFFICIAL_OUTPUT
    for folder in sorted(RUNS_ROOT.glob("*"), reverse=True):
        if folder.is_dir() and (folder / "classement_pseudonymise_usage_local.csv").exists():
            label = folder.name.replace("_", " ")
            executions[f"Nouvelle exécution - {label}"] = folder
    return executions


available_executions = discover_executions()
if not available_executions:
    st.error(
        "Aucune sortie analytique n'est disponible. Exécutez d'abord le pipeline officiel "
        "ou placez une exécution valide dans le dossier runs."
    )
    st.stop()

selected_execution_label = st.sidebar.selectbox(
    "Exécution analytique consultée",
    list(available_executions.keys()),
    help="Chaque exécution conserve ses propres résultats. Changer d'exécution ne recalcule pas le modèle.",
)
OUTPUT = available_executions[selected_execution_label]
st.sidebar.caption(f"Référentiel actif : {OUTPUT.name}")

st.sidebar.divider()
navigation = st.sidebar.radio(
    "Navigation",
    ["Tableau de bord", "Guide du système"],
    index=0,
)

if navigation == "Guide du système":
    render_system_guide()
    st.stop()

assign_path = OUTPUT / "classement_pseudonymise_usage_local.csv"
profile_path = OUTPUT / "profils_agreges.csv"
summary_path = OUTPUT / "resume_resultats.json"

assign = pd.read_csv(assign_path)
profiles = pd.read_csv(profile_path)
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
assign["rang_priorisation"] = pd.to_numeric(assign["rang_priorisation"], errors="coerce").astype("Int64")
assign["score_priorisation"] = pd.to_numeric(assign["score_priorisation"], errors="coerce")


def scenario_capacity(label: str, custom_value: int) -> int:
    n = int(assign["rang_priorisation"].notna().sum())
    if label == "Aucune capacité fixée":
        return 0
    if label == "Nombre personnalisé":
        return min(max(int(custom_value), 0), n)
    percentage = int(label.replace(" %", ""))
    return int(math.ceil(n * percentage / 100))


def apply_capacity(frame: pd.DataFrame, capacity: int) -> pd.DataFrame:
    out = frame.copy()
    classable = out["rang_priorisation"].notna()
    if capacity <= 0:
        out["statut_scenario"] = "Capacité non fixée"
    else:
        out["statut_scenario"] = "Non prioritaire dans ce scénario"
        out.loc[classable & out["rang_priorisation"].le(capacity), "statut_scenario"] = (
            "Prioritaire pour vérification"
        )
    out.loc[~classable, "statut_scenario"] = "Non classé - données insuffisantes"
    return out


REPORT_LABELS = {
    "fies_probability": "Probabilité associée à la FIES",
    "food_consumption": "Consommation alimentaire",
    "dietary_diversity": "Diversité alimentaire",
    "household_hunger": "Faim du ménage",
    "reduced_coping": "Stratégies d’adaptation réduites",
    "livelihood_coping": "Stratégies liées aux moyens d’existence",
    "income_capacity": "Capacité de revenu",
    "education_capacity": "Capacité d’éducation",
    "water_deprivation": "Privation d’eau",
    "sanitation_deprivation": "Privation d’assainissement",
    "shock_conflit": "Exposition aux conflits",
    "shock_economique": "Exposition aux chocs économiques",
    "shock_climatique": "Exposition aux chocs climatiques",
    "shock_menage_production": "Choc sur la production du ménage",
    "dimension__acces_consommation": "Accès et consommation",
    "dimension__capacites_structurelles": "Capacités structurelles",
    "dimension__exposition_chocs": "Exposition aux chocs",
    "dimension__strategies_adaptation": "Stratégies d’adaptation",
}


def report_number(value: object, decimals: int = 3) -> str:
    """Format a numeric value for the French-language HTML report."""
    if pd.isna(value):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    return f"{number:.{decimals}f}".replace(".", ",")


def report_integer(value: object) -> str:
    if pd.isna(value):
        return "—"
    return f"{int(round(float(value))):,}".replace(",", " ")


def summary_value(data: dict, *keys: str, default: object = "—") -> object:
    for key in keys:
        if key in data:
            return data[key]
    return default


def normalize_hex_color(value: str, fallback: str = "#315D67") -> str:
    candidate = str(value).strip().upper()
    if len(candidate) == 7 and candidate.startswith("#"):
        try:
            int(candidate[1:], 16)
            return candidate
        except ValueError:
            pass
    return fallback


def blend_hex_color(color: str, target: str, amount: float) -> str:
    """Blend a hexadecimal color toward a target color."""
    source = normalize_hex_color(color)
    destination = normalize_hex_color(target, fallback="#FFFFFF")
    ratio = min(max(float(amount), 0.0), 1.0)
    source_rgb = tuple(int(source[index:index + 2], 16) for index in (1, 3, 5))
    target_rgb = tuple(int(destination[index:index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(
        round(start * (1 - ratio) + end * ratio)
        for start, end in zip(source_rgb, target_rgb)
    )
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


def contrast_text_color(background: str) -> str:
    color = normalize_hex_color(background)
    red, green, blue = (int(color[index:index + 2], 16) for index in (1, 3, 5))
    luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255
    return "#17232C" if luminance > 0.58 else "#FFFFFF"


def build_report_html(
    profile_frame: pd.DataFrame,
    trace: dict,
    analyzable_n: int,
    scenario_capacity_value: int,
    priority_count: int,
    primary_color: str = "#315D67",
) -> str:
    """Build a self-contained report designed for A4 portrait printing."""
    ordered_profiles = profile_frame.sort_values("profil").reset_index(drop=True)
    profile_count = int(ordered_profiles["profil"].nunique(dropna=True))

    profile_names: list[str] = []
    for profile_value in ordered_profiles["profil"]:
        if pd.isna(profile_value):
            profile_names.append("Non classé")
        else:
            profile_names.append(f"Profil {int(profile_value)}")

    distribution_rows: list[str] = []
    for position, row in ordered_profiles.iterrows():
        count = row.get("n", pd.NA)
        share = (float(count) / analyzable_n * 100) if pd.notna(count) and analyzable_n else math.nan
        distribution_rows.append(
            "<tr>"
            f"<td>{html.escape(profile_names[position])}</td>"
            f"<td class='num'>{report_integer(count)}</td>"
            f"<td class='num'>{report_number(share, 1)} %</td>"
            "</tr>"
        )

    excluded_columns = {"profil", "n"}
    dimension_columns = [
        column for column in ordered_profiles.columns if column.startswith("dimension__")
    ]
    indicator_columns = [
        column
        for column in ordered_profiles.columns
        if column not in excluded_columns and column not in dimension_columns
    ]
    profile_headers = "".join(
        f"<th>{html.escape(name)}</th>" for name in profile_names
    )

    def profile_comparison_rows(columns: list[str]) -> str:
        rows: list[str] = []
        for column in columns:
            label = REPORT_LABELS.get(column, column.replace("_", " ").capitalize())
            values = "".join(
                f"<td class='num'>{report_number(row[column])}</td>"
                for _, row in ordered_profiles.iterrows()
            )
            rows.append(f"<tr><td>{html.escape(label)}</td>{values}</tr>")
        if not rows:
            colspan = len(profile_names) + 1
            return f"<tr><td colspan='{colspan}'>Aucune donnée disponible.</td></tr>"
        return "".join(rows)

    generated_at = datetime.now(timezone.utc).strftime("%d/%m/%Y à %H:%M:%S UTC")
    capacity_label = (
        "Non fixée"
        if scenario_capacity_value <= 0
        else f"{report_integer(scenario_capacity_value)} dossiers"
    )
    pipeline_version = summary_value(trace, "version_pipeline")
    source_file = summary_value(trace, "fichier_source")
    representation = summary_value(
        trace,
        "principe de représentation",
        "principe_de_representation",
        default="Distance de Gower sur quatre scores dimensionnels",
    )
    algorithm = summary_value(trace, "algorithme_retenu", default="k-médoïdes")
    retained_k = summary_value(trace, "k_retenu", default=profile_count)
    repetitions = summary_value(
        trace,
        "répétitions_sous_echantillonnage",
        "repetitions_sous_echantillonnage",
    )
    warnings = summary_value(trace, "avertissements", default=[])
    if isinstance(warnings, list):
        warnings_text = "Aucun avertissement détecté" if not warnings else " ; ".join(map(str, warnings))
    else:
        warnings_text = str(warnings)

    primary = normalize_hex_color(primary_color)
    primary_dark = blend_hex_color(primary, "#142F42", 0.58)
    primary_soft = blend_hex_color(primary, "#FFFFFF", 0.93)
    primary_border = blend_hex_color(primary, "#FFFFFF", 0.76)
    primary_text = contrast_text_color(primary)

    safe = lambda value: html.escape(str(value))
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rapport de synthèse — classification et priorisation</title>
  <style>
    @page {{ size: A4 portrait; margin: 14mm 13mm; }}
    * {{ box-sizing: border-box; }}
    html {{ background: #e9eef2; }}
    body {{
      margin: 0;
      color: #263541;
      background: #e9eef2;
      font-family: Aptos, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      font-size: 12px;
      line-height: 1.45;
      font-synthesis: none;
      text-rendering: optimizeLegibility;
      -webkit-font-smoothing: antialiased;
    }}
    .page {{
      width: min(100% - 32px, 190mm);
      margin: 24px auto;
      padding: 16mm 15mm;
      background: #ffffff;
      box-shadow: 0 8px 28px rgba(28, 45, 58, 0.14);
    }}
    .print-action {{ text-align: right; margin-bottom: 12px; }}
    .print-action button {{
      padding: 8px 12px;
      border: 1px solid {primary};
      border-radius: 4px;
      color: {primary_text};
      background: {primary};
      cursor: pointer;
      font-weight: 600;
    }}
    .report-header {{
      padding: 0 0 14px 16px;
      border-left: 5px solid {primary};
      border-bottom: 1px solid #d8e0e5;
    }}
    .eyebrow {{
      margin: 0 0 5px;
      color: {primary_dark};
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      color: {primary_dark};
      font-family: "Aptos Display", "Segoe UI Semibold", "Segoe UI", Arial, sans-serif;
      font-size: 25px;
      font-weight: 650;
      line-height: 1.18;
      letter-spacing: -0.015em;
    }}
    .subtitle {{ margin: 7px 0 0; color: #5f6d77; font-size: 13px; }}
    .meta {{ margin: 8px 0 0; color: #74818a; font-size: 10px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .card {{
      min-height: 78px;
      padding: 12px 14px;
      border: 1px solid #d7e0e5;
      border-radius: 6px;
      background: {primary_soft};
    }}
    .card-label {{ display: block; color: #687680; font-size: 10px; }}
    .card-value {{ display: block; margin-top: 4px; color: {primary_dark}; font-size: 22px; font-weight: 700; }}
    section {{ margin-top: 20px; }}
    h2 {{
      margin: 0 0 9px;
      padding-bottom: 5px;
      color: {primary_dark};
      border-bottom: 2px solid {primary_border};
      font-family: "Aptos Display", "Segoe UI Semibold", "Segoe UI", Arial, sans-serif;
      font-size: 15px;
      font-weight: 650;
    }}
    .section-note {{ margin: -2px 0 9px; color: #697781; font-size: 10px; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    th, td {{
      padding: 7px 8px;
      border: 1px solid #d6dfe4;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    th {{ color: {primary_text}; background: {primary}; font-size: 10px; text-align: left; }}
    tbody tr:nth-child(even) {{ background: {primary_soft}; }}
    td:first-child, th:first-child {{ width: 52%; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .technical td:first-child {{ width: 34%; color: #405662; font-weight: 600; }}
    .notice {{
      margin-top: 20px;
      padding: 11px 13px;
      border-left: 4px solid #b77a25;
      background: #fff8ed;
      color: #62471e;
    }}
    footer {{
      margin-top: 22px;
      padding-top: 9px;
      border-top: 1px solid #d8e0e5;
      color: #78858e;
      font-size: 9px;
    }}
    @media (max-width: 700px) {{
      .page {{ width: 100%; margin: 0; padding: 20px; box-shadow: none; }}
      .cards {{ grid-template-columns: 1fr; }}
    }}
    @media print {{
      html, body {{ background: #ffffff; }}
      .page {{ width: auto; margin: 0; padding: 0; box-shadow: none; }}
      .no-print {{ display: none !important; }}
      .card, h2, thead {{ break-inside: avoid; }}
      tr {{ break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <div class="print-action no-print">
      <button type="button" onclick="window.print()">Imprimer ou enregistrer en PDF</button>
    </div>
    <header class="report-header">
      <p class="eyebrow">Système d’aide à la décision</p>
      <h1>Rapport de synthèse</h1>
      <p class="subtitle">Classification et priorisation analytiques des ménages</p>
      <p class="meta">Généré le {safe(generated_at)}</p>
    </header>

    <div class="cards">
      <div class="card"><span class="card-label">Ménages analysables</span><span class="card-value">{report_integer(analyzable_n)}</span></div>
      <div class="card"><span class="card-label">Profils analytiques</span><span class="card-value">{report_integer(profile_count)}</span></div>
      <div class="card"><span class="card-label">Capacité du scénario</span><span class="card-value">{safe(capacity_label)}</span></div>
      <div class="card"><span class="card-label">Dossiers prioritaires pour vérification</span><span class="card-value">{report_integer(priority_count)}</span></div>
    </div>

    <section>
      <h2>Répartition des ménages par profil</h2>
      <table>
        <thead><tr><th>Profil</th><th>Effectif</th><th>Part</th></tr></thead>
        <tbody>{''.join(distribution_rows)}</tbody>
      </table>
    </section>

    <section>
      <h2>Scores moyens des dimensions</h2>
      <p class="section-note">Les scores sont orientés de 0 à 1. Une valeur plus élevée traduit une intensité plus forte de la vulnérabilité représentée.</p>
      <table>
        <thead><tr><th>Dimension</th>{profile_headers}</tr></thead>
        <tbody>{profile_comparison_rows(dimension_columns)}</tbody>
      </table>
    </section>

    <section>
      <h2>Caractéristiques moyennes des profils</h2>
      <p class="section-note">Le numéro d’un profil est un identifiant analytique ; il ne correspond pas à une phase IPC.</p>
      <table>
        <thead><tr><th>Indicateur orienté</th>{profile_headers}</tr></thead>
        <tbody>{profile_comparison_rows(indicator_columns)}</tbody>
      </table>
    </section>

    <section>
      <h2>Traçabilité méthodologique</h2>
      <table class="technical">
        <tbody>
          <tr><td>Fichier source</td><td>{safe(source_file)}</td></tr>
          <tr><td>Version du pipeline</td><td>{safe(pipeline_version)}</td></tr>
          <tr><td>Représentation</td><td>{safe(representation)}</td></tr>
          <tr><td>Algorithme retenu</td><td>{safe(algorithm)}</td></tr>
          <tr><td>Nombre de profils retenu</td><td>{safe(retained_k)}</td></tr>
          <tr><td>Sous-échantillonnages</td><td>{safe(repetitions)}</td></tr>
          <tr><td>Avertissements techniques</td><td>{safe(warnings_text)}</td></tr>
        </tbody>
      </table>
    </section>

    <div class="notice">
      <strong>Limite d’utilisation.</strong> Ce document présente des profils et un ordre de vérification analytiques. Il ne constitue ni une liste de bénéficiaires ni une décision automatique d’assistance.
    </div>
    <footer>Rapport local pseudonymisé — aucune donnée nominative n’est affichée.</footer>
  </main>
</body>
</html>"""


st.sidebar.header("Scénario de vérification")
scenario = st.sidebar.selectbox(
    "Capacité disponible",
    ["Aucune capacité fixée", "10 %", "20 %", "30 %", "Nombre personnalisé"],
)
custom_capacity = st.sidebar.number_input(
    "Nombre de dossiers",
    min_value=0,
    max_value=max(1, len(assign)),
    value=min(100, len(assign)),
    disabled=scenario != "Nombre personnalisé",
)
capacity = scenario_capacity(scenario, custom_capacity)
assign = apply_capacity(assign, capacity)

classable_n = int(assign["rang_priorisation"].notna().sum())
priority_n = int((assign["statut_scenario"] == "Prioritaire pour vérification").sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ménages analysables", f"{classable_n:,}".replace(",", " "))
c2.metric("Profils", int(assign["profil"].nunique(dropna=True)))
c3.metric("Capacité active", "Non fixée" if capacity == 0 else f"{capacity:,}".replace(",", " "))
c4.metric("Score médian", f"{assign['score_priorisation'].median():.3f}")

st.caption(f"Exécution affichée : {selected_execution_label}")

(
    tab_list,
    tab_profiles,
    tab_visuals,
    tab_provinces,
    tab_household,
    tab_trace,
    tab_new_data,
) = st.tabs(
    [
        "Liste classée",
        "Profils agrégés",
        "Visualisations",
        "Synthèse provinciale",
        "Fiche et suivi",
        "Traçabilité",
        "Nouvelle analyse",
    ]
)

with tab_list:
    f1, f2, f3, f4 = st.columns(4)
    provinces = ["Toutes"]
    if "adm1_name" in assign:
        provinces += sorted(assign["adm1_name"].dropna().astype(str).unique().tolist())
    province = f1.selectbox("Province", provinces)
    profile_values = sorted(assign["profil"].dropna().astype(int).unique().tolist())
    chosen_profile = f2.selectbox("Profil", ["Tous"] + profile_values)
    statuses = ["Tous"] + sorted(assign["statut_scenario"].dropna().unique().tolist())
    status = f3.selectbox("Statut du scénario", statuses)
    minimum_score = f4.number_input(
        "Score minimal", min_value=0.0, max_value=1.0, value=0.0, step=0.05, format="%.2f"
    )

    view = assign.copy()
    if province != "Toutes":
        view = view[view["adm1_name"].astype(str).eq(province)]
    if chosen_profile != "Tous":
        view = view[view["profil"].eq(int(chosen_profile))]
    if status != "Tous":
        view = view[view["statut_scenario"].eq(status)]
    view = view[view["score_priorisation"].fillna(-1).ge(minimum_score)]
    view = view.sort_values("rang_priorisation", na_position="last")

    display_columns = [
        c for c in [
            "menage_pseudonyme", "adm1_name", "profil", "score_priorisation",
            "rang_priorisation", "statut_scenario",
        ] if c in view
    ]
    st.write(f"{len(view):,} ménage(s) affiché(s)".replace(",", " "))
    st.dataframe(view[display_columns], width="stretch", hide_index=True)
    st.download_button(
        "Exporter la vue pseudonymisée",
        data=view[display_columns].to_csv(index=False).encode("utf-8-sig"),
        file_name="vue_pseudonymisee.csv",
        mime="text/csv",
    )

with tab_profiles:
    st.dataframe(profiles, width="stretch", hide_index=True)
    st.caption(
        "Les profils décrivent des configurations moyennes. Le numéro d’un profil n’est pas "
        "un niveau de gravité et ne doit pas être interprété comme une phase IPC."
    )

with tab_visuals:
    st.subheader("Lecture graphique des résultats et de la validation")
    st.caption(
        "Ces graphiques décrivent la partition, sa validation interne et la sensibilité du classement. "
        "Ils ne mesurent pas une 'puissance' prédictive."
    )

    dimension_columns = [c for c in profiles.columns if c.startswith("dimension__")]
    if dimension_columns and "profil" in profiles.columns:
        dimension_chart = profiles.set_index("profil")[dimension_columns].T
        dimension_chart.index = [
            c.replace("dimension__", "").replace("_", " ").capitalize()
            for c in dimension_chart.index
        ]
        dimension_chart.columns = [f"Profil {int(c)}" for c in dimension_chart.columns]
        st.markdown("**Scores moyens des quatre dimensions par profil**")
        # Matplotlib est utilisé ici pour garantir des barres réellement côte à côte
        # quelle que soit la version de Streamlit installée.
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        dimension_chart.plot(kind="bar", ax=ax, width=0.78)
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("Score moyen orienté (0–1)")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption(
            "Échelle 0–1 : une valeur plus élevée correspond à une situation plus défavorable. "
            "Les deux profils sont comparés côte à côte ; leurs scores ne sont pas additionnés."
        )

    score_values = assign["score_priorisation"].dropna()
    if not score_values.empty:
        score_bins = pd.cut(score_values, bins=10, include_lowest=True)
        counts = score_bins.value_counts(sort=False)
        pretty_labels = []
        for interval in counts.index:
            left = max(float(interval.left), 0.0)
            right = min(float(interval.right), 1.0)
            pretty_labels.append(f"{left:.2f}–{right:.2f}".replace(".", ","))
        score_distribution = pd.DataFrame({"Ménages": counts.to_numpy()}, index=pretty_labels)
        st.markdown("**Distribution du score de priorisation**")
        fig, ax = plt.subplots(figsize=(9.5, 4.2))
        ax.bar(score_distribution.index, score_distribution["Ménages"])
        ax.set_xlabel("Intervalle du score")
        ax.set_ylabel("Nombre de ménages")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption("Les classes de score sont présentées sur une échelle commune de 0 à 1.")

    validation_path = OUTPUT / "validation_nombre_profils.csv"
    if validation_path.exists():
        validation_table = pd.read_csv(validation_path)
        st.markdown("**Validation des partitions candidates**")
        st.dataframe(validation_table, width="stretch", hide_index=True)
        k_col = "k" if "k" in validation_table.columns else None
        method_col = next(
            (c for c in ["method", "methode", "algorithme", "algorithm"] if c in validation_table.columns),
            None,
        )
        metrics = [
            c for c in validation_table.columns
            if any(token in c.lower() for token in ["silhouette", "stability", "stabilite", "stabilité", "ari"])
            and pd.api.types.is_numeric_dtype(validation_table[c])
        ]
        if k_col and metrics:
            metric = st.selectbox("Indicateur de validation à visualiser", metrics)
            chart_source = validation_table[[k_col, metric] + ([method_col] if method_col else [])].dropna()
            if method_col:
                chart_source = chart_source.pivot(index=k_col, columns=method_col, values=metric)
            else:
                chart_source = chart_source.set_index(k_col)
            st.line_chart(chart_source)
            st.caption(
                "La sélection finale ne repose pas sur la silhouette seule : la stabilité ARI, "
                "l'indice de Dunn et la taille minimale des groupes sont également pris en compte."
            )

    sensitivity_path = OUTPUT / "sensibilite_priorisation.csv"
    if sensitivity_path.exists():
        st.markdown("**Sensibilité du classement aux pondérations**")
        sensitivity_table = pd.read_csv(sensitivity_path)
        display_sensitivity = sensitivity_table.copy()
        # La p-value n'est pas l'argument central de robustesse dans ce mémoire.
        if "p_value" in display_sensitivity.columns:
            display_sensitivity = display_sensitivity.drop(columns=["p_value"])
        percent_columns = [
            c for c in display_sensitivity.columns
            if c.startswith("recouvrement_top_")
        ]
        for column in percent_columns:
            display_sensitivity[column] = pd.to_numeric(
                display_sensitivity[column], errors="coerce"
            ).map(lambda value: "—" if pd.isna(value) else f"{100 * value:.1f} %".replace(".", ","))
        if "rho_spearman" in display_sensitivity.columns:
            display_sensitivity["rho_spearman"] = pd.to_numeric(
                display_sensitivity["rho_spearman"], errors="coerce"
            ).map(lambda value: "—" if pd.isna(value) else f"{value:.3f}".replace(".", ","))
        rename_sensitivity = {
            "scenario": "Scénario",
            "rho_spearman": "Spearman ρ",
            "n": "Ménages",
            "recouvrement_top_10pct": "Recouvrement Top 10 %",
            "recouvrement_top_20pct": "Recouvrement Top 20 %",
            "recouvrement_top_30pct": "Recouvrement Top 30 %",
        }
        display_sensitivity = display_sensitivity.rename(columns=rename_sensitivity)
        st.dataframe(display_sensitivity, width="stretch", hide_index=True)
        st.caption(
            "Spearman décrit la stabilité globale du rang ; les recouvrements Top 10/20/30 % "
            "montrent la sensibilité près des seuils de capacité."
        )

with tab_provinces:
    st.subheader("Synthèse provinciale de l'échantillon analysé")
    if "adm1_name" not in assign.columns:
        st.info("La province n'est pas disponible dans cette exécution.")
    else:
        provincial = (
            assign.groupby("adm1_name", dropna=False)
            .agg(
                menages=("menage_pseudonyme", "count"),
                score_moyen=("score_priorisation", "mean"),
                score_median=("score_priorisation", "median"),
            )
            .reset_index()
            .rename(columns={"adm1_name": "province"})
        )
        for profile_value in sorted(assign["profil"].dropna().astype(int).unique()):
            counts = (
                assign.loc[assign["profil"].eq(profile_value)]
                .groupby("adm1_name")["menage_pseudonyme"]
                .count()
            )
            provincial[f"profil_{profile_value}_n"] = provincial["province"].map(counts).fillna(0).astype(int)
        if capacity > 0:
            priority_counts = (
                assign.loc[assign["statut_scenario"].eq("Prioritaire pour vérification")]
                .groupby("adm1_name")["menage_pseudonyme"]
                .count()
            )
            provincial["prioritaires_verification_n"] = (
                provincial["province"].map(priority_counts).fillna(0).astype(int)
            )
        st.dataframe(provincial, width="stretch", hide_index=True)
        st.caption(
            "Les effectifs décrivent les ménages présents dans le fichier analysé. "
            "Ils ne doivent pas être interprétés comme un recensement provincial."
        )
        chart = provincial.set_index("province")[["menages"]]
        st.bar_chart(chart)

with tab_household:
    options = assign.sort_values("rang_priorisation", na_position="last")["menage_pseudonyme"].tolist()
    selected_id = st.selectbox("Ménage pseudonymisé", options)
    record = assign.loc[assign["menage_pseudonyme"].eq(selected_id)].iloc[0]
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Profil", "Non classé" if pd.isna(record["profil"]) else int(record["profil"]))
    h2.metric("Score", "—" if pd.isna(record["score_priorisation"]) else f"{record['score_priorisation']:.3f}")
    h3.metric("Rang", "—" if pd.isna(record["rang_priorisation"]) else int(record["rang_priorisation"]))
    h4.metric("Statut", record["statut_scenario"])

    dimension_columns = [c for c in assign if c.startswith("score_dimension__")]
    if dimension_columns:
        dimension_table = pd.DataFrame({
            "Dimension": [c.replace("score_dimension__", "").replace("_", " ") for c in dimension_columns],
            "Score orienté": [record[c] for c in dimension_columns],
        })
        st.dataframe(dimension_table, width="stretch", hide_index=True)

    followup_path = OUTPUT / "suivi_verification_local.csv"
    analyst = st.text_input("Initiales de l’utilisateur", max_chars=12)
    decision = st.selectbox(
        "État du suivi",
        ["Non examiné", "Vérification programmée", "Vérification en cours", "Vérification terminée"],
    )
    note = st.text_area("Observation factuelle")
    if st.button("Enregistrer le suivi"):
        if not analyst.strip():
            st.warning("Indiquez les initiales de l’utilisateur avant l’enregistrement.")
        else:
            row = pd.DataFrame([{
                "horodatage_utc": datetime.now(timezone.utc).isoformat(),
                "utilisateur": analyst.strip(),
                "menage_pseudonyme": selected_id,
                "statut_suivi": decision,
                "observation": note.strip(),
                "capacite_scenario": capacity,
                "statut_scenario": record["statut_scenario"],
            }])
            history = pd.read_csv(followup_path) if followup_path.exists() else pd.DataFrame()
            pd.concat([history, row], ignore_index=True).to_csv(
                followup_path, index=False, encoding="utf-8-sig"
            )
            st.success("Suivi enregistré séparément du score algorithmique.")
    if followup_path.exists():
        history = pd.read_csv(followup_path)
        st.dataframe(
            history[history["menage_pseudonyme"].eq(selected_id)],
            width="stretch",
            hide_index=True,
        )

with tab_trace:
    st.json(summary)
    journal_path = OUTPUT / "journal_execution.json"
    if journal_path.exists():
        with st.expander("Journal d'exécution", expanded=False):
            st.json(json.loads(journal_path.read_text(encoding="utf-8")))
    with st.expander("Personnaliser et prévisualiser le rapport", expanded=True):
        report_color = st.color_picker(
            "Couleur principale du rapport",
            value="#315D67",
            help=(
                "Cette couleur est appliquée aux titres, aux en-têtes de tableaux et aux "
                "éléments d’accent. Les contrastes sont ajustés automatiquement."
            ),
        )
        st.caption(
            "Choisissez la couleur avant le téléchargement. L’aperçu ci-dessous se met à jour automatiquement."
        )
        report = build_report_html(
            profile_frame=profiles,
            trace=summary,
            analyzable_n=classable_n,
            scenario_capacity_value=capacity,
            priority_count=priority_n,
            primary_color=report_color,
        )
        components.html(report, height=760, scrolling=True)
        st.download_button(
            "Télécharger le rapport synthétique",
            data=report.encode("utf-8"),
            file_name="rapport_synthetique.html",
            mime="text/html",
        )

with tab_new_data:
    st.subheader("Analyser un nouveau fichier compatible")
    st.caption(
        "Le fichier officiel du mémoire reste inchangé. Chaque nouvelle analyse est conservée dans runs/."
    )

    uploaded = st.file_uploader(
        "Sélectionner un fichier CSV",
        type=["csv"],
        key="new_analysis_upload",
    )

    if uploaded is not None:
        source_name = Path(uploaded.name).name.replace(" ", "_")
        while source_name.lower().endswith(".csv.csv"):
            source_name = source_name[:-4]
        if not source_name.lower().endswith(".csv"):
            source_name += ".csv"

        source_path = IMPORTS_ROOT / source_name
        source_path.write_bytes(uploaded.getvalue())

        try:
            incoming = pd.read_csv(source_path)
        except Exception as exc:
            st.error(f"Lecture impossible : {exc}")
            incoming = None

        if incoming is not None:
            config_path = ROOT / "config" / "variables.json"
            id_candidates = ["survey_id"]
            if config_path.exists():
                try:
                    config_data = json.loads(config_path.read_text(encoding="utf-8"))
                    configured_ids = config_data.get("id_candidates", [])
                    if isinstance(configured_ids, list):
                        id_candidates = list(dict.fromkeys(configured_ids + id_candidates))
                except (OSError, json.JSONDecodeError):
                    pass

            id_column = next((name for name in id_candidates if name in incoming.columns), None)
            exact_duplicates = int(incoming.duplicated().sum())
            duplicate_ids = 0
            if id_column:
                duplicate_ids = int(incoming[id_column].dropna().duplicated().sum())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Lignes", f"{len(incoming):,}".replace(",", " "))
            m2.metric("Colonnes", incoming.shape[1])
            m3.metric("Doublons exacts", exact_duplicates)
            m4.metric("Identifiants dupliqués", duplicate_ids if id_column else "Non vérifiable")

            hidden_columns = [name for name in id_candidates if name in incoming.columns]
            preview = incoming.drop(columns=hidden_columns, errors="ignore").head(8)
            st.caption("Aperçu pseudonymisé : les identifiants techniques sont masqués.")
            st.dataframe(preview, width="stretch", hide_index=True)

            remove_exact_duplicates = False
            if exact_duplicates:
                remove_exact_duplicates = st.checkbox(
                    "Retirer les lignes strictement identiques dans la copie de travail",
                    value=True,
                    help="Le fichier importé reste intact. Seule la copie transmise au pipeline est nettoyée.",
                )

            working = incoming.drop_duplicates().copy() if remove_exact_duplicates else incoming.copy()
            remaining_duplicate_ids = 0
            if id_column:
                remaining_duplicate_ids = int(working[id_column].dropna().duplicated().sum())

            if remaining_duplicate_ids:
                st.error(
                    "Des identifiants sont encore dupliqués après le contrôle. "
                    "Ils ne sont pas supprimés automatiquement car les lignes peuvent contenir des informations différentes."
                )

            full_validation = st.checkbox(
                "Je confirme que cette analyse est distincte des résultats officiels du chapitre III.",
                value=False,
            )

            can_run = full_validation and remaining_duplicate_ids == 0
            if st.button("Lancer l'analyse", disabled=not can_run):
                working_name = source_name
                if remove_exact_duplicates and exact_duplicates:
                    working_name = f"{Path(source_name).stem}_nettoye.csv"
                working_path = IMPORTS_ROOT / working_name
                working.to_csv(working_path, index=False, encoding="utf-8-sig")

                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = RUNS_ROOT / f"{stamp}_{Path(source_name).stem}"
                run_dir.mkdir(parents=True, exist_ok=True)

                if getattr(sys, "frozen", False):
                    command = [
                        sys.executable, "--pipeline",
                        "--input", str(working_path),
                        "--output", str(run_dir),
                        "--k-min", "2", "--k-max", "6",
                        "--repetitions", "30", "--seed", "42",
                        "--capacity", "0",
                        "--expected-rows", "0",
                        "--expected-columns", "0",
                    ]
                else:
                    command = [
                        sys.executable, str(ROOT / "run_pipeline.py"),
                        "--input", str(working_path),
                        "--output", str(run_dir),
                        "--k-min", "2", "--k-max", "6",
                        "--repetitions", "30", "--seed", "42",
                        "--capacity", "0",
                        "--expected-rows", "0",
                        "--expected-columns", "0",
                    ]

                with st.spinner("Analyse en cours..."):
                    process = subprocess.run(
                        command,
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        env=os.environ.copy(),
                    )

                if process.returncode == 0 and (run_dir / "resume_resultats.json").exists():
                    st.success("Analyse terminée. La nouvelle exécution est enregistrée séparément dans runs/.")
                    st.info("Actualisez la page puis choisissez cette exécution dans la barre latérale.")
                    if process.stdout.strip():
                        with st.expander("Détails techniques"):
                            st.code(process.stdout[-5000:], language="text")
                else:
                    st.error("Le fichier n'a pas pu être analysé avec le protocole configuré.")
                    diagnostic = (process.stdout + "\n" + process.stderr).strip()
                    if diagnostic:
                        with st.expander("Diagnostic technique", expanded=True):
                            st.code(diagnostic[-8000:], language="text")
