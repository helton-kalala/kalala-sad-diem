from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def weighted_mean(values: pd.Series, weights: pd.Series | None):
    v = pd.to_numeric(values, errors="coerce")
    if weights is None:
        return float(v.mean())
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    return float(np.average(v[mask], weights=w[mask])) if mask.any() else np.nan


def cluster_profiles(oriented, dim_scores, labels, weights=None):
    data = pd.concat([oriented, dim_scores.add_prefix("dimension__")], axis=1)
    data["profil"] = labels
    if weights is not None:
        data["__weight"] = pd.to_numeric(weights, errors="coerce")
    rows = []
    for profile, block in data.groupby("profil"):
        row = {"profil": int(profile), "n": int(len(block))}
        for col in [c for c in data.columns if c not in {"profil", "__weight"}]:
            row[col] = weighted_mean(block[col], block.get("__weight"))
        rows.append(row)
    return pd.DataFrame(rows)


def save_figures(k_table, profiles, output_dir):
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=.9)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, col, title in zip(
        axes,
        ["silhouette", "dunn", "stabilite_ari_moyenne"],
        ["Silhouette", "Indice de Dunn", "Stabilite ARI"],
    ):
        if "algorithme" in k_table.columns:
            sns.lineplot(
                data=k_table, x="k", y=col, hue="algorithme", style="algorithme",
                markers=True, dashes=False, ax=ax, palette=["#303030", "#888888"],
            )
            if ax is not axes[0] and ax.legend_ is not None:
                ax.legend_.remove()
        else:
            sns.lineplot(data=k_table, x="k", y=col, marker="o", ax=ax, color="#404040")
        ax.set_title(title)
        ax.set_xlabel("Nombre de profils")
    fig.tight_layout()
    fig.savefig(fig_dir / "selection_nombre_profils.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    dim_cols = [c for c in profiles if c.startswith("dimension__")]
    if dim_cols:
        heat = profiles.set_index("profil")[dim_cols]
        heat.columns = [c.replace("dimension__", "").replace("_", " ") for c in heat.columns]
        fig, ax = plt.subplots(figsize=(8, max(3.5, .6 * len(heat))))
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="Greys", vmin=0, vmax=1, ax=ax)
        ax.set_title("Profils moyens par dimension")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Profil")
        fig.tight_layout()
        fig.savefig(fig_dir / "profils_dimensions.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def dump_json(payload, path):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
