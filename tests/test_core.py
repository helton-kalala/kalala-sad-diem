import numpy as np
import pandas as pd
from pathlib import Path

from src.gower_distance import weighted_gower_numeric
from src.clustering import kmedoids_alternate
from src.features import _binary_any, build_features, load_config
from src.validation import rank_sensitivity, partition_sensitivity, evaluate_candidate_grid


def test_gower_symmetry_and_diagonal():
    x = pd.DataFrame({"a": [0., .5, 1.], "b": [0., 1., 1.]})
    d = weighted_gower_numeric(x, {"a": 1., "b": 1.}, {"b"})
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0)
    assert np.all((d >= 0) & (d <= 1))


def test_kmedoids_two_groups():
    x = pd.DataFrame({"a": [0., .05, .95, 1.], "b": [0., .1, .9, 1.]})
    d = weighted_gower_numeric(x, {"a": 1., "b": 1.})
    result = kmedoids_alternate(d, 2, seed=1, n_init=5)
    assert result["labels"][0] == result["labels"][1]
    assert result["labels"][2] == result["labels"][3]
    assert result["labels"][0] != result["labels"][2]


def test_asymmetric_double_absence_is_not_similarity_evidence():
    x = pd.DataFrame({"continuous": [0., 1.], "shock": [0., 0.]})
    d = weighted_gower_numeric(x, {"continuous": 1., "shock": 1.}, {"shock"})
    assert np.isclose(d[0, 1], 1.0)


def test_shock_group_preserves_all_missing_row():
    df = pd.DataFrame({"s1": [np.nan, 0, 1], "s2": [np.nan, 0, 0]})
    result = _binary_any(df, ["s1", "s2"])
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == 0
    assert result.iloc[2] == 1


def test_shock_group_recodes_official_yes_no_strings():
    df = pd.DataFrame({"s1": ["No", "Yes", None], "s2": ["No", "No", None]})
    result = _binary_any(df, ["s1", "s2"])
    assert result.iloc[0] == 0
    assert result.iloc[1] == 1
    assert np.isnan(result.iloc[2])


def test_official_r5_modalities_restore_four_dimensions():
    config_path = Path(__file__).resolve().parents[1] / "config" / "variables.json"
    config = load_config(config_path)
    df = pd.DataFrame({
        "p_mod": [0.1, 0.5, 0.9],
        "fcs": [80, 45, 20],
        "hdds_score": [10, 5, 2],
        "hhs": [0, 2, 5],
        "rcsi_score": [2, 15, 35],
        "lcsi": [0, 1, 3],
        "tot_income": [500, 200, 50],
        "hh_education": [
            "Completed higher education (university, college) degree",
            "Basic education",
            "No education",
        ],
        "hh_wealth_water": ["Private tap from piped water", "Spring water", "River"],
        "hh_wealth_toilet": [
            "Flush latrine (toilet with water)",
            "Traditional pit latrine (no water)",
            "None - bush",
        ],
        "shock_violenceinsecconf": ["No", "No", "Yes"],
        "shock_higherfoodprices": ["No", "Yes", "Yes"],
        "shock_drought": ["No", "Yes", "No"],
        "shock_sicknessordeathofhh": ["No", "No", "Yes"],
    })
    features, catalogue, warnings = build_features(df, config)
    dimensions = set(catalogue.loc[catalogue["statut"].eq("active"), "dimension"])
    assert dimensions == set(config["required_dimensions"])
    assert features.shape[1] == 14
    assert not warnings


def test_rank_sensitivity_reports_threshold_overlap():
    scores = pd.DataFrame({
        "acces_consommation": [.9, .7, .2, .1],
        "strategies_adaptation": [.8, .6, .3, .1],
        "exposition_chocs": [.7, .5, .2, .2],
        "capacites_structurelles": [.9, .4, .3, .1],
    })
    main = scores.mean(axis=1)
    result = rank_sensitivity(scores, main)
    assert "recouvrement_top_20pct" in result.columns
    equal = result.loc[result["scenario"].eq("dimensions_poids_egaux")].iloc[0]
    assert np.isclose(equal["rho_spearman"], 1.0)
    assert np.isclose(equal["recouvrement_top_20pct"], 1.0)


def test_partition_sensitivity_returns_two_comparisons():
    dimensions = pd.DataFrame({
        "d1": [0.0, .1, .9, 1.0],
        "d2": [0.0, .2, .8, 1.0],
    })
    variables = dimensions.rename(columns={"d1": "v1", "d2": "v2"})
    result = partition_sensitivity(
        np.array([0, 0, 1, 1]),
        dimensions,
        variables,
        {"v1": 1.0, "v2": 1.0},
        set(),
        2,
        seed=1,
    )
    assert set(result["scenario"]) == {"gower_variables_actives", "kmeans_dimensions"}
    assert result["ari_avec_partition_principale"].between(-1, 1).all()


def test_candidate_grid_compares_both_algorithms_and_k_values():
    x = pd.DataFrame({
        "d1": [0.0, .05, .10, .85, .95, 1.0],
        "d2": [0.0, .10, .05, .90, .85, 1.0],
    })
    distance = weighted_gower_numeric(x, {"d1": 1.0, "d2": 1.0})
    table, chosen_k, chosen_method, labels = evaluate_candidate_grid(
        distance, [2, 3], seed=1, repeats=2
    )
    assert set(table["algorithme"]) == {"k-medoids", "hierarchique_moyenne"}
    assert set(table["k"]) == {2, 3}
    assert int(table["retenu"].sum()) == 1
    assert chosen_k in {2, 3}
    assert chosen_method in {"k-medoids", "hierarchique_moyenne"}
    assert len(labels) == len(x)
