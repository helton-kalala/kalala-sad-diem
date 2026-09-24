# Projet Python — mémoire de DEA DIEM RDC Round 5

## Correctif V2 — recodages officiels Round 5

Cette version corrige un défaut détecté lors de la première exécution sur le fichier officiel :

- les indicateurs de chocs encodés sous forme `Yes`/`No` sont désormais lus comme variables binaires ;
- les modalités exactes de `hh_education`, `hh_wealth_water` et `hh_wealth_toilet` sont reconnues ;
- les quatre dimensions prévues sont obligatoires au niveau de la base ;
- le résumé indique le nombre réellement utilisé de dimensions, au lieu d'un libellé fixe ;
- deux tests ciblés ont été ajoutés, portant le total à neuf tests unitaires.

Les rangs ordinaux retenus pour l'eau et l'assainissement constituent une opérationnalisation
scientifique explicite. Ils doivent être présentés au guide de mémoire pour validation avant le
dépôt définitif du travail. Voir `NOTE_CORRECTIF_V2.txt`.

Ce projet exécute le protocole de classification et de priorisation décrit dans la version unifiée du mémoire. Il ne contient aucune microdonnée FAO et ne produit aucune décision d’assistance.

## Source finale attendue

Le fichier scientifique final est le CSV officiel FAO `data_anon_df_COD_R5.csv`, annoncé dans le mémoire avec 2 716 lignes et 310 variables. Il doit être copié localement dans `data/raw/` sans être joint au code, au mémoire ou aux annexes publiques.

Le classeur réduit de 89 variables peut servir à un essai technique séparé. Ses chiffres ne doivent pas remplacer les résultats de l’exécution officielle.

## Ordre conseillé sous Windows

1. Installer Python 3.12 ou 3.13 en version 64 bits depuis python.org. L'installateur du projet detecte aussi une installation standard qui n'a pas encore ete ajoutee au `PATH` de Windows.
2. Exécuter `01_INSTALLER.bat`.
3. Copier le CSV officiel dans `data\raw`.
4. Exécuter `02_TEST_TECHNIQUE.bat`.
5. Exécuter `04_TESTS_UNITAIRES.bat`.
6. Exécuter `03_ANALYSE_FINALE.bat`.
7. Contrôler les fichiers du dossier `outputs` avant de rédiger le chapitre III.
8. Exécuter `05_TABLEAU_DE_BORD.bat`.

## Commande finale équivalente

```text
.venv\Scripts\python.exe run_pipeline.py --input "data\raw\data_anon_df_COD_R5.csv" --output "outputs" --k-min 2 --k-max 6 --repetitions 30 --seed 42 --capacity 0 --expected-rows 2716 --expected-columns 310
```

La valeur `--capacity 0` conserve le score et le rang sans imposer de statut prioritaire. La capacité est ensuite appliquée dans le tableau de bord comme scénario de vérification.

## Méthode codée

- audit du fichier et empreintes SHA-256 ;
- quatorze variables candidates réparties en quatre dimensions ;
- normalisation robuste et orientation vers `1 = situation plus défavorable` ;
- distance de Gower principale sur les quatre scores dimensionnels à poids égaux ;
- comparaison conjointe de PAM/k-medoids et de la hiérarchie moyenne pour chaque `k = 2..6` ;
- sélection quantitative pondérée : silhouette 40 %, stabilité 40 %, Dunn 20 %, avec rejet des groupes inférieurs à 5 % ;
- score et rang séparés du numéro de profil ;
- sensibilité des pondérations, des listes top 10/20/30 %, de la représentation à quatorze variables et de k-means ;
- restitution pseudonymisée et suivi humain séparé.

## Règles d’interprétation

Les profils sont exploratoires et propres à la vague analysée. Le score n’est pas une probabilité. Le rang ne désigne pas un bénéficiaire. Une capacité crée une liste de dossiers prioritaires pour vérification, jamais une décision automatique d’assistance.

## Sorties essentielles

- `audit_resume.json` et `audit_variables.csv` ;
- `variables_actives_et_transformations.csv` ;
- `scores_dimensions_intermediaires.csv` ;
- `validation_nombre_profils.csv` ;
- `comparaison_algorithmes.csv` ;
- `profils_agreges.csv` ;
- `classement_pseudonymise_usage_local.csv` ;
- `sensibilite_priorisation.csv` et `sensibilite_partitions.csv` ;
- `journal_execution.json` et `AVERTISSEMENTS.txt`.

Toute valeur du chapitre III doit être recopiée depuis ces sorties après contrôle, et non depuis l’ancien classeur de développement.
