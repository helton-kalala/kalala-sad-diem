## Version actuelle du projet

Cette version correspond à la **V2.7 du système SAD DIEM**, obtenue après plusieurs cycles de développement, de tests, de corrections et de vérifications techniques.

Les validations ont notamment porté sur :
- l’exécution complète du pipeline scientifique ;
- la classification des ménages ;
- la stabilité des résultats ;
- le calcul du score et du classement ;
- les tests unitaires ;
- l’import de nouvelles analyses ;
- l’historisation des exécutions ;
- le fonctionnement du tableau de bord ;
- l’installation et le lancement sous Windows.

## Fonctionnement du système

Le système suit la chaîne de traitement suivante :

Données DIEM  
→ Préparation des données  
→ Construction des dimensions  
→ Distance de Gower  
→ Classification des ménages  
→ Validation des résultats  
→ Construction des profils  
→ Score et classement  
→ Historisation  
→ Restitution dans l’interface.

La classification compare notamment **PAM/k-medoids** et la **classification hiérarchique**. Le choix de la solution repose sur plusieurs critères de validation, notamment la silhouette, l’indice de Dunn et la stabilité des partitions. :contentReference[oaicite:0]{index=0}

## Guide intégré du système

L’application contient également un **Guide du système** accessible depuis le menu latéral.

Ce guide permet de comprendre visuellement les différentes étapes :

1. Données DIEM
2. Préparation
3. Distance de Gower
4. Classification
5. Validation
6. Profils
7. Score et rang
8. Historisation
9. Interface

Chaque étape explique son rôle dans le système ainsi que les principales opérations réalisées.

Le guide permet ainsi à un utilisateur, un chercheur ou un évaluateur de comprendre le cheminement complet depuis les données initiales jusqu’au résultat affiché dans le tableau de bord.

## Principe de décision

Le score produit par le système n’est pas une probabilité d’assistance et le rang ne désigne pas automatiquement un bénéficiaire.

Le système constitue un **outil d’aide à la décision** permettant de classer et de prioriser les ménages pour une vérification humaine avant toute décision opérationnelle.
