from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def _init_medoids(distance: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = distance.shape[0]
    first = int(rng.integers(0, n))
    medoids = [first]
    closest = distance[:, first].astype(float)
    while len(medoids) < k:
        probabilities = closest ** 2
        probabilities[medoids] = 0
        total = probabilities.sum()
        if total <= 0:
            candidates = np.setdiff1d(np.arange(n), medoids)
            nxt = int(rng.choice(candidates))
        else:
            nxt = int(rng.choice(n, p=probabilities / total))
        medoids.append(nxt)
        closest = np.minimum(closest, distance[:, nxt])
    return np.array(medoids, dtype=int)


def kmedoids_alternate(distance: np.ndarray, k: int, seed: int = 42, n_init: int = 8, max_iter: int = 100):
    """K-medoids par alternance, avec plusieurs initialisations reproductibles."""
    best = None
    master = np.random.default_rng(seed)
    for _ in range(n_init):
        rng = np.random.default_rng(int(master.integers(0, 2**31 - 1)))
        medoids = _init_medoids(distance, k, rng)
        for _iteration in range(max_iter):
            labels = np.argmin(distance[:, medoids], axis=1)
            new_medoids = medoids.copy()
            for cluster in range(k):
                members = np.flatnonzero(labels == cluster)
                if len(members) == 0:
                    candidates = np.setdiff1d(np.arange(distance.shape[0]), new_medoids)
                    new_medoids[cluster] = int(rng.choice(candidates))
                    continue
                within = distance[np.ix_(members, members)].sum(axis=1)
                new_medoids[cluster] = int(members[np.argmin(within)])
            if np.array_equal(np.sort(new_medoids), np.sort(medoids)):
                medoids = new_medoids
                break
            medoids = new_medoids
        labels = np.argmin(distance[:, medoids], axis=1)
        cost = float(distance[np.arange(distance.shape[0]), medoids[labels]].sum())
        result = {"labels": labels, "medoids": medoids, "cost": cost}
        if best is None or cost < best["cost"]:
            best = result
    return best


def hierarchical_average(distance: np.ndarray, k: int):
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average")
    labels = fcluster(tree, t=k, criterion="maxclust") - 1
    return {"labels": labels.astype(int), "linkage": tree}

