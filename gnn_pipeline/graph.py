"""Train/Val/Test-Split für Link Prediction mit Negativ-Sampling.

Paarweiser Split (siehe `split_graph`):
    Jedes ungeordnete Knotenpaar {u, v} landet mit beiden Richtungen im selben Split. Sonst läge für eine
    Val-/Test-Kante A->B die Gegenrichtung B->A im Message-Passing-Graphen, und das Modell könnte sie einfach
    "wiedererkennen". `reverse_edge_leakage` prüft das.

Ein gemeinsamer Message-Passing-Graph:
    Die Train-Paare werden nochmals geteilt: (1 - supervision_ratio) bilden den Message-Passing-Graphen,
    supervision_ratio dienen nur als Trainingslabels (wie `disjoint_train_ratio` bei `RandomLinkSplit`).
    Train, Val und Test rechnen auf *demselben* Graphen. So fehlt jede zu bewertende Kante im Graphen
    (keine "ist dst schon mein Nachbar?"-Abkürzung), und alle drei Splits sehen dieselbe Graphdichte.

Negative:
    "uniform":  beliebige Knotenpaare ohne Kante (Standard in der Literatur, aber leicht: Luftlinie AUC ~0.94).
    "hole":     Jede entfernte Kante hinterlässt an ihren Endpunkten ein "Loch" (geringerer Grad). Beide Endpunkte
                eines Negativs stammen aus den Endpunkten der Positiven desselben Splits und liegen unter den
                `spatial_k` räumlich nächsten Knoten.
    "spatial":  unter den `spatial_k` räumlich nächsten Knoten, ohne Loch-Ausgleich.
    "distance": wie "hole", aber mit derselben Entfernung wie eine echte Kante (`_sample_distance_matched_pairs`).
                Flugrouten sind oft lang; Negative aus der direkten Nachbarschaft wären kürzer als echte Routen und
                damit wieder an der Luftlinie erkennbar. Standard für OpenFlights.
"""

import math
from dataclasses import dataclass, field

import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

# Wird in load_bundle geprüft: ältere Splits müssen mit data_prep_OpenFlights.ipynb neu erzeugt werden.
BUNDLE_VERSION = 3


@dataclass
class GraphBundle:
    """Alles, was die Modell-Notebooks brauchen – und nichts aus den Rohdaten."""

    data: Data            # vollständiger Graph (für Embedding-Visualisierung), data.pos = Koordinaten (km)
    train: Data           # edge_index/edge_attr = gemeinsamer Message-Passing-Graph, edge_label_* = Trainingspaare
    val: Data
    test: Data
    node_feature_columns: list[str]
    edge_attr_columns: list[str]
    node_type: list[str]  # Kategorie pro PyG-Knotenindex (OpenFlights: Weltregion)
    original_node_ids: list[int]
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        type_counts = pd.Series(self.node_type).value_counts().to_dict()
        lines = [
            f"Knoten: {self.data.num_nodes} {type_counts}, Kanten: {self.data.edge_index.size(1)}",
            f"Knotenfeatures ({len(self.node_feature_columns)}): {self.node_feature_columns}",
            f"Kantenfeatures ({len(self.edge_attr_columns)}): {self.edge_attr_columns}",
        ]
        for name, split in [("train", self.train), ("val", self.val), ("test", self.test)]:
            pos = int((split.edge_label == 1).sum())
            neg = int((split.edge_label == 0).sum())
            lines.append(f"{name}: msg-passing Kanten={split.edge_index.size(1)}, Label-Paare pos={pos} neg={neg}")
        for name, split in [("train", self.train), ("val", self.val), ("test", self.test)]:
            leaked, total = reverse_edge_leakage(split, split)
            in_graph = edges_in_graph(split, split)
            lines.append(
                f"{name}: positive Label-Kanten im Message-Passing-Graph: {in_graph}/{total}, "
                f"davon Gegenrichtung im Graph: {leaked}/{total} ({leaked / max(total, 1):.1%})"
            )
        if self.meta:
            lines.append(f"meta: {self.meta}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schlüssel-Hilfsfunktionen
# ---------------------------------------------------------------------------

def _edge_keys(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Gerichteter Schlüssel: (u, v) und (v, u) sind verschieden."""
    return edge_index[0] * num_nodes + edge_index[1]


def _pair_keys(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Richtungsunabhängiger Schlüssel: (u, v) und (v, u) fallen auf (min, max) zusammen."""
    lo = torch.minimum(edge_index[0], edge_index[1])
    hi = torch.maximum(edge_index[0], edge_index[1])
    return lo * num_nodes + hi


def reverse_edge_leakage(msg_graph: Data, split: Data) -> tuple[int, int]:
    """Zählt positive Label-Kanten (u->v) in `split`, deren Gegenrichtung (v->u) im Message-Passing-Graphen
    `msg_graph` liegt. Für Val/Test ist das der eigene Graph des Splits (`reverse_edge_leakage(split, split)`).
    Gibt (geleakt, gesamt) zurück; beim paarweisen Split ist geleakt == 0."""
    num_nodes = msg_graph.num_nodes
    pos = split.edge_label_index[:, split.edge_label == 1]
    reverse_keys = _edge_keys(pos.flip(0), num_nodes)
    msg_keys = _edge_keys(msg_graph.edge_index, num_nodes)
    leaked = int(torch.isin(reverse_keys, msg_keys).sum())
    return leaked, int(pos.size(1))


def edges_in_graph(msg_graph: Data, split: Data) -> int:
    """Anzahl positiver Label-Kanten von `split`, die selbst im Message-Passing-Graphen liegen (soll 0 sein)."""
    num_nodes = msg_graph.num_nodes
    pos = split.edge_label_index[:, split.edge_label == 1]
    return int(torch.isin(_edge_keys(pos, num_nodes), _edge_keys(msg_graph.edge_index, num_nodes)).sum())


# ---------------------------------------------------------------------------
# Negative Sampling
# ---------------------------------------------------------------------------

NEG_STRATEGIES = ("uniform", "spatial", "hole", "distance")


def _knn_candidates(pos: torch.Tensor, k: int) -> torch.Tensor:
    """Die k räumlich nächsten Knoten jedes Knotens (ohne sich selbst), Form [N, k]."""
    _, idx = cKDTree(pos.numpy()).query(pos.numpy(), k=k + 1)
    return torch.as_tensor(idx[:, 1:], dtype=torch.long)


def _sample_negative_pairs(
    excluded_pair_keys: torch.Tensor,
    num_nodes: int,
    num_pairs: int,
    generator: torch.Generator,
    anchors: torch.Tensor | None = None,
    candidates: torch.Tensor | None = None,
    allowed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Zieht `num_pairs` verschiedene ungeordnete Knotenpaare {u, v}, u != v, die nicht in
    `excluded_pair_keys` liegen (existierende Kanten, bereits vergebene Negative). Rejection Sampling.

    candidates is None ("uniform"): u, v gleichverteilt über alle Knoten.
    sonst: u = Endpunkt einer zufälligen Kante aus `anchors` [2, E], v = zufälliger Eintrag aus candidates[u];
    mit `allowed` (bool-Maske über die Knoten) werden nur v aus der Maske akzeptiert. Der Suchradius
    (k nächste Knoten insgesamt) ist damit für alle Splits gleich, unabhängig davon, wie viele Knoten erlaubt sind.
    """
    collected = []
    unique_keys = torch.empty(0, dtype=torch.long)
    for _ in range(1000):
        if unique_keys.numel() >= num_pairs:
            break
        n_draw = int((num_pairs - unique_keys.numel()) * 1.2) + 16
        if candidates is None:
            u = torch.randint(0, num_nodes, (n_draw,), generator=generator)
            v = torch.randint(0, num_nodes, (n_draw,), generator=generator)
        else:
            pick = torch.randint(0, anchors.size(1), (n_draw,), generator=generator)
            side = torch.randint(0, 2, (n_draw,), generator=generator)
            u = anchors[side, pick]
            v = candidates[u, torch.randint(0, candidates.size(1), (n_draw,), generator=generator)]
        keep = u != v
        if allowed is not None:
            keep &= allowed[v]
        keys = torch.minimum(u, v)[keep] * num_nodes + torch.maximum(u, v)[keep]
        collected.append(keys[~torch.isin(keys, excluded_pair_keys)])
        unique_keys = torch.unique(torch.cat(collected))
    else:
        raise RuntimeError("Nicht genug Negative gefunden - spatial_k erhöhen oder neg_sampling_ratio senken.")
    # torch.unique sortiert - wieder mischen
    unique_keys = unique_keys[torch.randperm(unique_keys.numel(), generator=generator)[:num_pairs]]
    return torch.stack([unique_keys // num_nodes, unique_keys % num_nodes])


def _sample_distance_matched_pairs(
    excluded_pair_keys: torch.Tensor,
    pos: torch.Tensor,
    positives: torch.Tensor,
    num_pairs: int,
    generator: torch.Generator,
    window: int,
) -> torch.Tensor:
    """Negative mit Loch *und* passender Entfernung ("distance").

    Pro Ziehung: zufällige positive Kante (u, v) des Splits, u ist ein zufälliger Endpunkt. Gesucht wird w unter
    den Endpunkten der Positiven dieses Splits, dessen Abstand zu u möglichst nah an |u - v| liegt (zufällig unter
    den `window` nächsten Rängen auf jeder Seite). Die Entfernungsverteilung der Negativen entspricht damit der
    der Positiven, die Luftlinie trennt sie nicht mehr. Bei "spatial"/"hole" liegen die Negativen dagegen immer
    unter den nächsten Nachbarn. Bei Flugrouten sind Negative dann kürzer als echte
    Routen, und "weit weg = Link" wird zur umgekehrten Abkürzung.
    Speicher: Distanzmatrix [Anker, erlaubte Knoten], gedacht für Graphen mit einigen tausend Knoten.
    """
    num_nodes = pos.size(0)
    allowed = torch.unique(positives.flatten())
    anchors = allowed                                          # u ist immer ein Endpunkt einer Positiven
    row_of = torch.full((num_nodes,), -1, dtype=torch.long)
    row_of[anchors] = torch.arange(anchors.numel())
    sorted_dist, order = torch.sort(torch.cdist(pos[anchors], pos[allowed]), dim=1)
    pos_dist = (pos[positives[0]] - pos[positives[1]]).norm(dim=1)

    collected = []
    unique_keys = torch.empty(0, dtype=torch.long)
    for _ in range(1000):
        if unique_keys.numel() >= num_pairs:
            break
        n_draw = int((num_pairs - unique_keys.numel()) * 1.2) + 16
        pick = torch.randint(0, positives.size(1), (n_draw,), generator=generator)
        side = torch.randint(0, 2, (n_draw,), generator=generator)
        u = positives[side, pick]
        rows = row_of[u]
        rank = torch.searchsorted(sorted_dist[rows], pos_dist[pick].unsqueeze(1)).squeeze(1)
        rank = (rank + torch.randint(-window, window + 1, (n_draw,), generator=generator)).clamp(0, allowed.numel() - 1)
        w = allowed[order[rows, rank]]
        keep = u != w
        keys = torch.minimum(u, w)[keep] * num_nodes + torch.maximum(u, w)[keep]
        collected.append(keys[~torch.isin(keys, excluded_pair_keys)])
        unique_keys = torch.unique(torch.cat(collected))
    else:
        raise RuntimeError("Nicht genug entfernungsgleiche Negative gefunden - neg_sampling_ratio senken.")
    unique_keys = unique_keys[torch.randperm(unique_keys.numel(), generator=generator)[:num_pairs]]
    return torch.stack([unique_keys // num_nodes, unique_keys % num_nodes])


# ---------------------------------------------------------------------------
# Split-Varianten
# ---------------------------------------------------------------------------

def _split_graph_pairwise(
    data: Data, seed: int, val_ratio: float, test_ratio: float, neg_sampling_ratio: float,
    neg_strategy: str, spatial_k: int, supervision_ratio: float,
) -> tuple[Data, Data, Data]:
    """Paarweiser Split über ungeordnete Knotenpaare {u, v}; beide Richtungen eines Paars erben denselben
    Split (verhindert den reziproken Leak). Alle Splits teilen sich einen Message-Passing-Graphen."""
    if not 0.0 < supervision_ratio < 1.0:
        raise ValueError("supervision_ratio muss in (0, 1) liegen.")
    generator = torch.Generator().manual_seed(seed)
    num_nodes = data.num_nodes

    pair_keys = _pair_keys(data.edge_index, num_nodes)
    unique_pairs, pair_of_edge = torch.unique(pair_keys, return_inverse=True)
    num_pairs = unique_pairs.numel()

    # Pro Knotenpaar: 0 = Message Passing, 1 = Val, 2 = Test, 3 = Train-Supervision (nur Label)
    perm = torch.randperm(num_pairs, generator=generator)
    n_val = int(round(val_ratio * num_pairs))
    n_test = int(round(test_ratio * num_pairs))
    n_sup = int(round(supervision_ratio * (num_pairs - n_val - n_test)))
    pair_split = torch.zeros(num_pairs, dtype=torch.long)
    pair_split[perm[:n_val]] = 1
    pair_split[perm[n_val:n_val + n_test]] = 2
    pair_split[perm[n_val + n_test:n_val + n_test + n_sup]] = 3
    edge_split = pair_split[pair_of_edge]
    msg_mask = edge_split == 0

    # Positive Label: die tatsächlich existierenden gerichteten Kanten (bei OpenFlights beide Richtungen)
    label_split = {"train": 3, "val": 1, "test": 2}
    positives = {name: data.edge_index[:, edge_split == s] for name, s in label_split.items()}

    # Negative Label: Paare, die in keiner Richtung existieren, jeweils in beide Richtungen eingetragen
    # (neg/pos ~= neg_sampling_ratio, Negative symmetrisch). Splits nacheinander, damit sie disjunkt bleiben.
    knn = _knn_candidates(data.pos, spatial_k) if neg_strategy in ("spatial", "hole") else None
    excluded = unique_pairs
    negatives = {}
    for name in ("train", "val", "test"):
        n_neg_pairs = math.ceil(neg_sampling_ratio * positives[name].size(1) / 2)
        if neg_strategy == "distance":
            chunk = _sample_distance_matched_pairs(excluded, data.pos, positives[name], n_neg_pairs, generator,
                                                   window=spatial_k // 2)
            excluded = torch.cat([excluded, chunk[0] * num_nodes + chunk[1]])
            negatives[name] = torch.cat([chunk, chunk.flip(0)], dim=1)
            continue
        anchors, allowed = None, None
        if neg_strategy == "spatial":
            anchors = data.edge_index
        elif neg_strategy == "hole":
            # u und v sind Endpunkte von Positiven dieses Splits, haben also ebenfalls ein "Loch"
            anchors = positives[name]
            allowed = torch.zeros(num_nodes, dtype=torch.bool)
            allowed[anchors.flatten()] = True
        chunk = _sample_negative_pairs(excluded, num_nodes, n_neg_pairs, generator, anchors, knn, allowed)
        excluded = torch.cat([excluded, chunk[0] * num_nodes + chunk[1]])
        negatives[name] = torch.cat([chunk, chunk.flip(0)], dim=1)

    def make_split(name: str) -> Data:
        label_index = torch.cat([positives[name], negatives[name]], dim=1)
        label = torch.cat([torch.ones(positives[name].size(1)), torch.zeros(negatives[name].size(1))])
        return Data(
            x=data.x,
            pos=data.pos,
            edge_index=data.edge_index[:, msg_mask],
            edge_attr=data.edge_attr[msg_mask],
            num_nodes=num_nodes,
            edge_label_index=label_index,
            edge_label=label,
        )

    return make_split("train"), make_split("val"), make_split("test")


def split_graph(
    data: Data,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    neg_sampling_ratio: float = 1.0,
    neg_strategy: str = "distance",
    spatial_k: int = 6,
    supervision_ratio: float = 0.3,
) -> tuple[Data, Data, Data]:
    """Paarweiser Train/Val/Test-Split mit festem Seed und gemeinsamem Message-Passing-Graphen
    (siehe Modul-Docstring).

    supervision_ratio: Anteil der Train-Paare, die nur als Trainingslabels dienen (nicht im Graphen).
    neg_strategy:      "distance" (Standard), "hole", "spatial" oder "uniform" (siehe Modul-Docstring);
                       bei "distance" ist das Fenster spatial_k // 2 Ränge um die passende Entfernung.

    Rückgabe (train, val, test), jeweils ein Data-Objekt mit:
    - x/pos: Knotenfeatures und Koordinaten des vollständigen Graphen
    - edge_index/edge_attr: gemeinsamer Message-Passing-Graph
    - edge_label_index/edge_label: zu bewertende Paare (1 = Kante, 0 = Negativ)
    Die Zielkanten tragen bewusst keine Kantenfeatures: Der Decoder darf nur Knoten-Embeddings sehen.
    """
    if neg_strategy not in NEG_STRATEGIES:
        raise ValueError(f"neg_strategy muss eines von {NEG_STRATEGIES} sein, nicht {neg_strategy!r}.")
    return _split_graph_pairwise(
        data, seed, val_ratio, test_ratio, neg_sampling_ratio, neg_strategy, spatial_k, supervision_ratio
    )
