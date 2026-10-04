"""Einordnung der Modellergebnisse: Heuristik-Baselines und Fehleranalyse nach Gruppen."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data


def _undirected_adjacency(edge_index: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    ei = edge_index.cpu().numpy()
    a = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(num_nodes, num_nodes)).tocsr()
    a = ((a + a.T) > 0).astype(np.float64)
    a.setdiag(0)
    a.eliminate_zeros()
    return a.tocsr()


def heuristic_scores(split: Data) -> dict[str, np.ndarray]:
    """Klassische Link-Prediction-Heuristiken für split.edge_label_index, berechnet auf dem
    Message-Passing-Graphen des Splits (ungerichtet) und den Koordinaten split.pos (OpenFlights: Punkte auf der Erdkugel in km)."""
    u, v = (t.cpu().numpy() for t in split.edge_label_index)
    adj = _undirected_adjacency(split.edge_index, split.num_nodes)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv_log_deg = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0)
    pos = split.pos.cpu().numpy()
    return {
        "-Distanz (Koordinaten)": -np.linalg.norm(pos[u] - pos[v], axis=1),
        "Gemeinsame Nachbarn": np.asarray(adj[u].multiply(adj[v]).sum(axis=1)).ravel(),
        "Adamic-Adar": np.asarray(adj[u].multiply(adj[v] @ sp.diags(inv_log_deg)).sum(axis=1)).ravel(),
        "Niedriger Grad (Loch)": -(deg[u] + deg[v]),
        "Preferential Attachment": deg[u] * deg[v],
    }


def heuristic_baselines(train: Data, val: Data, test: Data) -> pd.DataFrame:
    """ROC-AUC der Heuristiken je Split. Zeigt, wie schwer die Aufgabe ohne Lernen ist.

    - Hohe AUC für "-Distanz": Negative liegen räumlich weit auseinander (leichte Negative, z. B. "uniform").
    - "Niedriger Grad (Loch)": Die entfernte Label-Kante hinterlässt an ihren Endpunkten einen geringeren Grad.
      Bei OpenFlights ist der Wert < 0.5, weil dort ein *hoher* Grad (Hub) auf eine Route hinweist.
    - Werte unter 0.5 bedeuten, dass die Heuristik umgekehrt trennt (1 - AUC).
    """
    rows = {}
    for name, split in [("train", train), ("val", val), ("test", test)]:
        labels = split.edge_label.cpu().numpy()
        rows[name] = {h: roc_auc_score(labels, s) for h, s in heuristic_scores(split).items()}
    return pd.DataFrame(rows).rename_axis("Heuristik")


@torch.no_grad()
def link_scores(model, split: Data) -> tuple[np.ndarray, np.ndarray]:
    """(Wahrscheinlichkeiten, Labels) für split.edge_label_index, Modell im eval-Modus."""
    model.eval()
    emb = model(split.x, split.edge_index, split.edge_attr)
    scores = torch.sigmoid(model.predict_link(emb, split.edge_label_index))
    return scores.cpu().numpy(), split.edge_label.cpu().numpy()


def auc_by_bucket(scores: np.ndarray, labels: np.ndarray, buckets: np.ndarray, name: str = "Gruppe") -> pd.DataFrame:
    """Fehleranalyse: ROC-AUC innerhalb jeder Gruppe (Positive und Negative derselben Gruppe).

    Positive werden nur mit Negativen derselben Gruppe verglichen, z. B. Kurz- mit Kurzstrecken. Das zeigt, ob das Modell auch innerhalb einer Gruppe trennt oder nur zwischen Gruppen.
    """
    rows = []
    for b in pd.unique(buckets):
        mask = buckets == b
        n_pos, n_neg = int(labels[mask].sum()), int((1 - labels[mask]).sum())
        rows.append({
            name: b,
            "positive Paare": n_pos,
            "negative Paare": n_neg,
            "ROC-AUC": roc_auc_score(labels[mask], scores[mask]) if n_pos and n_neg else np.nan,
            "mittlere Wahrscheinlichkeit (pos)": float(scores[mask & (labels == 1)].mean()) if n_pos else np.nan,
            "mittlere Wahrscheinlichkeit (neg)": float(scores[mask & (labels == 0)].mean()) if n_neg else np.nan,
        })
    return pd.DataFrame(rows).set_index(name)
