"""Bausteine, die alle Modell-Notebooks teilen, damit sich die Modelle nur im Message Passing unterscheiden."""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


def batch_norm(num_features: int) -> nn.BatchNorm1d:
    """BatchNorm ohne Running-Stats: normalisiert immer mit den Statistiken des aktuellen Graphen.

    Bei Full-Batch-Training sieht jeder Forward-Pass alle Knoten; Training und Auswertung rechnen auf
    demselben Graphen, die Normalisierung ist dadurch in beiden Modi identisch. Mit Running-Stats würden
    eval- und train-Modus voneinander abweichen und die Loss-Kurven springen.
    """
    return nn.BatchNorm1d(num_features, track_running_stats=False)


class EdgeAwareInput(nn.Module):
    """Eingangsschicht: h_i = W_x x_i + sum_{j->i} W_e e_ji.

    In GATv2 wirken Kantenfeatures nur auf die Attention-Gewichte, nicht auf die Nachricht, und der Grad geht
    durch die Softmax-Normierung verloren. Diese Schicht bringt Kantenfeatures und Grad (Summe über eingehende
    Kanten) direkt in die Knotenrepräsentation, für beide Modelle gleich.
    """

    def __init__(self, in_channels: int, edge_dim: int, hidden_channels: int):
        super().__init__()
        self.lin_x = nn.Linear(in_channels, hidden_channels)
        self.lin_e = nn.Linear(edge_dim, hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        incoming = scatter(self.lin_e(edge_attr), edge_index[1], dim=0, dim_size=x.size(0), reduce="sum")
        return self.lin_x(x) + incoming


class LinkDecoder(nn.Module):
    """MLP auf der Kombination von h_src und h_dst -> ein Logit pro Paar.

    mode="concat":   concat(h_src, h_dst), richtungsabhängig (A->B != B->A), für gerichtete Netze.
    mode="hadamard": h_src * h_dst, symmetrisch, für ungerichtete Netze (OpenFlights). Das elementweise Produkt
                     misst direkt, wie gut zwei Embeddings zusammenpassen, und muss die Symmetrie nicht erst lernen.

    Nutzt bewusst keine Features der Zielkante, sonst wäre die Vorhersage trivial.
    """

    MODES = ("concat", "hadamard")

    def __init__(self, embedding_dim: int, hidden_channels: int, dropout: float = 0.3, mode: str = "concat"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode muss eines von {self.MODES} sein, nicht {mode!r}.")
        self.mode = mode
        in_dim = embedding_dim * 2 if mode == "concat" else embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, node_embeddings: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        src, dst = node_embeddings[edge_label_index[0]], node_embeddings[edge_label_index[1]]
        pair = torch.cat([src, dst], dim=1) if self.mode == "concat" else src * dst
        return self.mlp(pair).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
