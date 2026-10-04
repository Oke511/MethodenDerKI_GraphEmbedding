"""Datenpipeline und Trainingshilfen für Link Prediction auf dem OpenFlights-Flugroutennetz.

Aufbau:
    openflights.py - Download, Parsing, Feature Engineering und PyG-Graph für OpenFlights
    graph.py       - paarweiser Split mit gemeinsamem Message-Passing-Graphen, Negativ-Sampling
    io.py          - fertigen Split speichern/laden
    models.py      - gemeinsame Bausteine: Eingangsschicht, Decoder, BatchNorm
    training.py    - Trainingsloop, Seeds, Mehr-Seed-Auswertung, Learning-Curve-Plots
    evaluation.py  - Heuristik-Baselines und Fehleranalyse nach Gruppen
"""

from .openflights import (
    ensure_openflights,
    load_openflights,
    OpenFlightsFrames,
    OpenFlightsConfig,
    OPENFLIGHTS_NODE_FEATURES,
    register_airport_feature,
    build_openflights_graph,
)
from .graph import split_graph, reverse_edge_leakage, edges_in_graph, GraphBundle
from .io import save_bundle, load_bundle
from .models import EdgeAwareInput, LinkDecoder, batch_norm, count_parameters
from .training import (
    set_seed,
    train_link_predictor,
    evaluate_link_predictor,
    evaluate_splits,
    run_seeds,
    plot_learning_curves,
)
from .evaluation import heuristic_scores, heuristic_baselines, auc_by_bucket, link_scores

__all__ = [
    "ensure_openflights",
    "load_openflights",
    "OpenFlightsFrames",
    "OpenFlightsConfig",
    "OPENFLIGHTS_NODE_FEATURES",
    "register_airport_feature",
    "build_openflights_graph",
    "split_graph",
    "GraphBundle",
    "reverse_edge_leakage",
    "edges_in_graph",
    "save_bundle",
    "load_bundle",
    "EdgeAwareInput",
    "LinkDecoder",
    "batch_norm",
    "count_parameters",
    "set_seed",
    "train_link_predictor",
    "evaluate_link_predictor",
    "evaluate_splits",
    "run_seeds",
    "plot_learning_curves",
    "heuristic_scores",
    "heuristic_baselines",
    "auc_by_bucket",
    "link_scores",
]
