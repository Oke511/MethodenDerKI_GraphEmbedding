"""Gemeinsamer Trainingsloop für Link Prediction.

Erwartete Modell-Schnittstelle (so implementieren es die Modellklassen in den GATv2-/GINE-Notebooks):
    model(x, edge_index, edge_attr) -> Knoten-Embeddings
    model.predict_link(node_embeddings, edge_label_index) -> Logits pro Kantenpaar

Wichtige Eigenschaften des Loops:
- Train, Val und Test rechnen auf demselben Message-Passing-Graphen (siehe `graph.py`); keine zu bewertende
  Kante liegt selbst im Graphen. Pro Epoche wird deshalb nur ein Forward-Pass für die Auswertung gebraucht.
- `train_loss`/`train_auc` werden wie Val/Test im eval-Modus gemessen (ohne Dropout), damit die Kurven direkt
  vergleichbar sind; der Loss des Optimierungsschritts (mit Dropout) steht in `train_loss_step`.
- Checkpoint: Die Gewichte der Epoche mit der besten Val-AUC werden am Ende wiederhergestellt.
- Stabilisierung: linearer LR-Warmup, Gradient Clipping, ReduceLROnPlateau auf der Val-AUC, Weight Decay.
"""

import copy
import os
import random

# Muss vor der ersten cuBLAS-Nutzung gesetzt sein, sonst sind Matmuls auf CUDA nicht deterministisch
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import Data

from .evaluation import link_scores


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seedet alle RNGs und schaltet (soweit verfügbar) deterministische Kernels ein.

    Manche Scatter-Operationen in PyG haben auf CUDA keine deterministische Variante; dafür gibt
    PyTorch eine Warnung aus (warn_only=True). Vollständig bitgleiche GPU-Läufe sind deshalb nicht
    garantiert - Modellvergleiche daher immer über mehrere Seeds (`run_seeds`).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.benchmark = not deterministic


def _require_graph(split: Data) -> None:
    if "edge_attr" not in split:
        raise RuntimeError("Split ohne edge_attr - data_prep_OpenFlights.ipynb mit der aktuellen Pipeline neu ausführen.")


def _same_graph(a: Data, b: Data) -> bool:
    return a.edge_index.shape == b.edge_index.shape and bool(torch.equal(a.edge_index, b.edge_index))


def _scores(model, emb: torch.Tensor, split: Data, loss_fn) -> tuple[float, float]:
    scores = model.predict_link(emb, split.edge_label_index)
    labels = split.edge_label.float()
    loss = loss_fn(scores, labels).item()
    auc = roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())
    return loss, auc


@torch.no_grad()
def evaluate_splits(model, splits: dict[str, Data], loss_fn=None) -> dict[str, tuple[float, float]]:
    """(BCE-Loss, ROC-AUC) je Split im eval-Modus. Splits mit identischem Message-Passing-Graphen teilen
    sich einen Forward-Pass."""
    loss_fn = loss_fn or BCEWithLogitsLoss()
    model.eval()
    results, cache = {}, []
    for name, split in splits.items():
        _require_graph(split)
        emb = next((e for g, e in cache if _same_graph(g, split)), None)
        if emb is None:
            emb = model(split.x, split.edge_index, split.edge_attr)
            cache.append((split, emb))
        results[name] = _scores(model, emb, split, loss_fn)
    return results


def evaluate_link_predictor(model, split: Data, loss_fn=None) -> tuple[float, float]:
    """Embeddings auf dem Message-Passing-Graphen des Splits, bewertet auf split.edge_label_index.
    Gibt (BCE-Loss, ROC-AUC) zurück."""
    return evaluate_splits(model, {"split": split}, loss_fn)["split"]


def train_link_predictor(
    model,
    train: Data,
    val: Data,
    test: Data,
    epochs: int = 300,
    lr: float = 0.002,
    weight_decay: float = 5e-4,
    grad_clip: float | None = 1.0,
    warmup_epochs: int = 10,
    lr_patience: int = 20,
    lr_factor: float = 0.5,
    min_lr: float = 1e-5,
    early_stopping_patience: int | None = None,
    restore_best: bool = True,
    log_every: int | None = 10,
) -> dict[str, list[float]]:
    """Full-Batch-Training, gibt die History (pro Epoche) zurück.

    Keys: train_loss, train_auc, val_loss, val_auc, test_loss, test_auc (alle im eval-Modus),
    train_loss_step (Loss des Optimierungsschritts im train-Modus), lr.
    Mit restore_best=True hat `model` danach die Gewichte der Epoche mit der besten Val-AUC.
    Test wird nur protokolliert, nie zur Modellauswahl verwendet.
    """
    for split in (train, val, test):
        _require_graph(split)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_factor, patience=lr_patience, min_lr=min_lr
    )
    loss_fn = BCEWithLogitsLoss()
    splits = {"train": train, "val": val, "test": test}
    keys = ["train_loss", "train_auc", "val_loss", "val_auc", "test_loss", "test_auc", "train_loss_step", "lr"]
    history: dict[str, list[float]] = {k: [] for k in keys}
    best_auc, best_epoch, best_state = -1.0, 0, None

    for epoch in range(1, epochs + 1):
        if epoch <= warmup_epochs:
            for group in optimizer.param_groups:
                group["lr"] = lr * epoch / warmup_epochs

        model.train()
        optimizer.zero_grad()
        emb = model(train.x, train.edge_index, train.edge_attr)
        loss = loss_fn(model.predict_link(emb, train.edge_label_index), train.edge_label.float())
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        results = evaluate_splits(model, splits, loss_fn)
        current_lr = optimizer.param_groups[0]["lr"]
        val_auc = results["val"][1]
        if epoch > warmup_epochs:
            scheduler.step(val_auc)

        for name, (split_loss, split_auc) in results.items():
            history[f"{name}_loss"].append(split_loss)
            history[f"{name}_auc"].append(split_auc)
        history["train_loss_step"].append(loss.item())
        history["lr"].append(current_lr)

        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = copy.deepcopy(model.state_dict())

        if log_every and (epoch % log_every == 0 or epoch == 1 or epoch == epochs):
            print(
                f"Epoch {epoch:03d} | train loss {results['train'][0]:.4f} AUC {results['train'][1]:.4f} | "
                f"val loss {results['val'][0]:.4f} AUC {val_auc:.4f} | test AUC {results['test'][1]:.4f} | "
                f"lr {current_lr:.1e}"
            )
        if early_stopping_patience is not None and epoch - best_epoch >= early_stopping_patience:
            if log_every:
                print(f"Early Stopping in Epoche {epoch} (keine Verbesserung seit Epoche {best_epoch}).")
            break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    if log_every:
        best = best_epoch - 1
        print(
            f"\nBeste Val-AUC {history['val_auc'][best]:.4f} in Epoche {best_epoch} "
            f"(Test-AUC dort: {history['test_auc'][best]:.4f}); Test-AUC letzte Epoche {history['test_auc'][-1]:.4f}"
            + (f"\nModell auf Epoche {best_epoch} zurückgesetzt." if restore_best else "")
        )
    return history


def run_seeds(make_model, train: Data, val: Data, test: Data, seeds, **train_kwargs) -> tuple[pd.DataFrame, list[dict]]:
    """Trainiert `make_model()` je Seed neu und sammelt die Kennzahlen an der besten Val-Epoche.

    Gibt (DataFrame mit einer Zeile pro Seed, Liste der Histories) zurück. `test_ap` ist die Average Precision
    auf Test (Modell der besten Val-Epoche, verlangt restore_best=True). Einzelne Läufe streuen
    (Initialisierung, Dropout, nicht-deterministische GPU-Kernels) - Vergleiche zwischen
    Modellen deshalb über Mittelwert ± Standardabweichung.
    """
    train_kwargs.setdefault("log_every", None)
    device = train.x.device
    rows, histories = [], []
    for seed in seeds:
        set_seed(seed)
        model = make_model().to(device)
        history = train_link_predictor(model, train, val, test, **train_kwargs)
        best = int(np.argmax(history["val_auc"]))
        scores, labels = link_scores(model, test)
        rows.append({
            "seed": seed,
            "best_epoch": best + 1,
            "val_auc": history["val_auc"][best],
            "test_auc": history["test_auc"][best],
            "test_ap": average_precision_score(labels, scores),
            "test_loss": history["test_loss"][best],
        })
        histories.append(history)
        print(f"Seed {seed}: beste Val-AUC {rows[-1]['val_auc']:.4f} (Epoche {best + 1}), Test-AUC dort {rows[-1]['test_auc']:.4f}")
    df = pd.DataFrame(rows)
    print(
        f"\n{len(df)} Seeds: Val-AUC {df['val_auc'].mean():.4f} ± {df['val_auc'].std():.4f} | "
        f"Test-AUC {df['test_auc'].mean():.4f} ± {df['test_auc'].std():.4f}"
    )
    return df, histories


# Farben für die Learning Curves (farbfehlsicht-sicher, Reihenfolge fest): Train = blau, Val = orange, Test = aqua
_SERIES_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
_SURFACE, _INK, _MUTED, _GRID, _AXIS = "#ffffff", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def _style_axis(ax, xlabel: str, ylabel: str, n_epochs: int):
    # Farben explizit setzen: PyCharm rendert Notebook-Plots sonst mit dunklem Hintergrund/weißem Text
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.grid(True, color=_GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED, labelcolor=_INK, length=0)
    ax.set_xlabel(xlabel, color=_INK)
    ax.set_ylabel(ylabel, color=_INK)
    ax.set_xlim(1, n_epochs * 1.22)   # rechts Platz für Endlabels
    ax.set_xticks([t for t in ax.get_xticks() if 0 < t <= n_epochs])


def _label_line_ends(ax, series: list[tuple[str, float, str]], fmt: str, x_end: int):
    """Wert am Linienende beschriften; bei Kollision Labels vertikal auseinanderschieben, Leader-Linie verbindet."""
    to_frac = lambda v: ax.transAxes.inverted().transform(ax.transData.transform((x_end, v)))[1]
    min_gap = 0.07   # Achsenanteil
    ordered = sorted(series, key=lambda s: to_frac(s[1]))
    positions: list[float] = []
    for _, y, _ in ordered:
        f = to_frac(y)
        positions.append(f if not positions else max(f, positions[-1] + min_gap))
    overflow = positions[-1] - 0.97
    if overflow > 0:
        positions = [p - overflow for p in positions]
    for (name, y, color), f in zip(ordered, positions):
        ax.annotate(
            f"{name} {fmt.format(y)}",
            xy=(x_end, y), xycoords="data", xytext=(1.03, f), textcoords="axes fraction",
            va="center", ha="left", fontsize=9, color=_INK,
            arrowprops=dict(arrowstyle="-", color=color, linewidth=1, shrinkA=0, shrinkB=3),
            annotation_clip=False,
        )


def _mark_point(ax, x: float, y: float, color: str):
    ax.plot(x, y, "o", ms=8, color=color, markeredgecolor=_SURFACE, markeredgewidth=2, zorder=5)


def plot_learning_curves(history: dict[str, list[float]], title: str = "Learning Curve"):
    """Zwei Panels: links BCE-Loss, rechts ROC-AUC, jeweils für Train/Val/Test.

    Alle Kurven sind im eval-Modus auf demselben Message-Passing-Graphen gemessen und damit direkt
    vergleichbar. Die Epoche mit der besten Val-AUC ist in beiden Panels als senkrechte Linie markiert;
    auf diese Gewichte wird das Modell nach dem Training zurückgesetzt.
    """
    n = len(history["train_loss"])
    epochs = list(range(1, n + 1))
    best = max(range(n), key=lambda i: history["val_auc"][i])
    best_ep = best + 1

    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(15, 5.4))
    fig.patch.set_facecolor(_SURFACE)

    # --- Loss ---
    loss_series = [
        ("Train", "train_loss", _SERIES_COLORS["train"]),
        ("Val", "val_loss", _SERIES_COLORS["val"]),
        ("Test", "test_loss", _SERIES_COLORS["test"]),
    ]
    for label, key, color in loss_series:
        ax_loss.plot(epochs, history[key], color=color, linewidth=2, solid_joinstyle="round", label=f"{label}-Loss")
    _style_axis(ax_loss, "Epoche", "BCE-Loss", n)
    ax_loss.set_title("Loss", loc="left", color=_INK, fontsize=12)
    _label_line_ends(ax_loss, [(l, history[k][-1], c) for l, k, c in loss_series], "{:.3f}", n)

    # --- ROC-AUC ---
    auc_series = [
        ("Train", "train_auc", _SERIES_COLORS["train"]),
        ("Val", "val_auc", _SERIES_COLORS["val"]),
        ("Test", "test_auc", _SERIES_COLORS["test"]),
    ]
    for label, key, color in auc_series:
        ax_auc.plot(epochs, history[key], color=color, linewidth=2, solid_joinstyle="round", label=f"{label}-ROC-AUC")
    y_min = min(min(history[k]) for _, k, _ in auc_series)
    ax_auc.set_ylim(max(0.0, 0.1 * int(y_min * 10) - 0.05), 1.01)
    _style_axis(ax_auc, "Epoche", "ROC-AUC", n)
    ax_auc.set_title("ROC-AUC", loc="left", color=_INK, fontsize=12)
    _label_line_ends(ax_auc, [(l, history[k][-1], c) for l, k, c in auc_series], "{:.3f}", n)

    # --- beste Val-Epoche in beiden Panels ---
    for ax in (ax_loss, ax_auc):
        ax.axvline(best_ep, color=_MUTED, linewidth=1, linestyle="-", alpha=0.8)
    _mark_point(ax_loss, best_ep, history["val_loss"][best], _SERIES_COLORS["val"])
    _mark_point(ax_auc, best_ep, history["val_auc"][best], _SERIES_COLORS["val"])
    ax_auc.annotate(
        f"beste Val-AUC {history['val_auc'][best]:.3f} in Epoche {best_ep}\n"
        f"Test-AUC dort {history['test_auc'][best]:.3f} · letzte Epoche {history['test_auc'][-1]:.3f}",
        xy=(best_ep, history["val_auc"][best]), xycoords="data",
        xytext=(0.97, 0.06), textcoords="axes fraction", ha="right", va="bottom", fontsize=9, color=_INK,
        arrowprops=dict(arrowstyle="-", color=_MUTED, linewidth=1, shrinkA=0, shrinkB=6),
    )

    for ax in (ax_loss, ax_auc):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=9,
                  labelcolor=_INK)

    fig.suptitle(title, x=0.01, ha="left", color=_INK, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
