"""Speichern/Laden des fertigen Splits, damit alle Modell-Notebooks dieselben Daten sehen."""

import os
from dataclasses import asdict

import torch

from .graph import BUNDLE_VERSION, GraphBundle

DEFAULT_BUNDLE_PATH = os.path.join("processed", "openflights_split.pt")


def save_bundle(bundle: GraphBundle, path: str = DEFAULT_BUNDLE_PATH) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = asdict(bundle)
    payload["meta"] = {**bundle.meta, "bundle_version": BUNDLE_VERSION}
    torch.save(payload, path)
    print(f"Split gespeichert: {path}")
    return path


def load_bundle(path: str = DEFAULT_BUNDLE_PATH, device: torch.device | str | None = None) -> GraphBundle:
    """Lädt den Split. PyG-Data-Objekte sind gepickelt, daher weights_only=False."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} nicht gefunden. Zuerst data_prep_OpenFlights.ipynb ausführen.")
    payload = torch.load(path, weights_only=False)
    version = payload.get("meta", {}).get("bundle_version", 1)
    if version != BUNDLE_VERSION:
        raise RuntimeError(
            f"{path} hat Version {version}, erwartet wird {BUNDLE_VERSION}. "
            "data_prep_OpenFlights.ipynb neu ausführen, um den Split mit der aktuellen Pipeline zu erzeugen."
        )
    bundle = GraphBundle(**payload)
    if device is not None:
        bundle.data = bundle.data.to(device)
        bundle.train = bundle.train.to(device)
        bundle.val = bundle.val.to(device)
        bundle.test = bundle.test.to(device)
    print(f"Split geladen: {path}\n{bundle.summary()}")
    return bundle
