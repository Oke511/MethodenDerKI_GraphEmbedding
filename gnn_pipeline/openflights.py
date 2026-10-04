"""OpenFlights-Flugroutennetz: Download, Parsing, Feature Engineering und PyG-Graph.

Quelle: https://openflights.org/data (Open Database License), Dateien `airports.dat` und `routes.dat` aus
https://github.com/jpatokal/openflights. Die Routen-Datei hat den Stand Juni 2014 und enthält eine Zeile pro
Airline und Flugrichtung.

Graph:
    Knoten = Flughäfen mit mindestens einer Route.
    Kanten = ungeordnete Flughafenpaare {u, v}, zwischen denen mindestens eine Airline in irgendeiner Richtung
             fliegt. Im PyG-Graphen steht jedes Paar in beiden Richtungen mit denselben Kantenfeatures.

Knotenfeatures sind über eine Registry wählbar (`OPENFLIGHTS_NODE_FEATURES`, siehe `OpenFlightsConfig`). Bewusst nicht
enthalten sind Größen, die aus den Routen berechnet werden (Grad, Zahl der Airlines am Flughafen): Sie würden
Information über die Val-/Test-Kanten in die Knotenfeatures tragen.

`data.pos` enthält die Flughäfen als Punkte auf der Erdkugel (x, y, z in km). Die euklidische Distanz darin ist
die Sehne und wächst streng monoton mit der Großkreisentfernung. Die räumlichen Negativ-Strategien und die
Luftlinien-Heuristik aus `graph.py`/`evaluation.py` funktionieren damit unverändert.
"""

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

RAW_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/{file}"
DEFAULT_OPENFLIGHTS_DIR = "OpenFlights_raw"
EARTH_RADIUS_KM = 6371.0

AIRPORT_COLUMNS = ["airport_id", "name", "city", "country", "iata", "icao", "lat", "lon", "altitude_ft",
                   "utc_offset", "dst", "tz_database", "type", "source"]
ROUTE_COLUMNS = ["airline", "airline_id", "src_code", "src_id", "dst_code", "dst_id", "codeshare", "stops",
                 "equipment"]
# Kontinent-Präfix der Zeitzonen-Datenbank (z. B. "Europe/Berlin") als grobe Weltregion
REGIONS = ["Africa", "America", "Asia", "Atlantic", "Australia", "Europe", "Indian", "Pacific"]


def ensure_openflights(extract_dir: str = DEFAULT_OPENFLIGHTS_DIR, timeout: float = 60) -> str:
    """Lädt airports.dat und routes.dat, falls sie noch nicht lokal liegen. Gibt den Ordner zurück."""
    os.makedirs(extract_dir, exist_ok=True)
    for file in ("airports.dat", "routes.dat"):
        target = os.path.join(extract_dir, file)
        if os.path.isfile(target):
            continue
        url = RAW_URL.format(file=file)
        print(f"Downloading {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        with open(target, "wb") as f:
            f.write(response.content)
    print(f"Daten liegen in: {extract_dir}")
    return extract_dir


@dataclass
class OpenFlightsFrames:
    """Rohdaten als DataFrames.

    airports: ein Flughafen pro Zeile (nur Flughäfen mit Routen), Spalte `node_idx` = PyG-Knotenindex
    routes:   Rohrouten, eine Zeile pro Airline und Richtung, mit `src_idx`/`dst_idx`
    pairs:    ungeordnete Flughafenpaare (u_idx < v_idx) mit aggregierten Kantenfeatures = Graphkanten
    """

    airports: pd.DataFrame
    routes: pd.DataFrame
    pairs: pd.DataFrame


def _great_circle_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, dtype=float)) for a in (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _sphere_xyz(lat, lon) -> np.ndarray:
    lat, lon = np.radians(np.asarray(lat, dtype=float)), np.radians(np.asarray(lon, dtype=float))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)


def load_openflights(data_dir: str = DEFAULT_OPENFLIGHTS_DIR) -> OpenFlightsFrames:
    """Liest beide Dateien ein, verwirft Routen ohne gültige Flughafen-ID und Selbstschleifen und
    aggregiert die Routen zu ungeordneten Flughafenpaaren."""
    airports = pd.read_csv(os.path.join(data_dir, "airports.dat"), header=None, names=AIRPORT_COLUMNS,
                           na_values="\\N", keep_default_na=False)
    routes = pd.read_csv(os.path.join(data_dir, "routes.dat"), header=None, names=ROUTE_COLUMNS,
                         na_values="\\N", keep_default_na=False)
    n_routes_raw = len(routes)

    routes = routes.dropna(subset=["src_id", "dst_id"]).astype({"src_id": int, "dst_id": int})
    known = set(airports["airport_id"])
    routes = routes[routes["src_id"].isin(known) & routes["dst_id"].isin(known)
                    & (routes["src_id"] != routes["dst_id"])].copy()

    used = np.union1d(routes["src_id"].unique(), routes["dst_id"].unique())
    airports = airports[airports["airport_id"].isin(used)].sort_values("airport_id").reset_index(drop=True)
    airports["node_idx"] = np.arange(len(airports))
    airports["region"] = airports["tz_database"].fillna("").str.split("/").str[0]
    airports.loc[~airports["region"].isin(REGIONS), "region"] = "Other"
    id_to_idx = dict(zip(airports["airport_id"], airports["node_idx"]))
    routes["src_idx"] = routes["src_id"].map(id_to_idx)
    routes["dst_idx"] = routes["dst_id"].map(id_to_idx)

    # Aggregation pro ungeordnetem Paar {u, v}
    routes["u_idx"] = routes[["src_idx", "dst_idx"]].min(axis=1)
    routes["v_idx"] = routes[["src_idx", "dst_idx"]].max(axis=1)
    routes["forward"] = routes["src_idx"] == routes["u_idx"]
    routes["is_codeshare"] = routes["codeshare"].eq("Y")
    routes["equipment_list"] = routes["equipment"].fillna("").str.split()
    grouped = routes.groupby(["u_idx", "v_idx"])
    pairs = grouped.agg(
        n_airlines=("airline", "nunique"),
        n_route_entries=("airline", "size"),
        codeshare_share=("is_codeshare", "mean"),
        has_forward=("forward", "any"),
        has_backward=("forward", lambda s: bool((~s).any())),
    ).reset_index()
    pairs["n_aircraft_types"] = grouped["equipment_list"].agg(lambda lists: len(set().union(*lists))).values
    pairs["bidirectional"] = (pairs["has_forward"] & pairs["has_backward"]).astype(int)
    pairs = pairs.drop(columns=["has_forward", "has_backward"])
    lat, lon = airports["lat"].values, airports["lon"].values
    pairs["distance_km"] = _great_circle_km(lat[pairs["u_idx"]], lon[pairs["u_idx"]],
                                            lat[pairs["v_idx"]], lon[pairs["v_idx"]])

    print(f"OpenFlights: {len(airports)} Flughäfen mit Routen, {n_routes_raw} Routen-Einträge "
          f"({len(routes)} gültig) -> {len(pairs)} ungerichtete Flughafenpaare")
    return OpenFlightsFrames(airports=airports, routes=routes, pairs=pairs)


# ---------------------------------------------------------------------------
# Knotenfeatures (Registry: Name -> Funktion airports-DataFrame -> Feature-Spalten)
# ---------------------------------------------------------------------------

AirportFeatureBuilder = Callable[[pd.DataFrame], pd.DataFrame]
OPENFLIGHTS_NODE_FEATURES: dict[str, AirportFeatureBuilder] = {}


def register_airport_feature(name: str):
    def decorator(fn: AirportFeatureBuilder) -> AirportFeatureBuilder:
        OPENFLIGHTS_NODE_FEATURES[name] = fn
        return fn

    return decorator


@register_airport_feature("coords")
def _coords(airports: pd.DataFrame) -> pd.DataFrame:
    """Position als Einheitsvektor (x, y, z). Anders als (lat, lon) ohne Sprung an der Datumsgrenze."""
    xyz = _sphere_xyz(airports["lat"], airports["lon"])
    return pd.DataFrame(xyz, columns=["sphere_x", "sphere_y", "sphere_z"], index=airports.index)


@register_airport_feature("altitude")
def _altitude(airports: pd.DataFrame) -> pd.DataFrame:
    """Höhe über dem Meer in Fuß, log1p (wenige Flughäfen liegen sehr hoch, negative Höhen -> 0)."""
    alt = airports["altitude_ft"].fillna(0).clip(lower=0)
    return pd.DataFrame({"log_altitude": np.log1p(alt.values)}, index=airports.index)


@register_airport_feature("timezone")
def _timezone(airports: pd.DataFrame) -> pd.DataFrame:
    """UTC-Offset in Stunden."""
    return pd.DataFrame({"utc_offset": airports["utc_offset"].fillna(0).values}, index=airports.index)


@register_airport_feature("region")
def _region(airports: pd.DataFrame) -> pd.DataFrame:
    """One-Hot der Weltregion aus der Zeitzonen-Datenbank (Europe, America, Asia, ...)."""
    cats = REGIONS + ["Other"]
    region = pd.Categorical(airports["region"], categories=cats)
    return pd.get_dummies(region, prefix="region").astype(float).set_index(airports.index)


@dataclass
class OpenFlightsConfig:
    """Steuert, welche Features in data.x und data.edge_attr landen."""

    node_features: list[str] = field(default_factory=lambda: ["coords", "altitude", "timezone", "region"])
    # Kantenfeatures pro Flughafenpaar; numerische Spalten werden log1p-transformiert und standardisiert
    edge_numerical: list[str] = field(
        default_factory=lambda: ["distance_km", "n_airlines", "n_route_entries", "n_aircraft_types"]
    )
    edge_plain: list[str] = field(default_factory=lambda: ["codeshare_share", "bidirectional"])


def build_openflights_graph(frames: OpenFlightsFrames, config: OpenFlightsConfig) -> tuple[Data, list[str], list[str]]:
    """PyG Data-Objekt: jedes Flughafenpaar in beiden Richtungen, gleiche Kantenfeatures für beide.

    Gibt (data, node_feature_columns, edge_attr_columns) zurück. data.pos = Punkte auf der Erdkugel in km.
    """
    unknown = [n for n in config.node_features if n not in OPENFLIGHTS_NODE_FEATURES]
    if unknown:
        raise KeyError(f"Unbekannte Knotenfeatures {unknown}. Verfügbar: {sorted(OPENFLIGHTS_NODE_FEATURES)}")
    airports, pairs = frames.airports, frames.pairs

    node_frame = pd.concat([OPENFLIGHTS_NODE_FEATURES[n](airports) for n in config.node_features], axis=1)
    node_matrix = node_frame.astype(float).values
    one_hot = [i for i, c in enumerate(node_frame.columns) if c.startswith("region_")]
    continuous = [i for i in range(node_matrix.shape[1]) if i not in one_hot]
    if continuous:
        node_matrix[:, continuous] = StandardScaler().fit_transform(node_matrix[:, continuous])

    num = np.log1p(pairs[config.edge_numerical].astype(float).values)
    num = StandardScaler().fit_transform(num) if config.edge_numerical else num
    edge_matrix = np.concatenate([num, pairs[config.edge_plain].astype(float).values], axis=1)
    edge_columns = list(config.edge_numerical) + list(config.edge_plain)

    pair_index = torch.tensor(pairs[["u_idx", "v_idx"]].values.T, dtype=torch.long)
    edge_attr = torch.tensor(edge_matrix, dtype=torch.float)
    edge_index = torch.cat([pair_index, pair_index.flip(0)], dim=1)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

    pos = torch.tensor(_sphere_xyz(airports["lat"], airports["lon"]) * EARTH_RADIUS_KM, dtype=torch.float)
    data = Data(x=torch.tensor(node_matrix, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr,
                pos=pos, num_nodes=len(airports))
    return data, list(node_frame.columns), edge_columns
