# GNNs_MethodenDerKI

Link Prediction auf dem OpenFlights-Flugroutennetz mit Graph Neural Networks (GATv2 und GINE, PyTorch Geometric).
Knoten sind Flughäfen, Kanten bestehende Flugrouten; die Modelle sollen vorhersagen, ob zwischen zwei Flughäfen eine Route existiert.

## Aufbau

- `data_prep_OpenFlights.ipynb` – Download, Feature Engineering und leckagefreier Train/Val/Test-Split (gespeichert in `processed/`)
- `GATV2_OpenFlights.ipynb`, `GINE_OpenFlights.ipynb` – Training und Auswertung der beiden Modelle über mehrere Seeds
- `Evaluation_OpenFlights.ipynb` – Vergleich der Modelle mit Heuristik-Baselines und Fehleranalyse
- `gnn_pipeline/` – gemeinsamer Code (Datenaufbereitung, Split, Modellbausteine, Trainingsloop, Evaluation)
- `plots/` – Learning Curves, ROC-/PR-Kurven und Modellvergleich
