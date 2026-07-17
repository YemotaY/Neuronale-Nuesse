# -*- coding: utf-8 -*-
#
#  nn_config.py
#
#  Created by YemotaY on 2026-07-17.
#  MIT Licensed, 2026. All rights reserved.
#
#   b59190308@gmail.com
#
"""Zentrale Konfiguration für Training und Runtime.

Die Werte werden aus ``config.json`` (im selben Verzeichnis) geladen; fehlende
Schlüssel werden mit den Defaults aus ``TrainerConfig`` aufgefüllt. So lassen
sich Bildgröße, Batch-Size, Epochen usw. an einer Stelle ändern.

Beispiel:
    from nn_config import load_config
    cfg = load_config()
    print(cfg.image_size, cfg.batch_size, cfg.epochs)
"""

import json
import os
from dataclasses import dataclass, asdict, fields

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG_PATH = os.path.join(_THIS_DIR, "config.json")


@dataclass
class TrainerConfig:
    """Alle konfigurierbaren Hyperparameter mit sinnvollen Defaults."""

    image_size: tuple = (200, 200)   # (Höhe, Breite) der Modell-Eingabe
    batch_size: int = 20             # Batch-Größe für Training/Validierung
    epochs: int = 40                 # Maximale Anzahl Epochen
    ask_epoch: int = 40              # Epoche, ab der interaktiv gefragt wird
    balance_count: int = 200         # Zielanzahl Bilder je Klasse (Balancing)
    learning_rate: float = 0.001     # Start-Lernrate
    seed: int = 123                  # Reproduzierbarkeit

    def as_dict(self):
        return asdict(self)


def load_config(path: str = _DEFAULT_CONFIG_PATH) -> TrainerConfig:
    """Lädt die Konfiguration aus einer JSON-Datei.

    Existiert die Datei nicht, werden die Defaults zurückgegeben. Unbekannte
    Schlüssel werden ignoriert, fehlende mit Defaults aufgefüllt.
    """
    cfg = TrainerConfig()
    if not os.path.isfile(path):
        return cfg

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"WARNUNG: config.json nicht lesbar ({exc}); nutze Defaults.")
        return cfg

    valid_keys = {f.name for f in fields(TrainerConfig)}
    for key, value in data.items():
        if key not in valid_keys:
            continue
        if key == "image_size" and isinstance(value, (list, tuple)):
            value = tuple(value)
        setattr(cfg, key, value)
    return cfg


def save_config(cfg: TrainerConfig, path: str = _DEFAULT_CONFIG_PATH) -> None:
    """Schreibt die Konfiguration als JSON (z. B. um eine Vorlage zu erzeugen)."""
    data = cfg.as_dict()
    data["image_size"] = list(cfg.image_size)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Schreibt eine Vorlage mit den aktuellen Defaults.
    save_config(TrainerConfig())
    print(f"Vorlage geschrieben nach: {_DEFAULT_CONFIG_PATH}")
